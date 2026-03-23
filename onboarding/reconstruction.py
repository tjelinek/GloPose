import copy
import logging
import select
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3
from kornia.image import ImageSize
from pycolmap import TwoViewGeometryOptions

logger = logging.getLogger(__name__)
from tqdm import tqdm

from data_providers.flow_provider import MatchingProvider
from data_providers.frame_provider import PrecomputedSegmentationProvider, PrecomputedFrameProvider
from onboarding.colmap_utils import colmap_K_params_vec
from utils.conversions import Se3_to_Rigid3d
from utils.image_utils import get_intrinsics_from_exif


def _load_image_and_segmentation(img_path: Path, seg_path: Path, device: str) \
        -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    image = PrecomputedFrameProvider.load_and_downsample_image(img_path, 1., device).squeeze()
    h, w = image.shape[-2:]
    segmentation = PrecomputedSegmentationProvider.load_and_downsample_segmentation(
        seg_path, ImageSize(h, w), device=device)
    return image, segmentation, h, w


def _match_image_pairs(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                       match_provider: MatchingProvider, match_sample_size: int, device: str,
                       progress=None) \
        -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                 Dict[Tuple[int, int], torch.Tensor],
                 nx.DiGraph, bool]:
    matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}
    view_graph = nx.DiGraph()
    single_camera = True

    for pair_idx, (i1, i2) in tqdm(enumerate(matching_pairs)):
        img1_pth, img2_pth = images[i1], images[i2]
        img1_id = i1 + 1
        img2_id = i2 + 1

        if img1_pth.parent != img2_pth.parent:
            raise ValueError(f"Image pair must be in the same directory: {img1_pth} vs {img2_pth}")

        if progress is not None:
            progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Matching image pairs for reconstruction")

        img1, img1_seg, h1, w1 = _load_image_and_segmentation(img1_pth, segmentations[i1], device)
        img2, img2_seg, h2, w2 = _load_image_and_segmentation(img2_pth, segmentations[i2], device)

        if h1 != h2 or w1 != w2:
            single_camera = False

        src_pts_xy_int, dst_pts_xy_int, certainty = \
            match_provider.get_source_target_points(img1, img2, match_sample_size, img1_seg.squeeze(),
                                                    img2_seg.squeeze(), Path(img1_pth.name),
                                                    Path(img2_pth.name), as_int=True,
                                                    zero_certainty_outside_segmentation=True,
                                                    only_foreground_matches=True)
        view_graph.add_edge(img1_id, img2_id)
        edge = (img1_id, img2_id)
        matching_edges[edge] = (src_pts_xy_int, dst_pts_xy_int)
        matching_edges_certainties[edge] = certainty

    return matching_edges, matching_edges_certainties, view_graph, single_camera


def _merge_tracks(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                  matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                  matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor],
                  view_graph: nx.DiGraph, match_provider: MatchingProvider, device: str,
                  progress=None) \
        -> Tuple[Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                 Dict[Tuple[int, int], torch.Tensor]]:
    for pair_idx, (i1, i2) in tqdm(enumerate(matching_pairs)):
        img1_pth, img2_pth = images[i1], images[i2]
        img1_id = i1 + 1
        img2_id = i2 + 1

        if progress is not None:
            progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Densifying matching...")

        img1, img1_seg, h1, w1 = _load_image_and_segmentation(img1_pth, segmentations[i1], device)
        img2, img2_seg, h2, w2 = _load_image_and_segmentation(img2_pth, segmentations[i2], device)

        previous_matching_pairs = view_graph.in_edges(img1_id)
        src_pts_xy_roma_int_can_be_added = []
        dst_pts_xy_roma_int_can_be_added = []
        certainties_can_be_added = []
        for (edge_u, edge_v) in previous_matching_pairs:
            src_pts_xy_int_nonsampled, dst_pts_xy_int_nonsampled, certainty_nonsampled = \
                match_provider.get_source_target_points(img1, img2, None, img1_seg.squeeze(),
                                                        img2_seg.squeeze(), Path(img1_pth.name),
                                                        Path(img2_pth.name), as_int=True,
                                                        zero_certainty_outside_segmentation=True,
                                                        only_foreground_matches=True)

            prev_match_certain_dst_pts = matching_edges[edge_u, edge_v][1]

            max_coord = max(prev_match_certain_dst_pts.max().item(), src_pts_xy_int_nonsampled.max().item()) + 1
            A_hash = prev_match_certain_dst_pts[:, 0] * max_coord + prev_match_certain_dst_pts[:, 1]
            B_hash = src_pts_xy_int_nonsampled[:, 0] * max_coord + src_pts_xy_int_nonsampled[:, 1]
            mask = torch.isin(B_hash, A_hash)

            src_pts_xy_roma_int_can_be_added.append(src_pts_xy_int_nonsampled[mask])
            dst_pts_xy_roma_int_can_be_added.append(dst_pts_xy_int_nonsampled[mask])
            certainties_can_be_added.append(certainty_nonsampled[mask])

        edge = (img1_id, img2_id)

        src_pts_xy_int = torch.cat([matching_edges[edge][0]] + src_pts_xy_roma_int_can_be_added)
        dst_pts_xy_int = torch.cat([matching_edges[edge][1]] + dst_pts_xy_roma_int_can_be_added)
        certainty = torch.cat([matching_edges_certainties[edge]] + certainties_can_be_added)

        matching_edges[edge] = (src_pts_xy_int, dst_pts_xy_int)
        matching_edges_certainties[edge] = certainty

    return matching_edges, matching_edges_certainties


def _write_colmap_database(images: List[Path],
                           matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                           matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor],
                           single_camera: bool, camera_K: Optional[torch.Tensor],
                           database_path: Path, device: str) -> Path:
    print(f"[pycolmap4-debug] _write_colmap_database: opening {database_path}")
    database = pycolmap.Database.open(str(database_path))

    keypoints, edge_match_indices = unique_keypoints_from_matches(matching_edges, None, matching_edges_certainties,
                                                                  eliminate_one_to_many_matches=True, device=device)

    new_cam_id = 1
    if single_camera:
        if camera_K is None:
            camera_K = get_intrinsics_from_exif(images[0])

        camera_model = pycolmap.CameraModelId.SIMPLE_PINHOLE
        params_vec = colmap_K_params_vec(camera_K, camera_model)

        h, w = PrecomputedFrameProvider.load_and_downsample_image(images[0], 1.).shape[-2:]

        new_camera = pycolmap.Camera(camera_id=new_cam_id, model=camera_model, width=w, height=h, params=params_vec)
        database.write_camera(new_camera, use_camera_id=True)

    for i, img in enumerate(images):
        if not single_camera:
            raise NotImplementedError("To be added")

        img_id = i + 1
        image = pycolmap.Image(image_id=img_id, camera_id=new_cam_id, name=str(img.name))
        database.write_image(image, use_image_id=True)

    for colmap_image_id in sorted(keypoints.keys()):
        keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
        database.write_keypoints(colmap_image_id, keypoints_np)

    for colmap_image_u, colmap_image_v in edge_match_indices.keys():
        match_indices_np = edge_match_indices[colmap_image_u, colmap_image_v].numpy(force=True)
        if match_indices_np.ndim != 2 or match_indices_np.shape[1] != 2:
            continue
        match_indices_np = match_indices_np.astype(np.uint32)
        database.write_matches(colmap_image_u, colmap_image_v, match_indices_np)

    database.close()
    print(f"[pycolmap4-debug] _write_colmap_database: done, wrote {len(keypoints)} image keypoints, "
          f"{len(edge_match_indices)} match pairs")
    return database_path


def reconstruct_images_using_sfm(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                                 init_with_first_two_images: bool, mapper: str, match_provider: MatchingProvider,
                                 match_sample_size: int, colmap_working_dir: Path, add_track_merging_matches: bool,
                                 camera_K: Optional[torch.Tensor] = None, device: str = 'cpu',
                                 progress=None, filter_points_by_seg: bool = False) \
        -> Tuple[Optional[pycolmap.Reconstruction], int]:
    if len(matching_pairs) == 0:
        raise ValueError("Needed at least 1 match.")
    if len(images) == 0:
        raise ValueError("No images provided for SfM reconstruction")

    database_path = colmap_working_dir / 'database.db'
    colmap_output_path = colmap_working_dir / 'output'
    colmap_image_path = colmap_working_dir / 'images'

    if database_path.exists():
        raise FileExistsError(f"COLMAP database already exists: {database_path}")

    matching_pairs = sorted(matching_pairs)

    matching_edges, matching_edges_certainties, view_graph, single_camera = _match_image_pairs(
        images, segmentations, matching_pairs, match_provider, match_sample_size, device, progress)

    if add_track_merging_matches:
        matching_edges, matching_edges_certainties = _merge_tracks(
            images, segmentations, matching_pairs, matching_edges, matching_edges_certainties,
            view_graph, match_provider, device, progress)

    _write_colmap_database(images, matching_edges, matching_edges_certainties, single_camera, camera_K,
                           database_path, device)
    shutil.copy(database_path, database_path.parent / 'database_before_ransac.db')

    two_view_geometry(database_path)

    shutil.copy(database_path, database_path.parent / 'database_after_ransac_before_rec.db')

    first_image_id = None
    second_image_id = None
    if init_with_first_two_images:
        first_image_id = 1
        second_image_id = 2

    if progress is not None:
        progress(0.5, desc="Running reconstruction...")

    ignore_two_view_tracks = not add_track_merging_matches
    num_reconstructions = run_mapper(colmap_output_path, database_path, colmap_image_path, mapper, first_image_id,
                                     second_image_id, ignore_two_view_tracks)

    if progress is not None:
        progress(1.0, desc="Reconstruction finished.")

    path_to_rec = colmap_output_path / '0'
    print(f"[pycolmap4-debug] reconstruct_images_using_sfm: loading reconstruction from {path_to_rec}")
    try:
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        print(f"[pycolmap4-debug] reconstruct_images_using_sfm: loaded OK")
        print(reconstruction.summary())
    except Exception as e:
        print(f"[pycolmap4-debug] reconstruct_images_using_sfm: load FAILED: {e}")
        return None, num_reconstructions

    if filter_points_by_seg:
        reconstruction = filter_points_by_segmentation(reconstruction, segmentations, images)

    return reconstruction, num_reconstructions


def filter_points_by_segmentation(reconstruction: pycolmap.Reconstruction,
                                   segmentations: List[Path],
                                   images: List[Path]) -> pycolmap.Reconstruction:
    """Remove 3D points whose 2D observation falls outside the segmentation mask in any image.
    Then run bundle adjustment on the cleaned reconstruction."""

    # Build mapping: image name -> segmentation path
    image_name_to_seg: Dict[str, Path] = {}
    for img_path, seg_path in zip(images, segmentations):
        image_name_to_seg[img_path.name] = seg_path

    # Load and cache segmentation masks as numpy arrays (H, W), values 0-255
    seg_cache: Dict[str, np.ndarray] = {}

    def get_seg_mask(image_name: str) -> np.ndarray | None:
        if image_name in seg_cache:
            return seg_cache[image_name]
        seg_path = image_name_to_seg.get(image_name)
        if seg_path is None or not seg_path.exists():
            return None
        import imageio
        mask = imageio.v3.imread(seg_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        seg_cache[image_name] = mask
        return mask

    # Collect point3D IDs to delete
    point3d_ids_to_delete = set()
    for point3d_id, point3d in reconstruction.points3D.items():
        for track_element in point3d.track.elements:
            image_id = track_element.image_id
            point2d_idx = track_element.point2D_idx

            image = reconstruction.images[image_id]
            point2d = image.points2D[point2d_idx]
            x, y = point2d.xy.astype(int)

            mask = get_seg_mask(image.name)
            if mask is None:
                continue

            h, w = mask.shape
            if y < 0 or y >= h or x < 0 or x >= w or mask[y, x] < 128:
                point3d_ids_to_delete.add(point3d_id)
                break

    num_before = len(reconstruction.points3D)
    for point3d_id in point3d_ids_to_delete:
        reconstruction.delete_point3D(point3d_id)
    num_after = len(reconstruction.points3D)
    print(f"Segmentation filtering: removed {num_before - num_after}/{num_before} points, "
          f"{num_after} remaining")

    # Run bundle adjustment on the cleaned reconstruction
    ba_options = pycolmap.BundleAdjustmentOptions()
    pycolmap.bundle_adjustment(reconstruction, ba_options)

    return reconstruction


def align_with_kabsch(reconstruction: pycolmap.Reconstruction, gt_Se3_world2cam_poses: Dict[str, Se3]) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    print(f"[pycolmap4-debug] align_with_kabsch: starting (deepcopy)")
    reconstruction = copy.deepcopy(reconstruction)

    gt_camera_centers = []
    pred_camera_centers = []

    for image_name, gt_Se3_world2cam in gt_Se3_world2cam_poses.items():

        pred_image = reconstruction.find_image_with_name(image_name)
        if pred_image is None:
            continue

        gt_cam_center = gt_Se3_world2cam.inverse().translation.numpy(force=True)
        pred_cam_center = pred_image.cam_from_world().inverse().translation

        gt_camera_centers.append(gt_cam_center)
        pred_camera_centers.append(pred_cam_center)

    if len(gt_camera_centers) < 3:
        print(f"[pycolmap4-debug] align_with_kabsch: too few matched cameras ({len(gt_camera_centers)}), skipping")
        return reconstruction, False

    gt_camera_centers = np.stack(gt_camera_centers)
    pred_camera_centers = np.stack(pred_camera_centers)

    sim3d_report = pycolmap.estimate_sim3d_robust(pred_camera_centers, gt_camera_centers)
    if sim3d_report is not None:
        sim3d = sim3d_report['tgt_from_src']
    else:
        sim3d = pycolmap.estimate_sim3d(pred_camera_centers, gt_camera_centers)

    if sim3d is None:
        print(f"[pycolmap4-debug] align_with_kabsch: Sim3d estimation failed")
        return reconstruction, False

    print(f"[pycolmap4-debug] align_with_kabsch: applying transform")
    reconstruction.transform(sim3d)
    print(f"[pycolmap4-debug] align_with_kabsch: done")

    return reconstruction, True


def two_view_geometry(colmap_db_path: Path):
    print(f"[pycolmap4-debug] two_view_geometry: starting match_exhaustive")
    opts = TwoViewGeometryOptions()
    opts.detect_watermark = False
    pycolmap.match_exhaustive(str(colmap_db_path), verification_options=opts)
    print(f"[pycolmap4-debug] two_view_geometry: done")


def run_mapper(colmap_output_path: Path, colmap_db_path: Path, colmap_image_path: Path, mapper: str = 'pycolmap',
               first_image_id: Optional[int] = None, second_image_id: Optional[int] = None,
               ignore_two_view_tracks: bool = True) -> int:
    """Run COLMAP/glomap mapper. Returns the number of reconstructions produced."""
    colmap_output_path.mkdir(exist_ok=True, parents=True)

    initial_pair_provided = first_image_id is not None and second_image_id is not None
    if mapper in ['colmap', 'glomap']:
        if mapper == 'glomap':

            import platform
            import os

            hostname = platform.node()
            binary = "glomap_amd" if hostname.startswith("g") else "glomap"
            glomap_path = os.path.expanduser(f"~/bin/{binary}")

            command = [
                f"{glomap_path}",
                "mapper",
                "--database_path", str(colmap_db_path),
                "--output_path", str(colmap_output_path),
                "--image_path", str(colmap_image_path),
                "--TrackEstablishment.min_num_view_per_track", str(3 if ignore_two_view_tracks else 2),
            ]

        elif mapper == 'colmap':
            command = [
                "colmap",
                "mapper",
                "--database_path", str(colmap_db_path),
                "--output_path", str(colmap_output_path),
                "--image_path", str(colmap_image_path),
                "--Mapper.tri_ignore_two_view_tracks", str(int(ignore_two_view_tracks)),
                *("--Mapper.init_image_id1", str(first_image_id) if initial_pair_provided else ""),
                *("--Mapper.init_image_id2", str(second_image_id) if initial_pair_provided else ""),
                "--log_to_stderr", str(1),
            ]
        else:
            raise ValueError("This code should not ve reachable")

        with subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
        ) as process:
            fds = [process.stdout.fileno(), process.stderr.fileno()]
            while True:
                ready_fds, _, _ = select.select(fds, [], [])
                for fd in ready_fds:
                    if fd == process.stdout.fileno():
                        line = process.stdout.readline()
                        if line:
                            print(f"STDOUT: {line.strip()}")
                    elif fd == process.stderr.fileno():
                        line = process.stderr.readline()
                        if line:
                            print(f"STDERR: {line.strip()}")
                if process.poll() is not None:
                    break

            process.wait()
            if process.returncode != 0:
                error_message = process.stderr.read()
                print(f"Error: {error_message}")
                raise subprocess.CalledProcessError(process.returncode, command, output=None, stderr=error_message)

        # Count output reconstruction directories
        num_recs = sum(1 for d in colmap_output_path.iterdir() if d.is_dir() and d.name.isdigit())
        return num_recs

    elif mapper == 'pycolmap':

        opts = pycolmap.IncrementalPipelineOptions()
        opts.triangulation.ignore_two_view_tracks = ignore_two_view_tracks
        opts.triangulation.max_transitivity = 2
        # opts.mapper.ba_local_num_images = 3
        # opts.ba_global_frames_freq = 3
        if initial_pair_provided:
            opts.init_image_id1 = first_image_id
            opts.init_image_id2 = second_image_id

        print(f"[pycolmap4-debug] run_mapper: starting incremental_mapping")
        maps = pycolmap.incremental_mapping(str(colmap_db_path), str(colmap_image_path), str(colmap_output_path),
                                            options=opts)
        print(f"[pycolmap4-debug] run_mapper: incremental_mapping returned {len(maps)} maps")
        if len(maps) > 0:
            # Pick the largest reconstruction (most registered images)
            best = max(maps.values(), key=lambda r: r.num_reg_images())
            if len(maps) > 1:
                sizes = [r.num_reg_images() for r in maps.values()]
                logger.warning("COLMAP produced %d reconstructions (sizes: %s), using largest (%d images)",
                               len(maps), sizes, best.num_reg_images())
            print(f"[pycolmap4-debug] run_mapper: writing best map ({best.num_reg_images()} images) to {colmap_output_path}")
            best.write(str(colmap_output_path))
            print(f"[pycolmap4-debug] run_mapper: write done")
        return len(maps)
    else:
        raise ValueError(f"Need to run either glomap or colmap, got mapper={mapper}")


def get_match_points_indices(keypoints, match_pts):
    N = keypoints.shape[0]
    keypoints_and_match_pts = torch.cat([keypoints, match_pts], dim=0)
    _, kpts_and_match_pts_indices = torch.unique(keypoints_and_match_pts, return_inverse=True, dim=0)

    if kpts_and_match_pts_indices.max() >= N:
        raise ValueError("Not all src_pts included in keypoints")
    assert torch.equal(keypoints[kpts_and_match_pts_indices[:N]], keypoints)
    match_pts_indices = kpts_and_match_pts_indices[N:]
    return match_pts_indices


def keypoints_unique_preserve_order(keypoints: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    first_kpt_idx_occurrence, inverse_indices = get_first_occurrence_indices(keypoints, dim=0)
    first_kpt_idx_occurrence_sorted, index_sort_permutation = torch.sort(first_kpt_idx_occurrence)
    unique_order_preserving = keypoints[first_kpt_idx_occurrence_sorted]

    idx_mapping = torch.zeros(len(index_sort_permutation), dtype=torch.long).to(keypoints.device)

    # Use scatter_ to place indices at the positions specified by index_sort_permutation
    # This creates our mapping from original positions to new positions after reordering
    idx_mapping.scatter_(0, index_sort_permutation, torch.arange(len(index_sort_permutation), device=keypoints.device))

    # Apply the mapping to get correct inverse indices
    inverse_indices_order_preserving = idx_mapping[inverse_indices]

    assert torch.all(torch.eq(keypoints, unique_order_preserving[inverse_indices_order_preserving]).view(-1))

    return unique_order_preserving, inverse_indices_order_preserving


def unique_keypoints_from_matches(matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                                  existing_database: pycolmap.Database = None,
                                  matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = None,
                                  eliminate_one_to_many_matches: bool = True, device: str = 'cpu') -> (
        Tuple)[Dict[int, torch.Tensor], Dict[Tuple[int, int], torch.Tensor]]:
    G = nx.DiGraph()
    G.add_edges_from(matching_edges.keys())

    existing_database_image_ids = []
    if existing_database is not None:
        existing_database_image_ids = [img.image_id for img in existing_database.read_all_images()]

    keypoints_for_node: Dict[int, torch.Tensor] = {}
    edge_match_indices: Dict[Tuple[int, int, int], torch.Tensor] = defaultdict(lambda: torch.zeros(0, ).to(device))

    for u in G.nodes():

        incoming_edges = list(G.in_edges(u))
        outgoing_edges = list(G.out_edges(u))

        if u in existing_database_image_ids and existing_database.read_keypoints(u).shape[0] > 0:
            existing_keypoints_u = [torch.from_numpy(existing_database.read_keypoints(u)).to(device)]
        else:
            existing_keypoints_u = [torch.zeros((0, 2)).to(torch.int).to(device)]
        existing_keypoints_lengths = [existing_keypoints_u[0].shape[0]]

        keypoints_u_incoming_list = [matching_edges[v, u][1] for v, _ in incoming_edges]
        keypoints_u_incoming_list_lengths = [matching_edges[v, u][1].shape[0] for v, _ in incoming_edges]
        keypoints_u_outgoing_list = [matching_edges[u, v][0] for _, v in outgoing_edges]
        keypoints_u_outgoing_list_lengths = [matching_edges[u, v][0].shape[0] for _, v in outgoing_edges]

        keypoints_u_all_lists = existing_keypoints_u + keypoints_u_incoming_list + keypoints_u_outgoing_list
        keypoints_u_all = torch.cat(keypoints_u_all_lists)

        keypoints_u_unique, match_indices_order_preserving = keypoints_unique_preserve_order(keypoints_u_all)

        num_existing = existing_keypoints_lengths[0]
        num_incoming = int(np.sum(keypoints_u_incoming_list_lengths))
        num_outgoing = int(np.sum(keypoints_u_outgoing_list_lengths))
        match_indices_sizes = [num_existing, num_incoming, num_outgoing]
        match_indices_delimiters = np.cumsum(match_indices_sizes)

        match_indices_existing, match_indices_incoming, match_indices_outgoing = (
            torch.split(match_indices_order_preserving, match_indices_sizes))

        if match_indices_incoming.shape[0] > 0:
            keypoints_matches_incoming_indices = match_indices_order_preserving[match_indices_delimiters[0]:
                                                                                match_indices_delimiters[1]]

            keypoints_matches_incoming_indices_split = torch.split(keypoints_matches_incoming_indices,
                                                                   keypoints_u_incoming_list_lengths)

            for i, (v, _) in enumerate(incoming_edges):
                # Handle if both (u, v), and (v, u) exists
                edge_match_indices[v, u, 1] = torch.cat([edge_match_indices[v, u, 1],
                                                         keypoints_matches_incoming_indices_split[i]], dim=0)

        if match_indices_outgoing.shape[0] > 0:
            keypoints_matches_outgoing_indices = match_indices_order_preserving[match_indices_delimiters[1]:]
            keypoints_matches_outgoing_indices_split = torch.split(keypoints_matches_outgoing_indices,
                                                                   keypoints_u_outgoing_list_lengths)

            for i, (_, v) in enumerate(outgoing_edges):
                # Handle if both (u, v), and (v, u) exists
                edge_match_indices[u, v, 0] = torch.cat([edge_match_indices[u, v, 0],
                                                         keypoints_matches_outgoing_indices_split[i]], dim=0)

        keypoints_for_node[u] = keypoints_u_unique

    edge_match_indices_concatenated = {}
    for u, v in G.to_undirected().edges():

        keypoints_indices_u = edge_match_indices[u, v, 0]
        keypoints_indices_v = edge_match_indices[u, v, 1]

        if eliminate_one_to_many_matches:
            if matching_edges_certainties is not None:
                certainty = matching_edges_certainties[u, v]
            else:
                certainty = torch.zeros(keypoints_indices_v.shape[0], device=device)

            certainty_sort_idx = torch.argsort(certainty, descending=True)
            keypoints_indices_u_sorted = keypoints_indices_u[certainty_sort_idx]
            keypoints_indices_v_sorted = keypoints_indices_v[certainty_sort_idx]

            unique_keypoints_indices_u, _ = get_first_occurrence_indices(keypoints_indices_u_sorted)
            unique_keypoints_indices_v, _ = get_first_occurrence_indices(keypoints_indices_v_sorted)

            unique_keypoints_mask_u = torch.zeros_like(keypoints_indices_u, device=device, dtype=torch.bool)
            unique_keypoints_mask_v = torch.zeros_like(keypoints_indices_v, device=device, dtype=torch.bool)

            unique_keypoints_mask_u[unique_keypoints_indices_u] = True
            unique_keypoints_mask_v[unique_keypoints_indices_v] = True

            ono_to_one_mask = unique_keypoints_mask_u & unique_keypoints_mask_v

            keypoints_indices_u = keypoints_indices_u_sorted[ono_to_one_mask]
            keypoints_indices_v = keypoints_indices_v_sorted[ono_to_one_mask]

        stacked_indices = torch.stack([keypoints_indices_u, keypoints_indices_v], dim=1)

        edge_match_indices_concatenated[(u, v)] = stacked_indices

    return keypoints_for_node, edge_match_indices_concatenated


def get_first_occurrence_indices(elements: torch.Tensor, dim: Optional[int] = None) \
        -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find the indices where each unique element first appears in the input tensor. Unlike torch.unique, this function
    guarantees ordering.

    This function identifies all unique elements in the input tensor and returns:
    1. The indices where each unique element first appears in the original tensor
    2. A mapping from each position in the original tensor to its corresponding unique element index

    Args:
        elements: Input tensor containing potentially duplicate elements
        dim: The dimension along which to find unique elements, if elements is multi-dimensional

    Returns:
        Tuple containing:
        - first_occurrence_indices: Indices where each unique element first appears
        - element_to_unique_mapping: Mapping from original positions to unique element indices
    """
    # Find unique elements and get mapping information
    unique_elements, element_to_unique_mapping, occurrence_counts = torch.unique(
        elements,
        sorted=True,
        dim=dim,
        return_inverse=True,
        return_counts=True
    )

    # Get indices that would sort the mapping array while preserving order of equal elements
    _, sorted_positions = torch.sort(element_to_unique_mapping, stable=True)

    # Calculate cumulative occurrence counts
    cumulative_counts = occurrence_counts.cumsum(0)

    # Shift cumulative counts to get starting positions of each unique value
    N = occurrence_counts.size(0)
    zero = torch.tensor([0], device=cumulative_counts.device, dtype=cumulative_counts.dtype)
    starting_positions = torch.cat((zero, cumulative_counts[:-1]))[:N]  # [:N] when called the function on empty Tensor

    # first-occurrence indices
    first_occurrence_indices = sorted_positions[starting_positions]

    return first_occurrence_indices, element_to_unique_mapping


def align_reconstruction_with_pose(reconstruction: pycolmap.Reconstruction, first_image_gt_Se3_world2cam: Se3,
                                   image_depths: Dict[str, torch.Tensor], first_image_name: str) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    reconstruction = copy.deepcopy(reconstruction)

    if not (first_image_colmap := reconstruction.find_image_with_name(first_image_name)):
        print("Alignment error. The 1st image was not registered.")
        return reconstruction, False

    first_image = reconstruction.find_image_with_name(first_image_name)
    first_image_name = first_image.name
    cam_from_world = first_image.cam_from_world()

    pred_first_image_point_depths = []
    gt_first_image_point_depths = []

    for point2D in first_image.points2D:
        if point2D.has_point3D():
            point2D_x, point2D_y = point2D.xy.astype('int')

            point3D_id = point2D.point3D_id
            point3D = reconstruction.point3D(point3D_id)

            point3D_cam = cam_from_world * point3D.xyz  # Transform 3D point from world to camera coordinates
            depth_pred = point3D_cam[2]
            depth_gt = image_depths[first_image_name][point2D_y, point2D_x].item()

            pred_first_image_point_depths.append(depth_pred)
            gt_first_image_point_depths.append(depth_gt)

    pred_first_image_point_depths = np.asarray(pred_first_image_point_depths)
    gt_first_image_point_depths = np.asarray(gt_first_image_point_depths)

    scale = np.median(gt_first_image_point_depths) / np.median(pred_first_image_point_depths)

    colmap_cam_from_world = first_image_colmap.cam_from_world()
    gt_cam_from_world = Se3_to_Rigid3d(first_image_gt_Se3_world2cam)
    gt_world_from_cam = gt_cam_from_world.inverse()

    colmap_world_from_cam = colmap_cam_from_world.inverse()  # world_from_cam

    gt_world_from_cam_scaled = pycolmap.Sim3d(
        scale=1.0 / scale,
        rotation=gt_world_from_cam.rotation,
        translation=gt_world_from_cam.translation
    )

    colmap_world_from_cam_sim3d = pycolmap.Sim3d(
        scale=1.0,
        rotation=colmap_world_from_cam.rotation,
        translation=colmap_world_from_cam.translation
    )

    Sim3d_first_image_colmap2gt = gt_world_from_cam_scaled * colmap_world_from_cam_sim3d.inverse()

    reconstruction.transform(Sim3d_first_image_colmap2gt)

    return reconstruction, True


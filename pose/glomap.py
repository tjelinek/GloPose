import copy
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
from torchvision.transforms import functional as TF
from tqdm import tqdm

from data_providers.flow_provider import FlowProviderDirect
from data_providers.frame_provider import PrecomputedSegmentationProvider, PrecomputedFrameProvider
from data_structures.view_graph import ViewGraph
from pose.colmap_utils import colmap_K_params_vec
from pose.frame_filter import compute_matching_reliability
from utils.conversions import Se3_to_Rigid3d
from utils.image_utils import get_intrinsics_from_exif
from visualizations.pose_estimation_visualizations import PoseEstimatorLogger


def reconstruct_images_using_sfm(images: List[Path], segmentations: List[Path], matching_pairs: List[Tuple[int, int]],
                                 init_with_first_two_images: bool, mapper: str, match_provider: FlowProviderDirect,
                                 match_sample_size: int, colmap_working_dir: Path, add_track_merging_matches: bool,
                                 camera_K: Optional[torch.Tensor] = None, device: str = 'cpu',
                                 progress=None) \
        -> Optional[pycolmap.Reconstruction]:
    if len(matching_pairs) == 0:
        raise ValueError("Needed at least 1 match.")

    database_path = colmap_working_dir / 'database.db'
    colmap_output_path = colmap_working_dir / 'output'
    colmap_image_path = colmap_working_dir / 'images'

    matching_pairs = sorted(matching_pairs)
    image_pairs = [(images[i1], images[i2]) for i1, i2 in matching_pairs]
    segmentation_pairs = [(segmentations[i1], segmentations[i2]) for i1, i2 in matching_pairs]

    assert not database_path.exists()

    single_camera = True
    assert len(images) > 0

    matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}

    view_graph = nx.DiGraph()

    for pair_idx, ((img1_pth, img2_pth), (seg1_pth, seg2_pth)) in tqdm(enumerate(zip(image_pairs, segmentation_pairs))):

        img1_pth = Path(img1_pth)
        img2_pth = Path(img2_pth)
        img1_id = images.index(img1_pth) + 1
        img2_id = images.index(img2_pth) + 1

        assert img1_pth.parent == img2_pth.parent

        if progress is not None:
            progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Matching image pairs for reconstruction")

        img1 = PrecomputedFrameProvider.load_and_downsample_image(img1_pth, 1., device).squeeze()
        img2 = PrecomputedFrameProvider.load_and_downsample_image(img2_pth, 1., device).squeeze()

        h1, w1 = img1.shape[-2:]
        h2, w2 = img2.shape[-2:]
        if h1 != h2 or w1 != w2:
            single_camera = False

        seg1_size = ImageSize(h1, w1)
        seg2_size = ImageSize(h2, w2)
        img1_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg1_pth, seg1_size, device=device)
        img2_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg2_pth, seg2_size, device=device)

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

    if add_track_merging_matches:
        for pair_idx, ((img1_pth, img2_pth), (seg1_pth, seg2_pth)) in \
                tqdm(enumerate(zip(image_pairs, segmentation_pairs))):

            img1_pth = Path(img1_pth)
            img2_pth = Path(img2_pth)
            img1_id = images.index(img1_pth) + 1
            img2_id = images.index(img2_pth) + 1

            if progress is not None:
                progress(0.5 * pair_idx / float(len(matching_pairs)), desc="Densifying matching...")

            img1 = PrecomputedFrameProvider.load_and_downsample_image(img1_pth, 1., device).squeeze()
            img2 = PrecomputedFrameProvider.load_and_downsample_image(img2_pth, 1., device).squeeze()
            h1, w1 = img1.shape[-2:]
            h2, w2 = img2.shape[-2:]
            seg1_size = ImageSize(h1, w1)
            seg2_size = ImageSize(h2, w2)
            img1_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg1_pth, seg1_size,
                                                                                        device=device)
            img2_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg2_pth, seg2_size,
                                                                                        device=device)

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

                A_set = set(map(tuple, prev_match_certain_dst_pts.cpu().numpy()))
                B_tuples = [tuple(row) for row in src_pts_xy_int_nonsampled.cpu().numpy()]

                mask = torch.tensor([tuple_b in A_set for tuple_b in B_tuples], dtype=torch.bool, device=device)

                src_pts_xy_roma_int_can_be_added_u = src_pts_xy_int_nonsampled[mask]
                dst_pts_xy_roma_int_can_be_added_v = dst_pts_xy_int_nonsampled[mask]
                certainties_can_be_added_this_match = certainty_nonsampled[mask]

                src_pts_xy_roma_int_can_be_added.append(src_pts_xy_roma_int_can_be_added_u)
                dst_pts_xy_roma_int_can_be_added.append(dst_pts_xy_roma_int_can_be_added_v)
                certainties_can_be_added.append(certainties_can_be_added_this_match)

            edge = (img1_id, img2_id)

            src_pts_xy_int = torch.cat([matching_edges[edge][0]] + src_pts_xy_roma_int_can_be_added)
            dst_pts_xy_int = torch.cat([matching_edges[edge][1]] + dst_pts_xy_roma_int_can_be_added)
            certainty = torch.cat([matching_edges_certainties[edge]] + certainties_can_be_added)

            matching_edges[edge] = (src_pts_xy_int, dst_pts_xy_int)
            matching_edges_certainties[edge] = certainty

    database = pycolmap.Database(str(database_path))

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
        database.write_matches(colmap_image_u, colmap_image_v, match_indices_np)

    database.close()
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
    run_mapper(colmap_output_path, database_path, colmap_image_path, mapper, first_image_id, second_image_id,
               ignore_two_view_tracks)

    if progress is not None:
        progress(1.0, desc="Reconstruction finished.")

    path_to_rec = colmap_output_path / '0'
    try:
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        print(reconstruction.summary())
    except Exception as e:
        print(e)
        return None

    return reconstruction


def align_with_kabsch(reconstruction: pycolmap.Reconstruction, gt_Se3_world2cam_poses: Dict[str, Se3]) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    reconstruction = copy.deepcopy(reconstruction)

    gt_camera_centers = []
    pred_camera_centers = []

    for image_name, gt_Se3_world2cam in gt_Se3_world2cam_poses.items():

        pred_image = reconstruction.find_image_with_name(image_name)
        if pred_image is None:
            continue

        gt_cam_center = gt_Se3_world2cam.inverse().translation.numpy(force=True)
        pred_cam_center = pred_image.cam_from_world.inverse().translation

        gt_camera_centers.append(gt_cam_center)
        pred_camera_centers.append(pred_cam_center)

    gt_camera_centers = np.stack(gt_camera_centers)
    pred_camera_centers = np.stack(pred_camera_centers)

    sim3d_report = pycolmap.estimate_sim3d_robust(pred_camera_centers, gt_camera_centers)
    if sim3d_report is not None:
        sim3d = sim3d_report['tgt_from_src']
    else:
        sim3d = pycolmap.estimate_sim3d(pred_camera_centers, gt_camera_centers)

    reconstruction.transform(sim3d)

    return reconstruction, True


def two_view_geometry(colmap_db_path: Path):
    opts = TwoViewGeometryOptions()
    opts.detect_watermark = False
    pycolmap.match_exhaustive(str(colmap_db_path), verification_options=opts)


def run_mapper(colmap_output_path: Path, colmap_db_path: Path, colmap_image_path: Path, mapper: str = 'pycolmap',
               first_image_id: Optional[int] = None, second_image_id: Optional[int] = None,
               ignore_two_view_tracks: bool = True):
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

    elif mapper == 'pycolmap':

        opts = pycolmap.IncrementalPipelineOptions()
        opts.triangulation.ignore_two_view_tracks = ignore_two_view_tracks
        opts.ba_local_num_images = 3
        opts.ba_global_images_freq = 3
        opts.triangulation.max_transitivity = 2
        if initial_pair_provided:
            opts.init_image_id1 = first_image_id
            opts.init_image_id2 = second_image_id

        maps = pycolmap.incremental_mapping(str(colmap_db_path), str(colmap_image_path), str(colmap_output_path),
                                            options=opts)
        if len(maps) > 0:
            maps[0].write(str(colmap_output_path))
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


def test_sim3d_equivalence(sim3d_new, sim3d_old):
    """Test if the two Sim3D construction methods are equivalent."""

    # Compare components
    print("=== COMPARISON ===")
    print(f"Scale - Old: {sim3d_old.scale:.6f}, New: {sim3d_new.scale:.6f}")
    print(f"Scale diff: {abs(sim3d_old.scale - sim3d_new.scale):.2e}")

    # Compare rotations (quaternions)
    old_quat = sim3d_old.rotation.quat
    new_quat = sim3d_new.rotation.quat
    quat_diff = np.linalg.norm(old_quat - new_quat)
    print(f"Rotation diff (quat norm): {quat_diff:.2e}")

    # Compare translations
    trans_diff = np.linalg.norm(sim3d_old.translation - sim3d_new.translation)
    print(f"Translation diff (norm): {trans_diff:.2e}")

    # Test on a sample point
    test_point = np.array([1.0, 2.0, 3.0])
    point_old = sim3d_old * test_point
    point_new = sim3d_new * test_point
    point_diff = np.linalg.norm(point_old - point_new)
    print(f"Point transformation diff: {point_diff:.2e}")

    # Test on camera pose
    test_cam = pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(),
        translation=np.array([0.0, 0.0, 5.0])
    )
    cam_old = sim3d_old.transform_camera_world(test_cam)
    cam_new = sim3d_new.transform_camera_world(test_cam)
    cam_trans_diff = np.linalg.norm(cam_old.translation - cam_new.translation)
    cam_quat_diff = np.linalg.norm(cam_old.rotation.quat - cam_new.rotation.quat)
    print(f"Camera translation diff: {cam_trans_diff:.2e}")
    print(f"Camera rotation diff: {cam_quat_diff:.2e}")

    # Overall equivalence check
    is_equivalent = (
            abs(sim3d_old.scale - sim3d_new.scale) < 1e-8 and
            quat_diff < 1e-8 and
            trans_diff < 1e-8
    )

    print(f"\n*** EQUIVALENT: {is_equivalent} ***")
    return is_equivalent


def align_reconstruction_with_pose(reconstruction: pycolmap.Reconstruction, first_image_gt_Se3_world2cam: Se3,
                                   image_depths: Dict[str, torch.Tensor], first_image_name: str) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    reconstruction = copy.deepcopy(reconstruction)

    if not (first_image_colmap := reconstruction.find_image_with_name(first_image_name)):
        print("Alignment error. The 1st image was not registered.")
        return reconstruction, False

    first_image = reconstruction.find_image_with_name(first_image_name)
    first_image_name = first_image.name
    cam_from_world = first_image.cam_from_world

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

    # points_xyz = []
    # for point3D_id, point3D in reconstruction.points3D.items():
    #     points_xyz.append(point3D.xyz)
    # points_xyz = np.stack(points_xyz)
    #
    # points_min = points_xyz.min(axis=0)
    # points_max = points_xyz.max(axis=0)
    #
    # print(f'Points orig min {points_min}')
    # print(f'Points orig max {points_max}')
    # print(f'Points orig range {points_max - points_min}')

    scale = np.median(gt_first_image_point_depths) / np.median(pred_first_image_point_depths)

    colmap_cam_from_world = first_image_colmap.cam_from_world
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

    # Sim3d_first_image_colmap2gt_old = pycolmap.Sim3d(
    #     scale=1.0 / scale,
    #     rotation=gt_world_from_cam.rotation * colmap_world_from_cam.rotation.inverse(),
    #     translation=gt_world_from_cam.translation - scale * (
    #         gt_world_from_cam.rotation.matrix() @ colmap_world_from_cam.translation))
    #
    # test_sim3d_equivalence(Sim3d_first_image_colmap2gt, Sim3d_first_image_colmap2gt_old)

    for point3D_id, point3D in reconstruction.points3D.items():
        point3D.xyz = Sim3d_first_image_colmap2gt * point3D.xyz

    for image_id, image in reconstruction.images.items():
        if image.has_pose:
            image.cam_from_world = Sim3d_first_image_colmap2gt.transform_camera_world(image.cam_from_world)

    # print("Pose alignment check:")
    # print(f'Scale: {scale}')
    # print(f"GT pose: {gt_cam_from_world}")
    # print(f"Rec pose: {reconstruction.find_image_with_name(first_image_name).cam_from_world}")
    #
    # for image_id, image in reconstruction.images.items():
    #     if image.has_pose:
    #         print(f'{image_id} pose {image.cam_from_world}')
    # points_xyz = []
    # for point3D_id, point3D in reconstruction.points3D.items():
    #     points_xyz.append(point3D.xyz)
    # points_xyz = np.stack(points_xyz)
    #
    # points_min = points_xyz.min(axis=0)
    # points_max = points_xyz.max(axis=0)
    #
    # print(f'Points transformed min {points_min}')
    # print(f'Points transformed max {points_max}')
    # print(f'Points transformed range {points_max - points_min}')

    return reconstruction, True


if __name__ == '__main__':
    im_pairs = [
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000001.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000056.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000056.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000172.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000172.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000202.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000202.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000255.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000255.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000308.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000308.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000453.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000453.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000482.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000482.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000565.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000565.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000622.jpg')),
        (Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000622.jpg'),
         Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000020/glomap_obj_000020/images/000706.jpg'))]

    colmap_db_path_ = Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000010/glomap_obj_000010/database.db')
    colmap_img_path_ = Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000010/glomap_obj_000010/images')
    colmap_output_path_ = Path('/mnt/personal/jelint19/results/FlowTracker/handal/obj_000010/glomap_obj_000010/output')

    print("COLMAP Python bindings path:", pycolmap.__file__)
    # time.sleep(20)
    # run_custom_matcher(colmap_output_path_, colmap_db_path_, colmap_img_path_, im_pairs)

    # run_mapper(colmap_output_path_, colmap_db_path_, colmap_img_path_, mapper = 'glomap')
    two_view_geometry(colmap_db_path_)


def predict_poses(query_img: torch.Tensor, camera_K: np.ndarray, view_graph: ViewGraph,
                  flow_provider: FlowProviderDirect, match_sample_size, match_min_certainty=0.,
                  match_reliability_threshold=0., query_img_segmentation: Optional[torch.Tensor] = None,
                  black_background: bool = True, pose_logger: PoseEstimatorLogger = None, device: str = 'cuda') \
        -> Se3 | None:
    # query_img_segmentation shape (H, W)

    path_to_colmap_db = view_graph.colmap_db_path
    path_to_reconstruction = view_graph.colmap_reconstruction_path

    path_to_cache = Path('/mnt/personal/jelint19/tmp/colmap_db_cache') / str(view_graph.object_id)
    cache_db_file = path_to_cache / path_to_colmap_db.name

    if path_to_cache.exists() and path_to_cache.is_dir():
        shutil.rmtree(path_to_cache)
    path_to_cache.mkdir(exist_ok=True, parents=True)
    shutil.copy(path_to_colmap_db, cache_db_file)

    database = pycolmap.Database(str(cache_db_file))

    h, w = query_img.shape[-2:]
    f_x = float(camera_K[0, 0])
    f_y = float(camera_K[1, 1])
    c_x = float(camera_K[0, 2])
    c_y = float(camera_K[1, 2])

    new_camera_id = database.num_cameras + 1
    new_camera = pycolmap.Camera(camera_id=new_camera_id, model=pycolmap.CameraModelId.PINHOLE, width=w,
                                 height=h,
                                 params=[f_x, f_y, c_x, c_y])

    new_image_id = database.num_images + 1
    new_database_image = pycolmap.Image(image_id=new_image_id, camera_id=new_camera_id, name='tmp_target')

    database.write_camera(new_camera, use_camera_id=True)
    database.write_image(new_database_image, use_image_id=True)

    matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}

    n = len(view_graph.view_graph.nodes())
    if n > 10:
        k = max(1, n // 10)
        while n // k < 10:
            k -= 1
    else:
        k = 1
    for frame_idx in list(view_graph.view_graph.nodes())[::k]:
        view_graph_node = view_graph.get_node_data(frame_idx)
        db_img_id = view_graph_node.colmap_db_image_id

        pose_graph_image = view_graph_node.observation.observed_image.to(device).squeeze()
        pose_graph_segmentation = view_graph_node.observation.observed_segmentation.to(device).squeeze()

        template_keypoints_xy_np = database.read_keypoints(db_img_id).astype(np.int32)
        template_keypoints_yx = torch.tensor(template_keypoints_xy_np, device=device)[:, [1, 0]]

        template_keypoints_mask = torch.zeros_like(pose_graph_segmentation)
        template_keypoints_mask[template_keypoints_yx[:, 0], template_keypoints_yx[:, 1]] = 1.
        pose_graph_segmentation_template_points = pose_graph_segmentation * template_keypoints_mask

        if black_background:
            pose_graph_image = pose_graph_image * pose_graph_segmentation

        if type(flow_provider) is FlowProviderDirect or True:
            query_img_resized = TF.resize(query_img, list(pose_graph_image.shape[-2:]))
            if query_img_segmentation is not None:
                query_seg_resized = TF.resize(query_img_segmentation[None],
                                              list(pose_graph_segmentation.shape[-2:])).squeeze()

                if black_background:
                    query_img_resized = query_img_resized * query_seg_resized.to(torch.float)
            else:
                query_seg_resized = None
            db_img_pts_xy, query_img_pts_xy, certainties = (
                flow_provider.get_source_target_points(pose_graph_image, query_img_resized, None,
                                                       pose_graph_segmentation_template_points, query_seg_resized,
                                                       as_int=True, only_foreground_matches=True))

        else:
            raise NotImplementedError('So far we can only work with RoMaFlowProviderDirect')

        if db_img_pts_xy.shape[0] == 0:  # No good matches found within the segmentation
            continue

        reliability = compute_matching_reliability(db_img_pts_xy, certainties,
                                                   pose_graph_segmentation_template_points, match_min_certainty)

        if pose_logger is not None:
            warp, certainty = flow_provider.compute_flow(pose_graph_image, query_img_resized,
                                                         source_image_segmentation=pose_graph_segmentation,
                                                         zero_certainty_outside_segmentation=True)

            pose_logger.visualize_pose_matching_rerun(db_img_pts_xy, query_img_pts_xy, certainties,
                                                      pose_graph_image, query_img_resized, reliability,
                                                      match_reliability_threshold, match_min_certainty, certainty,
                                                      viewgraph_image_segment=pose_graph_segmentation,
                                                      query_image_segment=query_seg_resized)
            pose_logger.rerun_sequence_id += 1

        print(f'Mean certainty: {certainties.mean().item()}, Reliability: {reliability}')
        if reliability >= match_reliability_threshold:
            matching_edges[(new_image_id, db_img_id)] = (query_img_pts_xy, db_img_pts_xy)
            matching_edges_certainties[(new_image_id, db_img_id)] = certainties

    if len(matching_edges) == 0:
        return None

    keypoints, edge_match_indices = \
        unique_keypoints_from_matches(matching_edges, database, eliminate_one_to_many_matches=True,
                                      matching_edges_certainties=matching_edges_certainties, device=device)

    all_image_ids = {img.image_id for img in database.read_all_images()}
    matched_images_ids = {node for edge in edge_match_indices.keys() for node in edge}
    non_matched_images_ids = all_image_ids - matched_images_ids
    non_matched_keypoints = {img_id: database.read_keypoints(img_id) for img_id in non_matched_images_ids}

    old_keypoints = {}
    for colmap_image_id in set(keypoints.keys()) - {new_image_id}:
        keypoints_np = database.read_keypoints(colmap_image_id)
        old_keypoints[colmap_image_id] = keypoints_np

    database.clear_keypoints()
    for colmap_image_id in sorted(keypoints.keys()):
        if colmap_image_id == new_image_id:
            keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
            database.write_keypoints(colmap_image_id, keypoints_np)
        else:
            keypoints_np = old_keypoints[colmap_image_id]
            database.write_keypoints(colmap_image_id, keypoints_np)

    for colmap_image_u, colmap_image_v in edge_match_indices.keys():
        match_indices_np = edge_match_indices[colmap_image_u, colmap_image_v].numpy(force=True)

        keypoints_u = database.read_keypoints(colmap_image_u)
        keypoints_v = database.read_keypoints(colmap_image_v)

        valid_match_mask = (match_indices_np[:, 0] < len(keypoints_u)) & (
                match_indices_np[:, 1] < len(keypoints_v))
        filtered_match_indices = match_indices_np[valid_match_mask]

        database.write_matches(colmap_image_u, colmap_image_v, filtered_match_indices)

    for img_id, keypoints in non_matched_keypoints.items():
        database.write_keypoints(img_id, keypoints)

    database_cache = pycolmap.DatabaseCache().create(database, 0, False, set())

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(str(path_to_reconstruction))
    reconstruction.add_camera(new_camera)
    reconstruction.add_image(new_database_image)

    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper.begin_reconstruction(reconstruction)
    mapper_options = pycolmap.IncrementalMapperOptions()

    # Register the new image
    print(f"Registering image #{new_image_id}")
    print(f"=> Image sees {mapper.observation_manager.num_visible_points3D(new_image_id)} / "
          f"{mapper.observation_manager.num_observations(new_image_id)} points")

    success = mapper.register_next_image(mapper_options, new_image_id)

    if success:
        print(f"Successfully registered image {new_image_id} into the reconstruction.")
        mapper.triangulate_image(
            mapper_options.triangulation, new_image_id
        )
        reconstruction.normalize()
        breakpoint()
    else:
        print(f"Failed to register image {new_image_id}.")

    return Se3.identity()

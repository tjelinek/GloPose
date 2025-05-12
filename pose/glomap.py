import copy
import select
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import imageio
import networkx as nx
import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3, Quaternion
from kornia.image import ImageSize
from pycolmap import TwoViewGeometryOptions, Sim3d
from tqdm import tqdm

from data_providers.frame_provider import PrecomputedSegmentationProvider, PrecomputedFrameProvider
from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.view_graph import ViewGraph

from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect, RoMaFlowProviderDirect
from data_structures.data_graph import DataGraph
from tracker_config import TrackerConfig
from utils.conversions import Se3_to_Rigid3d


class GlomapWrapper:

    def __init__(self, write_folder: Path, tracking_config: TrackerConfig, data_graph: DataGraph,
                 flow_provider: Optional[PrecomputedRoMaFlowProviderDirect] = None):
        self.write_folder = write_folder
        self.config = tracking_config

        self.flow_provider: Optional[PrecomputedRoMaFlowProviderDirect] = flow_provider

        self.colmap_base_path = (self.write_folder / f'glomap_{self.config.sequence}')

        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'
        self.feature_dir = self.colmap_base_path / 'features'

        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)
        self.feature_dir.mkdir(exist_ok=True, parents=True)

        self.data_graph = data_graph

        self.colmap_db_path = self.colmap_base_path / 'database.db'
        self.colmap_output_path = self.colmap_base_path / 'output'

    def dump_frame_node_for_glomap(self, frame_idx):

        device = self.config.device

        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to(device)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze(0).permute(1, 2, 0).to(device)

        if frame_data.image_filename is not None:
            image_filename = frame_data.image_filename
        else:
            image_filename = f'node_{frame_idx}.png'

        if frame_data.segmentation_filename is not None:
            seg_filename = frame_data.segmentation_filename
        else:
            seg_filename = f'segment_{frame_idx}.png'

        node_save_path = self.colmap_image_path / image_filename
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8).numpy(force=True))

        segmentation_save_path = self.colmap_seg_path / seg_filename
        imageio.v3.imwrite(segmentation_save_path, (img_seg * 255).to(torch.uint8).repeat(1, 1, 3).numpy(force=True))

        frame_data.image_save_path = copy.deepcopy(node_save_path)
        frame_data.segmentation_save_path = copy.deepcopy(segmentation_save_path)

    def run_glomap_from_image_list(self, images: List[Path], segmentations: List[Path],
                                   matching_pairs: List[Tuple[int, int]]) -> pycolmap.Reconstruction:
        if len(matching_pairs) == 0:
            raise ValueError("Needed at least 1 match.")

        device = self.config.device

        database_path = self.colmap_db_path

        image_pairs = [(images[i1], images[i2]) for i1, i2 in matching_pairs]
        segmentation_pairs = [(segmentations[i1], segmentations[i2]) for i1, i2 in matching_pairs]

        sample_size = self.config.roma_sample_size

        assert not database_path.exists()

        single_camera = True
        assert len(images) > 0

        matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}
        for (img1_path, img2_path), (seg1_path, seg2_path) in tqdm(zip(image_pairs, segmentation_pairs)):

            img1_path = Path(img1_path)
            img2_path = Path(img2_path)

            assert img1_path.parent == img2_path.parent

            img1 = PrecomputedFrameProvider.load_and_downsample_image(img1_path, 1., device)
            img2 = PrecomputedFrameProvider.load_and_downsample_image(img2_path, 1., device)

            h1, w1 = img1.shape[-2:]
            h2, w2 = img2.shape[-2:]
            if h1 != h2 or w1 != w2:
                single_camera = False

            seg1_size = ImageSize(h1, w1)
            seg2_size = ImageSize(h2, w2)
            img1_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg1_path, seg1_size, device)
            img2_seg = PrecomputedSegmentationProvider.load_and_downsample_segmentation(seg2_path, seg2_size, device)

            src_pts_xy_roma_int, dst_pts_xy_roma_int, certainty =\
                self.flow_provider.get_source_target_points_roma(img1, img2, sample_size, img1_seg.squeeze(),
                                                                 img2_seg.squeeze(), Path(img1_path.name),
                                                                 Path(img2_path.name), as_int=True,
                                                                 zero_certainty_outside_segmentation=True,
                                                                 only_foreground_matches=True)
            img1_id = images.index(img1_path) + 1
            img2_id = images.index(img2_path) + 1
            edge = (img1_id, img2_id)
            matching_edges[edge] = (src_pts_xy_roma_int, dst_pts_xy_roma_int)
            matching_edges_certainties[edge] = certainty

        database = pycolmap.Database(str(database_path))

        keypoints, edge_match_indices = unique_keypoints_from_matches(matching_edges, None, matching_edges_certainties,
                                                                      eliminate_one_to_many_matches=True, device=device)

        first_frame_data = self.data_graph.get_frame_data(0)
        h, w = first_frame_data.image_shape.height, first_frame_data.image_shape.width
        camera_K = first_frame_data.gt_pinhole_K
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])

        new_cam_id = 1
        if single_camera:
            new_camera = pycolmap.Camera(camera_id=new_cam_id, model=pycolmap.CameraModelId.PINHOLE, width=w, height=h,
                                         params=[f_x, f_y, c_x, c_y])
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

        two_view_geometry(self.colmap_db_path)

        first_image_id = None
        second_image_id = None
        if self.config.init_with_first_two_images:
            first_image_id = 1
            second_image_id = 2
        run_mapper(self.colmap_output_path, self.colmap_db_path, self.colmap_image_path, self.config.mapper,
                   first_image_id, second_image_id)

        path_to_rec = self.colmap_output_path / '0'
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        try:
            print(reconstruction.summary())
        except Exception as e:
            print(e)
            raise Exception("Reconstruction failed", e)

        return reconstruction


def align_with_kabsch(reconstruction: pycolmap.Reconstruction, gt_Se3_world2cam_poses: Dict[str, Se3])\
        -> pycolmap.Reconstruction:

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
    sim3d = sim3d_report['tgt_from_src']
    reconstruction.transform(sim3d)

    return reconstruction


def two_view_geometry(colmap_db_path: Path):
    opts = TwoViewGeometryOptions()
    opts.detect_watermark = False
    pycolmap.match_exhaustive(str(colmap_db_path), verification_options=opts)


def run_mapper(colmap_output_path: Path, colmap_db_path: Path, colmap_image_path: Path, mapper: str = 'pycolmap',
               first_image_id: Optional[int] = None, second_image_id: Optional[int] = None):

    colmap_output_path.mkdir(exist_ok=True, parents=True)

    initial_pair_provided = first_image_id is not None and second_image_id is not None
    if mapper in ['colmap', 'glomap']:
        if mapper == 'glomap':
            command = [
                "glomap",
                "mapper",
                "--database_path", str(colmap_db_path),
                "--output_path", str(colmap_output_path),
                "--image_path", str(colmap_image_path),
                "--TrackEstablishment.min_num_view_per_track", str(2),
            ]

        elif mapper == 'colmap':
            command = [
                "colmap",
                "mapper",
                "--database_path", str(colmap_db_path),
                "--output_path", str(colmap_output_path),
                "--image_path", str(colmap_image_path),
                "--Mapper.tri_ignore_two_view_tracks", str(0),
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
        opts.triangulation.ignore_two_view_tracks = False
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


def get_image_Se3_world2cam(image: pycolmap.Image, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = image.cam_from_world
    image_t_cam = torch.tensor(image_world2cam.translation).to(device).to(torch.float)
    image_q_cam_xyzw = torch.tensor(image_world2cam.rotation.quat[[3, 0, 1, 2]]).to(device).to(torch.float)
    Se3_image_world2cam = Se3(Quaternion(image_q_cam_xyzw), image_t_cam)

    return Se3_image_world2cam


def world2cam_from_reconstruction(reconstruction: pycolmap.Reconstruction) -> Dict[int, Se3]:
    poses = {}
    for image_id, image in reconstruction.images.items():
        Se3_world2cam = get_image_Se3_world2cam(image, 'cpu')
        poses[image_id] = Se3_world2cam
    return poses


def predict_poses(query_img: torch.Tensor, query_img_segmentation: torch.Tensor, camera_K: np.ndarray,
                  view_graph: ViewGraph, flow_provider: RoMaFlowProviderDirect | SIFTMatchingProvider,
                  config: TrackerConfig):
    device = config.device
    sequence = 'obj_000010'
    base_results_path = Path(f'/mnt/personal/jelint19/results/FlowTracker/')
    db_relative_path = Path(f'handal/{sequence}/glomap_{sequence}/')
    db_filename = Path('database.db')

    path_to_colmap_db = base_results_path / db_relative_path / db_filename
    path_to_reconstruction = Path(f'/mnt/personal/jelint19/cache/view_graph_cache/handal/{sequence}_down/reconstruction/0')
    path_to_cache = Path('/mnt/personal/jelint19/tmp/colmap_db_cache') / db_relative_path
    cache_db_file = path_to_cache / db_filename

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
    new_camera = pycolmap.Camera(camera_id=new_camera_id, model=pycolmap.CameraModelId.PINHOLE, width=w, height=h,
                                 params=[f_x, f_y, c_x, c_y])

    new_image_id = database.num_images + 1
    new_database_image = pycolmap.Image(image_id=new_image_id, camera_id=new_camera_id, name='tmp_target')

    database.write_camera(new_camera, use_camera_id=True)
    database.write_image(new_database_image, use_image_id=True)

    matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}

    for frame_idx in view_graph.view_graph.nodes():
        view_graph_node = view_graph.get_node_data(frame_idx)
        db_img_id = view_graph_node.colmap_db_image_id

        pose_graph_image = view_graph_node.observation.observed_image.to(device).squeeze()
        pose_graph_segmentation = view_graph_node.observation.observed_segmentation.to(device).squeeze()

        if type(flow_provider) is RoMaFlowProviderDirect or True:
            query_img_pts_xy, db_img_pts_xy, certainties = (
                flow_provider.get_source_target_points_roma(query_img, pose_graph_image, config.roma_sample_size,
                                                            query_img_segmentation, pose_graph_segmentation,
                                                            as_int=True, zero_certainty_outside_segmentation=True,
                                                            only_foreground_matches=True))
        else:
            raise NotImplementedError('So far we can only work with RoMaFlowProviderDirect')

        matching_edges[(new_image_id, db_img_id)] = (query_img_pts_xy, db_img_pts_xy)
        matching_edges_certainties[(new_image_id, db_img_id)] = certainties

    keypoints, edge_match_indices = unique_keypoints_from_matches(matching_edges, database,
                                                                  eliminate_one_to_many_matches=True,
                                                                  matching_edges_certainties=matching_edges_certainties,
                                                                  device=device)

    all_image_ids = {img.image_id for img in database.read_all_images()}
    matched_images_ids = {node for edge in edge_match_indices.keys() for node in edge}
    non_matched_images_ids = all_image_ids - matched_images_ids
    non_matched_keypoints = {img_id: database.read_keypoints(img_id) for img_id in non_matched_images_ids}

    database.clear_keypoints()
    for colmap_image_id in sorted(keypoints.keys()):
        keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
        database.write_keypoints(colmap_image_id, keypoints_np)

    for colmap_image_u, colmap_image_v in edge_match_indices.keys():
        match_indices_np = edge_match_indices[colmap_image_u, colmap_image_v].numpy(force=True)
        database.write_matches(colmap_image_u, colmap_image_v, match_indices_np)

    for img_id, keypoints in non_matched_keypoints.items():
        database.write_keypoints(img_id, keypoints)

    database_cache = pycolmap.DatabaseCache().create(database, 0, False, set())

    mapper = pycolmap.IncrementalMapper(database_cache)
    mapper_options = pycolmap.IncrementalMapperOptions()

    reconstruction_manager = pycolmap.ReconstructionManager()
    reconstruction_manager.read(str(path_to_reconstruction))

    reconstruction_idx = reconstruction_manager.add()
    reconstruction = pycolmap.Reconstruction(str(path_to_reconstruction))

    mapper.begin_reconstruction(reconstruction)

    # Register the new image
    success = mapper.register_next_image(mapper_options, new_image_id)

    if success:
        print(f"Successfully registered image {new_image_id} into the reconstruction.")
        mapper.triangulate_image(
            mapper_options.triangulation, new_image_id
        )
        reconstruction.normalize()
    else:
        print(f"Failed to register image {new_image_id}.")


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
    starting_positions = torch.cat((
        torch.tensor([0], device=cumulative_counts.device),
        cumulative_counts[:-1]
    ))

    # Get the indices of first occurrences
    first_occurrence_indices = sorted_positions[starting_positions]

    return first_occurrence_indices, element_to_unique_mapping


def align_reconstruction_with_pose(reconstruction: pycolmap.Reconstruction, first_image_gt_Se3_world2cam: Se3,
                                   image_depths: Dict[str, torch.Tensor], first_image_name: str) \
        -> Tuple[pycolmap.Reconstruction, bool]:
    # The alignment assumes that the COLMAP and GT spaces have the same origins.
    reconstruction = copy.deepcopy(reconstruction)

    if not (first_image_colmap := reconstruction.find_image_with_name(first_image_name)):
        print("Alignment error. The 1st image was not registered.")
        return reconstruction, False

    gt_first_image_world2cam = Se3_to_Rigid3d(first_image_gt_Se3_world2cam)

    colmap_first_image_world2cam = first_image_colmap.cam_from_world

    C_gt_world = gt_first_image_world2cam.inverse().translation  # metric
    C_colmap = colmap_first_image_world2cam.inverse().translation  # non-metric

    scale = np.linalg.norm(C_gt_world) / np.linalg.norm(C_colmap)

    Sim3d_gt_world = Sim3d(scale, gt_first_image_world2cam.rotation.matrix(), gt_first_image_world2cam.translation)
    Sim3d_colmap = Sim3d(1.0, colmap_first_image_world2cam.rotation.matrix(), colmap_first_image_world2cam.translation)

    Sim3d_pred2gt = Sim3d_gt_world * Sim3d_colmap.inverse()

    reconstruction.transform(Sim3d_pred2gt)
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

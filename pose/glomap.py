import copy
import os
import select
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import h5py
import imageio
import networkx as nx
import numpy as np
import pycolmap
import torch
from kornia.geometry import Se3, Quaternion
from kornia.image import ImageSize
from tqdm import tqdm

from data_providers.frame_provider import PrecomputedSegmentationProvider, PrecomputedFrameProvider
from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.view_graph import ViewGraph

from utils.colmap.h5_to_db import import_into_colmap
from utils.sift import detect_sift, get_exhaustive_image_pairs, match_features
from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect, RoMaFlowProviderDirect
from data_structures.data_graph import DataGraph
from tracker_config import TrackerConfig
from utils.general import extract_intrinsics_from_tensor


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
        img_ext = 'png'

        keypoints_data = defaultdict(lambda: torch.empty(0, 2).to(device))
        matches_data = defaultdict(dict)
        single_camera = True
        assert len(images) > 0

        for (img1_path, img2_path), (seg1_path, seg2_path) in tqdm(zip(image_pairs, segmentation_pairs)):

            img1_path = Path(img1_path)
            img2_path = Path(img2_path)

            assert img1_path.parent == img2_path.parent

            downsample_factor = self.config.image_downsample
            img1 = PrecomputedFrameProvider.load_and_downsample_image(img1_path, downsample_factor, device)
            img2 = PrecomputedFrameProvider.load_and_downsample_image(img2_path, downsample_factor, device)

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

            src_pts_xy_roma_all = torch.cat([keypoints_data[img1_path.name], src_pts_xy_roma_int])
            dst_pts_xy_roma_all = torch.cat([keypoints_data[img2_path.name], dst_pts_xy_roma_int])

            src_pts_xy_roma_unique = torch.unique(src_pts_xy_roma_all, return_inverse=False, dim=0)
            dst_pts_xy_roma_unique = torch.unique(dst_pts_xy_roma_all, return_inverse=False, dim=0)

            keypoints_data[img1_path.name] = src_pts_xy_roma_unique
            keypoints_data[img2_path.name] = dst_pts_xy_roma_unique

            matches_data[img1_path.name] = {img2_path.name: (src_pts_xy_roma_int, dst_pts_xy_roma_int)}

        # Delete possibly old data
        for file in self.feature_dir.iterdir():
            if file.is_file():
                file.unlink()
        if database_path.exists():
            database_path.unlink()

        matches_file = self.feature_dir / 'matches.h5'
        keypoints_file = self.feature_dir / 'keypoints.h5'

        assert not matches_file.exists()
        assert not keypoints_file.exists()

        with h5py.File(matches_file, mode='w') as f_match:
            for img1_key, match_data in matches_data.items():
                group = f_match.require_group(str(img1_key))
                for img2_key, (src_pts, dst_pts) in match_data.items():
                    keypoints_image1 = keypoints_data[img1_key].to(torch.int)
                    keypoints_image2 = keypoints_data[img2_key].to(torch.int)

                    src_pts_indices = get_match_points_indices(keypoints_image1, src_pts)
                    dst_pts_indices = get_match_points_indices(keypoints_image2, dst_pts)

                    match_indices = torch.stack([src_pts_indices, dst_pts_indices], dim=-1).numpy(force=True)

                    group.create_dataset(str(img2_key), data=match_indices)

        with h5py.File(keypoints_file, mode='w') as f_kp:
            for img_key, keypoints in keypoints_data.items():
                f_kp[str(img_key)] = keypoints.numpy(force=True)

        import_into_colmap(self.colmap_image_path, self.feature_dir, database_path, img_ext, single_camera)

        from time import sleep
        sleep(1)
        self.run_mapper(self.config.mapper)

        path_to_rec = self.colmap_output_path / '0'
        print(path_to_rec)
        sleep(1)  # Wait for the rec to be written
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        try:
            print(reconstruction.summary())
        except Exception as e:
            raise Exception("Reconstruction failed", e)

        return reconstruction

    def align_with_kabsch(self, reconstruction: pycolmap.Reconstruction) -> pycolmap.Reconstruction:
        reconstruction = copy.deepcopy(reconstruction)

        gt_reconstruction = pycolmap.Reconstruction()

        gt_K = self.data_graph.get_frame_data(0).gt_pinhole_K
        if gt_K is not None:
            fx, fy, cx, cy = extract_intrinsics_from_tensor(gt_K)
            fx = fx.item()
            fy = fy.item()
        else:
            pred_reconstruction_cam = reconstruction.cameras[1]
            cx, cy = (pred_reconstruction_cam.params[1], pred_reconstruction_cam.params[2])
            fx, fy = (pred_reconstruction_cam.params[0], pred_reconstruction_cam.params[0])

        gt_w = reconstruction.cameras[1].width  # Assuming single camera only
        gt_h = reconstruction.cameras[1].height

        gt_reconstruction_cam_id = 1
        cam = pycolmap.Camera(model=1, width=gt_w, height=gt_h, params=np.array([fx, fy, cx, cy]))
        gt_reconstruction.add_camera(cam)

        tgt_image_names = []
        tgt_3d_locations = []

        for n in self.data_graph.G.nodes:
            node_data = self.data_graph.get_frame_data(n)

            gt_Se3_world2cam = node_data.gt_Se3_cam2obj.inverse()
            gt_q_xyzw_world2cam = gt_Se3_world2cam.quaternion.q.squeeze().numpy(force=True)[[1, 2, 3, 0]].astype(
                np.float64)
            gt_t_world2cam = gt_Se3_world2cam.t.squeeze().numpy(force=True).astype(np.float64)

            image_name = node_data.image_filename
            gt_image = pycolmap.Image(name=image_name, image_id=n, camera_id=gt_reconstruction_cam_id)
            gt_world2cam = pycolmap.Rigid3d(rotation=gt_q_xyzw_world2cam, translation=gt_t_world2cam)
            gt_image.cam_from_world = gt_world2cam
            gt_reconstruction.add_image(gt_image)
            gt_reconstruction.register_image(n)

            tgt_image_names.append(image_name)
            tgt_3d_locations.append(gt_t_world2cam)

        sim3d = pycolmap.align_reconstructions_via_proj_centers(reconstruction, gt_reconstruction, 1.)

        reconstruction.transform(sim3d)

        return reconstruction

    def align_with_first_pose(self, reconstruction: pycolmap.Reconstruction, gt_Se3_obj2cam: Se3, frame_i: int) -> (
            pycolmap.Reconstruction):

        reconstruction = copy.deepcopy(reconstruction)

        frame_to_name = {frame: str(self.data_graph.get_frame_data(frame).image_filename)
                         for frame in self.data_graph.G.nodes}
        first_image_name = frame_to_name[frame_i]

        reconstruction_name_to_key = {reconstruction.images[k].name: k for k in reconstruction.images.keys()}

        first_image_colmap_index = reconstruction_name_to_key[first_image_name]
        ref_image_Se3_world2cam = get_image_Se3_world2cam(reconstruction, first_image_colmap_index, self.config.device)

        Se3_sim = gt_Se3_obj2cam * ref_image_Se3_world2cam.inverse()
        scale = (torch.linalg.norm(ref_image_Se3_world2cam.translation) /
                 (torch.linalg.norm(gt_Se3_obj2cam.translation) + 1e-10))

        R_sim_np = Se3_sim.quaternion.matrix().numpy(force=True)
        rot_3d = pycolmap.Rotation3d(R_sim_np)
        t_np = Se3_sim.t.numpy(force=True)

        sim_3D = pycolmap.Sim3d(scale.item(), rot_3d, t_np)

        reconstruction.transform(sim_3D)

        return reconstruction

    def run_mapper(self, mapper: str = 'pycolmap'):

        pycolmap.match_exhaustive(str(self.colmap_db_path))
        self.colmap_output_path.mkdir(exist_ok=True, parents=True)
        if mapper in ['colmap', 'glomap']:
            if mapper == 'glomap':
                pycolmap.match_exhaustive(self.colmap_db_path)
                command = [
                    "glomap",
                    "mapper",
                    "--database_path", str(self.colmap_db_path),
                    "--output_path", str(self.colmap_output_path),
                    "--image_path", str(self.colmap_image_path),
                    "--TrackEstablishment.min_num_view_per_track", str(2),
                ]

            elif mapper == 'colmap':
                pycolmap.match_exhaustive(self.colmap_db_path)
                command = [
                    "colmap",
                    "mapper",
                    "--database_path", str(self.colmap_db_path),
                    "--output_path", str(self.colmap_output_path),
                    "--image_path", str(self.colmap_image_path),
                    "--Mapper.tri_ignore_two_view_tracks", str(0),
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
            opts.min_num_matches = 0
            opts.ignore_watermarks = False
            opts.triangulation.ignore_two_view_tracks = False
            pycolmap.match_exhaustive(self.colmap_db_path)
            maps = pycolmap.incremental_mapping(self.colmap_db_path, self.colmap_image_path, self.colmap_output_path,
                                                options=opts)
            if len(maps) > 0:
                maps[0].write(self.colmap_output_path)
                print(maps[0].summary())
        else:
            raise ValueError(f"Need to run either glomap or colmap, got mapper={mapper}")

    def run_glomap_from_image_list_sift(self, keyframes: List[Path], segmentations: List[Path], matching_pairs=None):

        feature_dir = self.feature_dir
        device = self.config.device
        database_path = self.colmap_db_path

        detect_sift(keyframes,
                    segmentations,
                    self.config.sift_filter_num_feats,
                    device=self.config.device,
                    feature_dir=feature_dir)
        if matching_pairs is None:
            index_pairs = get_exhaustive_image_pairs(keyframes)
        else:
            index_pairs = matching_pairs
        print("Matching features")

        match_features(keyframes, index_pairs, feature_dir=feature_dir, device=device,
                       alg='adalam')
        dirname = os.path.dirname(keyframes[0])  # Assume all images are in the same directory
        print("Dirname", dirname)

        import_into_colmap(dirname, feature_dir=feature_dir, database_path=database_path, img_ext='png')
        print("Reconstruction")

        self.run_mapper(self.config.mapper)

        path_to_rec = self.colmap_output_path / '0'

        from time import sleep
        sleep(1)  # Wait for the rec to be written
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        try:
            print(reconstruction.summary())
        except Exception as e:
            print(e)

        return reconstruction


def get_match_points_indices(keypoints, match_pts):
    N = keypoints.shape[0]
    keypoints_and_match_pts = torch.cat([keypoints, match_pts], dim=0)
    _, kpts_and_match_pts_indices = torch.unique(keypoints_and_match_pts, return_inverse=True, dim=0)

    if kpts_and_match_pts_indices.max() >= N:
        raise ValueError("Not all src_pts included in keypoints")
    assert torch.equal(keypoints[kpts_and_match_pts_indices[:N]], keypoints)
    match_pts_indices = kpts_and_match_pts_indices[N:]
    return match_pts_indices


def get_image_Se3_world2cam(reconstruction: pycolmap.Reconstruction, image_key: int, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = reconstruction.images[image_key].cam_from_world
    image_t_cam = torch.tensor(image_world2cam.translation).to(device).to(torch.float)
    image_q_cam_xyzw = torch.tensor(image_world2cam.rotation.quat[[3, 0, 1, 2]]).to(device).to(torch.float)
    Se3_image_world2cam = Se3(Quaternion(image_q_cam_xyzw), image_t_cam)

    return Se3_image_world2cam


def predict_poses(query_img: torch.Tensor, query_img_segmentation: torch.Tensor, camera_K: np.ndarray,
                  view_graph: ViewGraph, flow_provider: RoMaFlowProviderDirect | SIFTMatchingProvider,
                  config: TrackerConfig):
    device = config.device
    sequence = 'obj_000005'
    base_results_path = Path(f'/mnt/personal/jelint19/results/FlowTracker/')
    db_relative_path = Path(f'hope/{sequence}/glomap_{sequence}/')
    db_filename = Path('database.db')

    path_to_colmap_db = base_results_path / db_relative_path / db_filename
    path_to_reconstruction = Path(f'/mnt/personal/jelint19/cache/view_graph_cache/hope/{sequence}/reconstruction')
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

    for frame_idx in view_graph.view_graph.nodes():
        view_graph_node = view_graph.get_node_data(frame_idx)
        db_img_id = view_graph_node.colmap_db_image_id

        pose_graph_image = view_graph_node.observation.observed_image.to(device).squeeze()
        pose_graph_segmentation = view_graph_node.observation.observed_segmentation.to(device).squeeze()

        if type(flow_provider) is RoMaFlowProviderDirect or True:
            query_img_pts_xy, db_img_pts_xy, _ = (
                flow_provider.get_source_target_points_roma(query_img, pose_graph_image, config.roma_sample_size,
                                                            query_img_segmentation, pose_graph_segmentation,
                                                            as_int=True, zero_certainty_outside_segmentation=True,
                                                            only_foreground_matches=True))
        else:
            raise NotImplementedError('So far we can only work with RoMaFlowProviderDirect')

        matching_edges[(new_image_id, db_img_id)] = (query_img_pts_xy, db_img_pts_xy)

    keypoints, edge_match_indices = unique_keypoints_from_matches(matching_edges, database, device)

    all_image_ids = {img.image_id for img in database.read_all_images()}
    matched_images_ids = {node for edge in edge_match_indices.keys() for node in edge}
    non_matched_images_ids = all_image_ids - matched_images_ids
    non_matched_keypoints = {img_id: database.read_keypoints(img_id) for img_id in non_matched_images_ids}

    database.clear_keypoints()
    for colmap_image_id in keypoints.keys():
        keypoints_np = keypoints[colmap_image_id].numpy(force=True)
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
    reconstruction_manager.read(path_to_reconstruction)

    reconstruction_idx = reconstruction_manager.add()
    reconstruction = pycolmap.Reconstruction(path_to_reconstruction)

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
    # Need to use numpy because sort=False argument of torch.unique() does not work

    keypoints_np = keypoints.numpy(force=True)

    # NumPy has a built-in way to get unique elements with indices
    _, index_np, inverse_indices_np = np.unique(keypoints_np, return_index=True, return_inverse=True, axis=0)

    # Sort indices to get elements in order of appearance
    index_sort_permutation_np = np.argsort(index_np)
    index_sort_permutation = torch.from_numpy(index_sort_permutation_np).to(keypoints.device)
    inverse_indices = torch.from_numpy(inverse_indices_np).to(keypoints.device)
    index = torch.from_numpy(index_np).to(keypoints.device)

    unique_order_preserving = keypoints[index[index_sort_permutation]]

    idx_mapping = torch.zeros(len(index_sort_permutation), dtype=torch.long).to(keypoints.device)

    # Use scatter_ to place indices at the positions specified by index_sort_permutation
    # This creates our mapping from original positions to new positions after reordering
    idx_mapping.scatter_(0, index_sort_permutation, torch.arange(len(index_sort_permutation), device=keypoints.device))

    # Apply the mapping to get correct inverse indices
    inverse_indices_order_preserving = idx_mapping[inverse_indices]

    assert torch.all(torch.eq(keypoints, unique_order_preserving[inverse_indices_order_preserving]).view(-1))

    return unique_order_preserving, inverse_indices_order_preserving


def unique_keypoints_from_matches(matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]],
                                  existing_database: pycolmap.Database = None, device: str = 'cpu') -> (
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

    edge_match_indices_concatenated: Dict[Tuple[int, int], torch.Tensor] = {
        (u, v): torch.stack([edge_match_indices[u, v, 0], edge_match_indices[u, v, 1]], dim=1)
        for u, v in G.to_undirected().edges()
    }

    return keypoints_for_node, edge_match_indices_concatenated

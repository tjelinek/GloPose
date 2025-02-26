import copy
import os
import select
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import imageio
import numpy as np
import pycolmap
import torch
from PIL import Image
from kornia.geometry import Se3, Quaternion
from torchvision import transforms
from romatch import roma_outdoor
from tqdm import tqdm

from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.view_graph import ViewGraph

from utils.colmap.h5_to_db import import_into_colmap
from utils.sift import detect_sift, get_exhaustive_image_pairs, match_features
from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect, RoMaFlowProviderDirect
from data_structures.data_graph import DataGraph
from flow import roma_warp_to_pixel_coordinates
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
        dtype = torch.float16 if 'cuda' in str(device) else torch.float32
        matcher = roma_outdoor(device, amp_dtype=dtype)

        database_path = self.colmap_db_path

        image_pairs = [(images[i1], images[i2]) for i1, i2 in matching_pairs]
        segmentation_pairs = [(segmentations[i1], segmentations[i2]) for i1, i2 in matching_pairs]

        transform_from_PIL = transforms.ToTensor()

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

            img1_PIL = Image.open(str(img1_path)).convert('RGB')
            img2_PIL = Image.open(str(img2_path)).convert('RGB')

            h1, w1 = img1_PIL.size
            h2, w2 = img2_PIL.size
            if h1 != h2 or w1 != w2:
                single_camera = False
            img1_seg = transform_from_PIL(Image.open(seg1_path).convert('L')).to(device)
            img2_seg = transform_from_PIL(Image.open(seg2_path).convert('L')).to(device)

            roma_size_hw = (864, 864)
            roma_h, roma_w = roma_size_hw

            img1_seg_roma_size = transforms.functional.resize(img1_seg.clone(), size=roma_size_hw)
            img2_seg_roma_size = transforms.functional.resize(img2_seg.clone(), size=roma_size_hw)
            if len(img1_seg_roma_size.shape) > 2:
                img1_seg_roma_size = img1_seg_roma_size.mean(dim=0)
            if len(img2_seg_roma_size.shape) > 2:
                img2_seg_roma_size = img2_seg_roma_size.mean(dim=0)

            result = None
            if self.flow_provider is not None:
                result = self.flow_provider.cached_flow_from_filenames(img1_path.name, img2_path.name)
            if result is None or True:
                warp, certainty = matcher.match(img1_PIL, img2_PIL, device=device)
            else:
                warp, certainty = result

            certainty = certainty.clone()
            certainty[:, :roma_w] *= img1_seg_roma_size.mT.squeeze().bool().float()
            certainty[:, roma_w:2 * roma_w] *= img2_seg_roma_size.mT.squeeze().bool().float()

            warp, certainty = matcher.sample(warp, certainty, self.config.roma_sample_size)
            warp = warp[certainty > 0]

            src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

            src_pts_xy_roma_int = src_pts_xy_roma.to(torch.int)
            dst_pts_xy_roma_int = dst_pts_xy_roma.to(torch.int)

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
            print(e)

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

    match_pts_indices = kpts_and_match_pts_indices[N:]
    return match_pts_indices


def get_image_Se3_world2cam(reconstruction: pycolmap.Reconstruction, image_key: int, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = reconstruction.images[image_key].cam_from_world
    image_t_cam = torch.tensor(image_world2cam.translation).to(device).to(torch.float)
    image_q_cam_xyzw = torch.tensor(image_world2cam.rotation.quat[[3, 0, 1, 2]]).to(device).to(torch.float)
    Se3_image_world2cam = Se3(Quaternion(image_q_cam_xyzw), image_t_cam)

    return Se3_image_world2cam


def predict_poses(image: torch.Tensor, segmentation: torch.Tensor, view_graph: ViewGraph,
                  flow_provider: RoMaFlowProviderDirect | SIFTMatchingProvider, config: TrackerConfig):
    breakpoint()

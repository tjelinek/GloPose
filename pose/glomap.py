import copy
import select
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import h5py
import imageio
import numpy as np
import pycolmap
import torch
from PIL import Image
from torchvision import transforms
from romatch import roma_outdoor
from tqdm import tqdm

from auxiliary_scripts.colmap.h5_to_db import import_into_colmap
from auxiliary_scripts.image_utils import ImageShape
from data_structures.data_graph import DataGraph
from data_structures.pose_icosphere import PoseIcosphere
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig


class GlomapWrapper:

    def __init__(self, write_folder: Path, tracking_config: TrackerConfig, data_graph: DataGraph,
                 image_shape: ImageShape, pose_icosphere: PoseIcosphere):
        self.write_folder = write_folder
        self.config = tracking_config
        self.colmap_base_path = (self.write_folder / f'glomap_{self.config.sequence}')

        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'
        self.feature_dir = self.colmap_base_path / 'features'

        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)
        self.feature_dir.mkdir(exist_ok=True, parents=True)

        self.image_width = image_shape.width
        self.image_height = image_shape.height

        self.data_graph = data_graph
        self.pose_icosphere = pose_icosphere

        self.colmap_db_path = self.colmap_base_path / 'database.db'
        self.colmap_output_path = self.colmap_base_path / 'output'

    def dump_frame_node_for_glomap(self, frame_idx):

        device = self.config.device

        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to(device)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze([0, 1]).permute(1, 2, 0).to(device)

        node_save_path = self.colmap_image_path / f'node_{frame_idx}.png'
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8).numpy(force=True))

        segmentation_save_path = self.colmap_seg_path / f'segment_{frame_idx}.png'
        imageio.v3.imwrite(segmentation_save_path, (img_seg * 255).to(torch.uint8).repeat(1, 1, 3).numpy(force=True))

        frame_data.image_save_path = copy.deepcopy(node_save_path)
        frame_data.segmentation_save_path = copy.deepcopy(segmentation_save_path)

    def run_glomap_from_image_list(self, images: List[Path], segmentations: List[Path],
                                   matching_pairs: List[Tuple[int, int]], datagraph_cache: Optional[DataGraph] = None)\
            -> pycolmap.Reconstruction:
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
        count = 0
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

            warp, certainty = matcher.match(img1_PIL, img2_PIL, device=device)

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

            n_matches = src_pts_xy_roma_int.shape[0]
            count += 1

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

        self.run_glomap()

        path_to_rec = self.colmap_output_path / '0'
        print(path_to_rec)
        from time import sleep
        sleep(1)  # Wait for the rec to be written
        reconstruction = pycolmap.Reconstruction(path_to_rec)
        try:
            print(reconstruction.summary())
        except Exception as e:
            print(e)

        return reconstruction

    def normalize_reconstruction(self, reconstruction: pycolmap.Reconstruction) -> pycolmap.Reconstruction:
        reconstruction = copy.deepcopy(reconstruction)

        gt_reconstruction = pycolmap.Reconstruction()

        gt_cam_params = self.data_graph.get_frame_data(0).gt_pinhole_params

        gt_w = int(gt_cam_params.width.item())
        gt_h = int(gt_cam_params.height.item())
        fx = gt_cam_params.fx.item()
        fy = gt_cam_params.fy.item()

        gt_reconstruction_cam_id = 1
        cam = pycolmap.Camera(model=1, width=gt_w, height=gt_h, params=np.array([fx, fy, gt_w / 2.0, gt_h / 2.0]))
        gt_reconstruction.add_camera(cam)

        tgt_image_names = []
        tgt_3d_locations = []

        for n in self.data_graph.G.nodes:
            node_data = self.data_graph.get_frame_data(n)

            cam_pose_Se3 = node_data.gt_pose_cam
            cam_pose_q_xyzw = cam_pose_Se3.quaternion.q.squeeze().numpy(force=True)[[1, 2, 3, 0]].astype(np.float64)
            cam_pose_t = cam_pose_Se3.t.squeeze().numpy(force=True).astype(np.float64)

            image_name = f'node_{n}.png'

            gt_image = pycolmap.Image(name=image_name, image_id=n, camera_id=gt_reconstruction_cam_id)
            gt_cam_pose = pycolmap.Rigid3d(rotation=cam_pose_q_xyzw, translation=cam_pose_t)
            gt_image.cam_from_world = gt_cam_pose
            gt_reconstruction.add_image(gt_image)

            tgt_image_names.append(image_name)
            tgt_3d_locations.append(cam_pose_t)

        reconstruction.align_poses(gt_reconstruction)
        sim3d = pycolmap.align_reconstructions_via_proj_centers(gt_reconstruction, reconstruction, 100.)

        return reconstruction

    def run_glomap(self, mapper: str = 'glomap'):

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
                    "--database_path", (self.colmap_db_path),
                    "--output_path", (self.colmap_output_path),
                    "--image_path", (self.colmap_image_path),
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


def get_match_points_indices(keypoints, match_pts):
    N = keypoints.shape[0]
    keypoints_and_match_pts = torch.cat([keypoints, match_pts], dim=0)
    _, kpts_and_match_pts_indices = torch.unique(keypoints_and_match_pts, return_inverse=True, dim=0)

    if kpts_and_match_pts_indices.max() >= N:
        raise ValueError("Not all src_pts included in keypoints")

    match_pts_indices = kpts_and_match_pts_indices[N:]
    return match_pts_indices

import select
import subprocess
from pathlib import Path

import imageio
import numpy as np
import pycolmap
import torch
from kornia.geometry import PinholeCamera

from auxiliary_scripts.colmap.colmap_database import COLMAPDatabase
from auxiliary_scripts.image_utils import ImageShape
from data_structures.data_graph import DataGraph
from data_structures.pose_icosphere import PoseIcosphere
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig


class GlomapWrapper:

    def __init__(self, write_folder: Path, tracking_config: TrackerConfig, data_graph: DataGraph,
                 image_shape: ImageShape, pinhole_params: PinholeCamera, pose_icosphere: PoseIcosphere):
        self.write_folder = write_folder
        self.tracking_config = tracking_config
        self.colmap_base_path = (self.write_folder /
                                  f'icosphere_dump_{self.tracking_config.sequence}')

        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'

        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)

        self.image_width = image_shape.width
        self.image_height = image_shape.height

        self.data_graph = data_graph
        self.pinhole_params = pinhole_params
        self.pose_icosphere = pose_icosphere

        self.colmap_db_path = self.colmap_base_path / 'database.db'
        self.colmap_output_path = self.colmap_base_path / 'output'
        self.colmap_db: COLMAPDatabase = COLMAPDatabase.connect(self.colmap_db_path)
        self.colmap_db.create_tables()

        colmap_db_camera_params = (np.array([self.pinhole_params.fx.item(), self.pinhole_params.fy.item(),
                                             self.pinhole_params.cx.item(), self.pinhole_params.cy.item()]).
                                   astype(np.float64))
        self.colmap_db.add_camera(1, self.image_width, self.image_height, colmap_db_camera_params, camera_id=1)

    def dump_icosphere_node_for_glomap(self, icosphere_node):

        frame_idx = icosphere_node.keyframe_idx_observed
        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to('cuda')
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze([0, 1]).permute(1, 2, 0).to('cuda')

        node_save_path = self.colmap_image_path / f'node_{frame_idx}.png'
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8).numpy(force=True))

        segmentation_save_path = self.colmap_seg_path / f'segment_{frame_idx}.png'
        imageio.v3.imwrite(segmentation_save_path, (img_seg * 255).to(torch.uint8).repeat(1, 1, 3).numpy(force=True))

        seg_target_nonzero = img_seg[..., 0].nonzero()
        seg_target_nonzero_unit = seg_target_nonzero
        seg_target_nonzero_xy_np = seg_target_nonzero_unit[..., [1, 0]].numpy(force=True).astype(np.float32)
        assert seg_target_nonzero_xy_np.dtype == np.float32

        self.colmap_db.add_image(name=f'./{str(node_save_path.name)}', camera_id=1, image_id=frame_idx + 1)
        self.colmap_db.add_keypoints(frame_idx + 1, seg_target_nonzero_xy_np.copy())

        icosphere_nodes_idx = {node.keyframe_idx_observed for node in self.pose_icosphere.reference_poses}
        incoming_edges = self.data_graph.G.in_edges(frame_idx)

        for edge in incoming_edges:
            edge_source = edge[0]
            edge_target = edge[1]

            if edge_source not in icosphere_nodes_idx:
                continue

            source_node = self.data_graph.get_frame_data(edge_source)
            img_seg_target = source_node.frame_observation.observed_segmentation.squeeze([0, 1]).permute(1, 2, 0).to('cuda')
            seg_source_nonzero = (img_seg_target[..., 0]).nonzero()
            edge_data = self.data_graph.get_edge_observations(edge_source, edge_target)

            if edge_data.reliability_score < 1 / 3 * self.tracking_config.flow_reliability_threshold:
                continue

            seg_source_nonzero_xy = seg_source_nonzero[..., [1, 0]].to('cuda')
            seg_target_nonzero_xy = seg_target_nonzero[..., [1, 0]].to('cuda')

            H, W = self.image_height, self.image_width
            src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(edge_data.flow_warp, H, W, H, W)

            src_pts_xy_roma = src_pts_xy_roma.to(torch.int).to('cuda')
            dst_pts_xy_roma = dst_pts_xy_roma.to(torch.int).to('cuda')

            src_pts_mask = ((seg_source_nonzero_xy.unsqueeze(0) == src_pts_xy_roma.unsqueeze(1)).all(-1)).any(1)
            dst_pts_mask = ((seg_target_nonzero_xy.unsqueeze(0) == dst_pts_xy_roma.unsqueeze(1)).all(-1)).any(1)
            src_pts_xy_roma_valid = src_pts_xy_roma[dst_pts_mask & src_pts_mask]
            dst_pts_xy_roma_valid = dst_pts_xy_roma[dst_pts_mask & src_pts_mask]

            src_pts_idxs = \
            torch.where((seg_source_nonzero_xy.unsqueeze(0) == src_pts_xy_roma_valid.unsqueeze(1)).all(-1))[1]
            dst_pts_idxs = \
            torch.where((seg_target_nonzero_xy.unsqueeze(0) == dst_pts_xy_roma_valid.unsqueeze(1)).all(-1))[1]

            # print(f'{src_pts_idxs.shape}, {dst_pts_idxs.shape}')

            matches = torch.stack([src_pts_idxs, dst_pts_idxs], dim=1)
            self.colmap_db.add_matches(edge_source + 1, edge_target + 1, matches.numpy(force=True).copy())
            # self.colmap_db.add_two_view_geometry(edge_source + 1, edge_target + 1, matches.numpy(force=True).copy())

        self.colmap_db.commit()

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


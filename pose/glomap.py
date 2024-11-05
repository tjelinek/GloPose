import subprocess
from pathlib import Path

import imageio
import numpy as np
import pycolmap
import torch
from kornia.geometry import PinholeCamera

from auxiliary_scripts.colmap_database import COLMAPDatabase
from auxiliary_scripts.image_utils import ImageShape
from data_structures.data_graph import DataGraph
from data_structures.pose_icosphere import PoseIcosphere
from tracker_config import TrackerConfig


class GlomapWrapper:

    def __init__(self, write_folder: Path, tracking_config: TrackerConfig, data_graph: DataGraph,
                 image_shape: ImageShape, pinhole_params: PinholeCamera, pose_icosphere: PoseIcosphere):
        self.write_folder = write_folder
        self.tracking_config = tracking_config
        self.colmap_image_path = (self.write_folder /
                                    f'icosphere_dump_{self.tracking_config.experiment_name}_'
                                    f'{self.tracking_config.sequence}')
        self.colmap_image_path.mkdir(exist_ok=True, parents=True)

        self.image_width = image_shape.width
        self.image_height = image_shape.height

        self.data_graph = data_graph
        self.pinhole_params = pinhole_params
        self.pose_icosphere = pose_icosphere

        self.colmap_db_path = self.colmap_image_path / 'database.db'
        self.colmap_output_path = self.colmap_image_path / 'output'
        self.colmap_db: COLMAPDatabase = COLMAPDatabase.connect(self.colmap_db_path)
        self.colmap_db.create_tables()

        colmap_db_camera_params = (np.array([self.pinhole_params.fx.item(), self.pinhole_params.fy.item(),
                                             self.pinhole_params.cx.item(), self.pinhole_params.cy.item()]).
                                   astype(np.float64))
        self.colmap_db.add_camera(1, self.image_width, self.image_height, colmap_db_camera_params, camera_id=1)

    def dump_icosphere_node_for_glomap(self, icosphere_node):
        frame_idx = icosphere_node.keyframe_idx_observed
        frame_data = self.data_graph.get_camera_specific_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze([0, 1]).permute(1, 2, 0)
        img *= img_seg

        node_save_path = self.colmap_image_path / f'node_{frame_idx}.png'
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8))

        seg_target_nonzero = img_seg[..., 0].nonzero()
        seg_target_nonzero_unit = seg_target_nonzero  #pixel_coords_to_unit_coords(self.image_width, self.image_height, seg_target_nonzero)
        seg_target_nonzero_xy_np = seg_target_nonzero_unit[..., [1, 0]].numpy(force=True).astype(np.float32)
        assert seg_target_nonzero_xy_np.dtype == np.float32

        # general_fame_data = self.data_graph.get_frame_data(frame_idx)
        # gt_t_obj = general_fame_data.gt_translation[None]
        # gt_r_obj = general_fame_data.gt_rot_axis_angle[None]
        # gt_Se3_obj = Se3(Quaternion.from_axis_angle(gt_r_obj), gt_t_obj)
        # Se3_world_to_cam = Se3.from_matrix(self.pinhole_params.extrinsics)
        # gt_Se3_cam = Se3_epipolar_cam_from_Se3_obj(gt_Se3_obj, Se3_world_to_cam)
        # gt_q_cam_np = gt_Se3_cam.quaternion.q.squeeze().numpy(force=True).copy()
        # gt_t_cam_np = gt_Se3_cam.t.squeeze().numpy(force=True).copy()

        self.colmap_db.add_image(name=f'./{str(node_save_path.name)}', camera_id=1, image_id=frame_idx + 1)
        self.colmap_db.add_keypoints(frame_idx + 1, seg_target_nonzero_xy_np.copy())

        icosphere_nodes_idx = {node.keyframe_idx_observed for node in self.pose_icosphere.reference_poses}
        incoming_edges = self.data_graph.G.in_edges(frame_idx)

        for edge in incoming_edges:
            edge_source = edge[0]
            edge_target = edge[1]

            if edge_source not in icosphere_nodes_idx:
                continue

            source_node = self.data_graph.get_camera_specific_frame_data(edge_source)
            img_seg_target = source_node.frame_observation.observed_segmentation.squeeze([0, 1]).permute(1, 2, 0)
            seg_source_nonzero = (img_seg_target[..., 0]).nonzero()
            edge_data = self.data_graph.get_edge_observations(edge_source, edge_target)

            src_pts = edge_data.src_pts_yx.to(torch.int)
            dst_pts = edge_data.dst_pts_yx.to(torch.int)

            dst_pts_mask = ((seg_target_nonzero.unsqueeze(0) == dst_pts.unsqueeze(1)).all(-1)).any(1)
            src_pts_indices = torch.where((seg_source_nonzero.unsqueeze(0) == src_pts.unsqueeze(1)).all(-1))[1]
            dst_pts_indices = torch.where((seg_target_nonzero.unsqueeze(0) == dst_pts.unsqueeze(1)).all(-1))[1]

            src_pts_indices_filtered = src_pts_indices[dst_pts_mask]

            matches = torch.stack([src_pts_indices_filtered, dst_pts_indices], dim=1)
            self.colmap_db.add_matches(edge_source + 1, edge_target + 1, matches.numpy(force=True).copy())
            # self.colmap_db.add_two_view_geometry(edge_source + 1, edge_target + 1, matches.numpy(force=True).copy())

        self.colmap_db.commit()

    def run_colmap(self):
        self.colmap_db.close()

        pycolmap.match_exhaustive(str(self.colmap_db_path))
        maps = pycolmap.incremental_mapping(str(self.colmap_db_path), str(self.colmap_image_path), str(self.colmap_output_path))

        maps[0].write(self.colmap_output_path)

        glomap_command = [
            "glomap",
            "mapper",
            "--database_path", self.colmap_db_path,
            "--output_path", self.colmap_output_path,
            "--image_path", self.colmap_image_path
        ]

        subprocess.run(glomap_command, check=True, capture_output=True, text=True)


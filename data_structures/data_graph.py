from dataclasses import dataclass, field
from typing import Any, List

import networkx as nx
import torch

from auxiliary_scripts.cameras import Cameras
from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, FrameObservation
from models.encoder import EncoderResult


@dataclass
class CommonFrameData:
    gt_rot_axis_angle: torch.Tensor = None
    gt_translation: torch.Tensor = None

    translations_during_optimization: List = field(default_factory=list)
    quaternions_during_optimization: List = field(default_factory=list)

    ransac_rot_obj_axis_angle: torch.Tensor = None

    frame_losses: Any = None
    encoder_result: EncoderResult = None


@dataclass
class CameraSpecificFrameData:

    frame_observation: FrameObservation = None
    renders: Any = None
    per_pixel_flow_error: Any = None

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'")


@dataclass
class CrossFrameData:
    # Optical Flow
    synthetic_flow_result: SyntheticFlowObservation = None
    observed_flow: FlowObservation = None

    src_pts_yx: torch.Tensor = None
    dst_pts_yx: torch.Tensor = None
    dst_pts_yx_gt: torch.Tensor = None
    ransac_inliers_mask: torch.Tensor = None
    ransac_inlier_ratio: float = None

    observed_flow_segmentation: torch.Tensor = None
    observed_visible_fg_points_mask: torch.Tensor = None
    gt_visible_fg_points_mask: torch.Tensor = None
    ransac_inliers: torch.Tensor = None
    ransac_outliers: torch.Tensor = None

    ransac_triangulated_points: torch.Tensor = None
    ransac_triangulated_points_gt_Rt: torch.Tensor = None
    ransac_triangulated_points_gt_Rt_gt_flow: torch.Tensor = None

    dust3r_point_cloud_im1: torch.Tensor = None
    dust3r_point_cloud_im2: torch.Tensor = None

    is_source_of_matching: bool = True


@dataclass
class DataGraph:
    used_cameras: List[Cameras]
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_new_frame(self, frame_idx: int) -> None:
        assert not self.G.has_node(frame_idx)

        self.G.add_node(frame_idx,
                        camera_specific_frame_data={camera: CameraSpecificFrameData() for camera in self.used_cameras},
                        frame_data=CommonFrameData())

    def add_new_arc(self, source_frame_idx: int, target_frame_idx: int) -> None:
        assert not self.G.has_edge(source_frame_idx, target_frame_idx)

        self.G.add_edge(source_frame_idx, target_frame_idx,
                        edge_observations={camera: CrossFrameData() for camera in self.used_cameras})

    def get_camera_specific_frame_data(self, frame_idx: int, camera: Cameras = Cameras.FRONTVIEW) \
            -> CameraSpecificFrameData:
        assert self.G.has_node(frame_idx)

        return self.G.nodes[frame_idx]['camera_specific_frame_data'][camera]

    def get_frame_data(self, frame_idx: int) -> CommonFrameData:
        assert self.G.has_node(frame_idx)

        return self.G.nodes[frame_idx]['frame_data']

    def get_edge_observations(self, source_frame_idx: int, target_frame_idx,
                              camera: Cameras = Cameras.FRONTVIEW) -> CrossFrameData:
        assert self.G.has_edge(source_frame_idx, target_frame_idx)

        return self.G.get_edge_data(source_frame_idx, target_frame_idx)['edge_observations'][camera]

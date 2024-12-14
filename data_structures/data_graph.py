from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Set

import networkx as nx
import torch
from kornia.geometry import Se3

from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, FrameObservation
from models.encoder import EncoderResult


@dataclass
class CommonFrameData:

    # Input data
    frame_observation: FrameObservation = None

    # Ground truth data
    gt_rot_axis_angle: torch.Tensor = None
    gt_translation: torch.Tensor = None
    gt_pose_cam: Se3 = None

    # Intermediate output data
    encoder_result: EncoderResult = None

    # Optimization
    frame_losses: Any = None
    translations_during_optimization: List = field(default_factory=list)
    quaternions_during_optimization: List = field(default_factory=list)

    # Long short jumps
    long_jump_source: int = None
    short_jump_source: int = None
    predicted_object_se3_long_jump: Se3 = Se3.identity(1, 'cuda')
    predicted_object_se3_total: Se3 = Se3.identity(1, 'cuda')
    predicted_object_se3_short_jump: Se3 = Se3.identity(1, 'cuda')
    predicted_obj_long_short_chain_diff: float = 0.0

    # Sources
    reliable_sources: Set[int] = field(default_factory=set)

    # Glomap
    image_save_path: Path = None
    segmentation_save_path: Path = None

    # Timings
    pose_estimation_time: float = None


@dataclass
class CrossFrameData:
    
    # Optical Flow observations
    synthetic_flow_result: SyntheticFlowObservation = None
    observed_flow: FlowObservation = None

    # Points and reliability
    src_pts_yx: torch.Tensor = None
    dst_pts_yx: torch.Tensor = None
    src_pts_xy_roma: torch.Tensor = None
    dst_pts_xy_roma: torch.Tensor = None
    dst_pts_yx_gt: torch.Tensor = None
    dst_pts_yx_chained: torch.Tensor = None
    remaining_pts_after_filtering: float = None
    ransac_inliers_mask: torch.Tensor = None
    ransac_inlier_ratio: float = None
    reliability_score: float = 0.0
    is_match_reliable: bool = False

    # RANSAC
    ransac_inliers: torch.Tensor = None
    ransac_outliers: torch.Tensor = None
    ransac_triangulated_points: torch.Tensor = None

    # Segmentation and masks
    adjusted_segmentation: torch.Tensor = None
    observed_visible_fg_points_mask: torch.Tensor = None
    gt_visible_fg_points_mask: torch.Tensor = None

    # Predicted SE3 transformations
    predicted_obj_delta_se3: Se3 = Se3.identity(1, 'cuda')
    predicted_obj_delta_se3_ransac: Se3 = Se3.identity(1, 'cuda')
    predicted_cam_delta_se3: Se3 = Se3.identity(1, 'cuda')
    predicted_cam_delta_se3_ransac: Se3 = Se3.identity(1, 'cuda')

    # Camera scaling
    camera_scale_per_axis_gt: torch.Tensor = None
    camera_scale_estimated: float = None


@dataclass
class DataGraph:
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_new_frame(self, frame_idx: int) -> None:
        assert not self.G.has_node(frame_idx)

        self.G.add_node(frame_idx, frame_data=CommonFrameData())

    def add_new_arc(self, source_frame_idx: int, target_frame_idx: int) -> None:
        assert not self.G.has_edge(source_frame_idx, target_frame_idx)

        self.G.add_edge(source_frame_idx, target_frame_idx,
                        edge_observations=CrossFrameData())

    def get_frame_data(self, frame_idx: int) -> CommonFrameData:
        assert self.G.has_node(frame_idx)

        return self.G.nodes[frame_idx]['frame_data']

    def get_edge_observations(self, source_frame_idx: int, target_frame_idx) -> CrossFrameData:
        assert self.G.has_edge(source_frame_idx, target_frame_idx)

        return self.G.get_edge_data(source_frame_idx, target_frame_idx)['edge_observations']

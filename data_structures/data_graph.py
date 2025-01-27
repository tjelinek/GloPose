from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

import networkx as nx
import torch
from kornia.geometry import Se3
from kornia.image import ImageSize

from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, FrameObservation


@dataclass
class CommonFrameData:

    # Input data
    frame_observation: FrameObservation = None

    # SIFT
    sift_keypoints: torch.Tensor = None
    sift_descriptors: torch.Tensor = None
    sift_lafs: torch.Tensor = None

    # Ground truth data
    gt_obj1_to_obji: Se3 = None
    gt_pose_cam: Se3 = None
    gt_pinhole_K: torch.Tensor = None
    image_shape: ImageSize = None

    # Long short jumps
    matching_source_keyframe: int = None
    predicted_object_se3_long_jump: Se3 = Se3.identity(1, 'cuda')
    predicted_object_se3_total: Se3 = Se3.identity(1, 'cuda')

    # Sources
    reliable_sources: Set[int] = field(default_factory=set)

    # Filename
    image_filename: Path = None
    segmentation_filename: Path = None

    # Glomap
    image_save_path: Path = None
    segmentation_save_path: Path = None

    # Timings
    pose_estimation_time: float = None

    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu()  # Move the tensor to CPU
        super().__setattr__(name, value)


@dataclass
class CrossFrameData:
    
    # Optical Flow observations
    synthetic_flow_result: SyntheticFlowObservation = None
    observed_flow: FlowObservation = None

    roma_flow_warp: torch.Tensor = None  # [W, H] format
    roma_flow_certainty: torch.Tensor = None  # [W, H] format
    src_pts_xy_roma: torch.Tensor = None
    dst_pts_xy_roma: torch.Tensor = None

    sift_keypoint_indices: torch.Tensor = None
    sift_dists: torch.Tensor = None

    # Points and reliability
    reliability_score: float = 0.0
    is_match_reliable: bool = False

    # RANSAC
    ransac_inliers: torch.Tensor = None
    ransac_outliers: torch.Tensor = None

    # Predicted SE3 transformations
    predicted_obj_delta_se3: Se3 = Se3.identity(1, 'cuda')
    predicted_cam_delta_se3: Se3 = Se3.identity(1, 'cuda')

    # SIFT
    num_matches: int = None

    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.cpu()  # Move the tensor to CPU
        super().__setattr__(name, value)


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

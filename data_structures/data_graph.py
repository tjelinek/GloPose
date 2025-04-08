from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

import networkx as nx
import torch
from kornia.geometry import Se3
from kornia.image import ImageSize

from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, FrameObservation, Observation


@dataclass
class DataGraphStorage:
    def __setattr__(self, name, value):
        if isinstance(value, torch.Tensor):
            value = value.to(self._storage_device)
        elif isinstance(value, Observation):
            value = value.send_to_device(self._storage_device)
        super().__setattr__(name, value)

    def __getattribute__(self, name):
        super_val = super().__getattribute__(name)
        if isinstance(super_val, torch.Tensor):
            return super_val.to(self._out_device)
        elif isinstance(super_val, Observation):
            return super_val.send_to_device(self._out_device)
        return super_val


@dataclass
class CommonFrameData(DataGraphStorage):
    _storage_device: str = 'cpu'
    _out_device: str = 'cpu'

    # Input data
    frame_observation: FrameObservation = None

    # SIFT
    sift_keypoints: torch.Tensor = None
    sift_descriptors: torch.Tensor = None
    sift_lafs: torch.Tensor = None

    # Ground truth data
    gt_Se3_cam2obj: Se3 = None
    gt_Se3_world2cam: Se3 = None
    gt_pinhole_K: torch.Tensor = None
    image_shape: ImageSize = None

    # Long short jumps
    matching_source_keyframe: int = None
    pred_Se3_cam2obj: Se3 = None

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


@dataclass
class CrossFrameData(DataGraphStorage):

    _storage_device: str = 'cpu'
    _out_device: str = 'cpu'

    # Optical Flow observations
    synthetic_flow_result: SyntheticFlowObservation = None
    observed_flow: FlowObservation = None

    roma_flow_warp: torch.Tensor = None  # [W, H] format
    roma_flow_warp_certainty: torch.Tensor = None  # [W, H] format

    src_pts_xy_roma: torch.Tensor = None
    dst_pts_xy_roma: torch.Tensor = None
    src_dst_certainty_roma: torch.Tensor = None

    sift_keypoint_indices: torch.Tensor = None
    sift_dists: torch.Tensor = None

    # Points and reliability
    reliability_score: float = 0.0
    is_match_reliable: bool = False

    # RANSAC
    ransac_inliers: torch.Tensor = None
    ransac_outliers: torch.Tensor = None

    # SIFT
    num_matches: int = None


class DataGraph:

    def __init__(self, out_device: 'str', storage_device: 'str' = 'cpu'):
        self.G: nx.DiGraph = nx.DiGraph()
        self.out_device: str = out_device
        self.storage_device: str = storage_device

    def add_new_frame(self, frame_idx: int) -> None:
        assert not self.G.has_node(frame_idx)

        self.G.add_node(frame_idx, frame_data=CommonFrameData(_storage_device=self.storage_device,
                                                              _out_device=self.out_device))

    def add_new_arc(self, source_frame_idx: int, target_frame_idx: int) -> None:
        assert not self.G.has_edge(source_frame_idx, target_frame_idx)

        self.G.add_edge(source_frame_idx, target_frame_idx,
                        edge_observations=CrossFrameData(_storage_device=self.storage_device,
                                                         _out_device=self.out_device))

    def get_frame_data(self, frame_idx: int) -> CommonFrameData:
        assert self.G.has_node(frame_idx)

        return self.G.nodes[frame_idx]['frame_data']

    def get_edge_observations(self, source_frame_idx: int, target_frame_idx) -> CrossFrameData:
        assert self.G.has_edge(source_frame_idx, target_frame_idx)

        return self.G.get_edge_data(source_frame_idx, target_frame_idx)['edge_observations']

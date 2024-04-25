from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple, List

import networkx as nx
import torch

from models.encoder import EncoderResult
from models.rendering import RenderedFlowResult


class Cameras(Enum):
    FRONTVIEW = 'frontview'
    BACKVIEW = 'backview'


@dataclass
class FrameResult:
    flow_render_result: RenderedFlowResult = None
    encoder_result: EncoderResult = None
    renders: Any = None
    frame_losses: Any = None
    per_pixel_flow_error: Any = None
    src_pts_yx_front: torch.Tensor = None
    dst_pts_yx_front: torch.Tensor = None
    dst_pts_yx_front_gt: torch.Tensor = None
    inliers_mask_front: torch.Tensor = None
    observed_flow_segmentation_front: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    observed_flow_fg_occlusion_front: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    inliers_front: Dict = field(default_factory=dict)
    outliers_front: Dict = field(default_factory=dict)
    src_pts_yx_back: torch.Tensor = None
    dst_pts_yx_back: torch.Tensor = None
    dst_pts_yx_back_gt: torch.Tensor = None
    inliers_mask_back: torch.Tensor = None
    inliers_back: Dict = field(default_factory=dict)
    outliers_back: Dict = field(default_factory=dict)
    observed_flow_segmentation_back: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    observed_flow_fg_occlusion_back: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    triangulated_points_frontview: Dict = None
    triangulated_points_backview: Dict = None
    source_of_matching: bool = True

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'")


@dataclass
class CrossFrameData:

    flow_render_result: RenderedFlowResult = None
    encoder_result: EncoderResult = None
    renders: Any = None
    frame_losses: Any = None
    per_pixel_flow_error: Any = None
    src_pts_yx_front: torch.Tensor = None
    dst_pts_yx_front: torch.Tensor = None
    inliers_mask_front: torch.Tensor = None
    observed_flow_segmentation_front: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    observed_flow_fg_occlusion_front: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    inliers_front: Dict = field(default_factory=dict)
    outliers_front: Dict = field(default_factory=dict)
    src_pts_yx_back: torch.Tensor = None
    dst_pts_yx_back: torch.Tensor = None
    inliers_mask_back: torch.Tensor = None
    inliers_back: Dict = field(default_factory=dict)
    outliers_back: Dict = field(default_factory=dict)
    observed_flow_segmentation_back: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    observed_flow_fg_occlusion_back: Dict[Tuple, torch.Tensor] = field(default_factory=dict)
    triangulated_points_frontview: Dict = None
    triangulated_points_backview: Dict = None
    source_of_matching: bool = True



@dataclass
class EssentialMatrixData:
    camera_rotations = {}
    camera_translations = {}
    source_points = {}
    target_points = {}
    inlier_mask = {}
    triangulated_points = {}


@dataclass
class DataGraph:
    used_cameras: List[Cameras]
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

    def add_new_frame(self, frame_idx: int) -> None:
        assert not self.G.has_node(frame_idx)

        self.G.add_node(frame_idx, frame_observations={camera: FrameResult() for camera in self.used_cameras})

    def add_new_arc(self, source_frame_idx: int, target_frame_idx: int) -> None:
        assert not self.G.has_edge(source_frame_idx, target_frame_idx)

        self.G.add_edge(source_frame_idx, target_frame_idx,
                        edge_observations={camera: CrossFrameData() for camera in self.used_cameras})

    def get_frame_observations(self, frame_idx: int, camera: Cameras = Cameras.FRONTVIEW) -> FrameResult:
        assert self.G.has_node(frame_idx)

        return self.G.nodes[frame_idx]['frame_observations'][camera]

    def get_edge_observations(self, source_frame_idx: int, target_frame_idx,
                              camera: Cameras = Cameras.FRONTVIEW) -> CrossFrameData:
        assert self.G.has_edge(source_frame_idx, target_frame_idx)

        return self.G.get_edge_data(source_frame_idx, target_frame_idx)['frame_observations'][camera]

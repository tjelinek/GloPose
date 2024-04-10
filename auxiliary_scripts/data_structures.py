from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

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

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"{self.__class__.__name__} has no attribute '{key}'")


@dataclass
class EssentialMatrixData:
    camera_rotations = {}
    camera_translations = {}
    source_points = {}
    target_points = {}
    inlier_mask = {}


@dataclass
class KeyframeBuffer:
    G: nx.DiGraph = field(default_factory=nx.DiGraph)

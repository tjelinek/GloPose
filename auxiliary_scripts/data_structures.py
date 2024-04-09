from dataclasses import dataclass
from typing import Any, Dict

from models.encoder import EncoderResult
from models.rendering import RenderedFlowResult


@dataclass
class FrameResult:
    flow_render_result: RenderedFlowResult = None
    encoder_result: EncoderResult = None
    renders: Any = None
    frame_losses: Any = None
    per_pixel_flow_error: Any = None
    inliers: Dict = None
    outliers: Dict = None
    inliers_back: Dict = None
    outliers_back: Dict = None
    triangulated_points_frontview: Dict = None
    triangulated_points_backview: Dict = None


@dataclass
class EssentialMatrixData:
    camera_rotations = {}
    camera_translations = {}
    source_points = {}
    target_points = {}
    inlier_mask = {}

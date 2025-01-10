from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from kornia.image import ImageSize
from configs.matching_configs.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching_configs.sift_configs.base_sift_config import BaseSiftConfig


@dataclass
class TrackerConfig:
    # General settings
    device: str = 'cuda'
    default_data_folder: Path = Path('/mnt/personal/jelint19/data/').expanduser()
    default_results_folder: Path = Path('/mnt/personal/jelint19/results/FlowTracker/').expanduser()
    write_folder: Path = None

    matching_visualization_type: str = 'matching'  # Either 'dots' or 'matching'
    dataset: str = None
    sequence: str = None
    experiment_name: str = None

    # Visualization settings
    write_to_rerun_rather_than_disk: bool = True

    # Input data settings
    input_frames: int = 0
    generate_synthetic_observations_if_possible: bool = True
    frame_provider: str = 'synthetic'  # 'precomputed' or 'synthetic'
    segmentation_provider: str = 'SAM2'  # 'precomputed', 'SAM2' or 'synthetic'
    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'

    # Camera settings
    camera_position: Tuple[float] = (0, 0, 5.0)
    camera_up: Tuple[float] = (0, 1, 0)
    rendered_image_shape: ImageSize = ImageSize(500, 500)
    image_downsample: float = 1.0
    sigmainv: float = 7000
    image_size: ImageSize = None
    features: str = 'deep'

    # Mesh settings
    mesh_normalize: bool = False
    texture_size: int = 1000
    use_lights: bool = False
    gt_mesh_path: str = None
    optimize_shape: bool = False
    gt_texture_path: str = None

    # Tracking initialization
    tran_init: Tuple[float] = None  # (0., 0., 0.)
    rot_init: Tuple[float] = None   # (0., 0., 0.)

    # Ground truth tracking
    gt_track_path: str = None
    augment_gt_track: bool = False

    # Optical flow and segmentation settings
    segmentation_mask_erosion_iters: int = 3
    segmentation_mask_threshold: float = 0.99
    occlusion_coef_threshold: float = 0.95

    # Matcher configurations
    roma_matcher_config: BaseRomaConfig = field(default_factory=BaseRomaConfig)
    roma_sample_size: int = 10000
    min_roma_certainty_threshold: float = 0.95
    flow_reliability_threshold: float = 0.5
    min_flow_reliability: float = 0.25
    min_number_of_reliable_matches: int = 0
    mapper: str = 'pycolmap'
    frame_filter: str = 'SIFT'
    reconstruction_matches: str = 'RoMa'

    # SIFT options
    sift_matcher_config: BaseSiftConfig = field(default_factory=BaseSiftConfig)
    sift_filter_min_matches: int = 100
    sift_filter_good_to_add_matches: int = 450
    sift_cache: Path = None
    sift_mapping_num_feats: int = 8192
    sift_mapping_min_matches: int = 15
    sift_mapping_single_camera: bool = True

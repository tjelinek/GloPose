from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import numpy as np
from kornia.image import ImageSize

from configs.matching_configs.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching_configs.sift_configs.base_sift_config import BaseSiftConfig


@dataclass
class TrackerConfig:

    # General settings
    device = 'cuda'
    features: str = 'deep'
    verbose: bool = True
    matching_visualization_type = 'matching'  # Either 'dots' or 'matching'

    # Visualization

    write_to_rerun_rather_than_disk: bool = True
    plot_mft_flow_kde_error_plot: bool = True
    analyze_ransac_matchings: bool = True
    analyze_ransac_matching_errors: bool = False
    visualize_outliers_distribution: bool = False

    analyze_ransac_matchings_frequency: int = 5
    mft_flow_kde_error_plot_frequency: int = 10
    write_folder: Path = None

    # Frame and keyframe settings
    input_frames: int = 0

    # Mesh settings
    mesh_normalize: bool = False
    texture_size: int = 1000
    use_lights: bool = False

    # Camera settings
    camera_position: Tuple[float] = (0, 0, 5.0)
    camera_up: Tuple[float] = (0, 1, 0)
    max_width: int = 500
    image_downsample: float = 1.0
    image_size: ImageSize = None

    # Tracking settings
    tran_init: Tuple[float] = None  # (0., 0., 0.)
    rot_init: Tuple[float] = None   # (0., 0., 0.)
    camera_intrinsics: np.ndarray = None
    camera_extrinsics: np.ndarray = None

    # Loss function coefficients
    loss_laplacian_weight: float = 1000.0
    loss_tv_weight: float = 0.001
    loss_iou_weight: float = 1.0
    loss_dist_weight: float = 0
    loss_rgb_weight: float = 1.0
    loss_flow_weight: float = 10.0
    occlusion_coef_threshold = 0.95  # Above this value, points will be considered as occluded
    segmentation_mask_threshold = 0.99

    # Additional settings
    sigmainv: float = 7000
    dataset: str = None
    sequence: str = None
    experiment_name: str = None

    # Ground truths
    gt_mesh_path: str = None
    optimize_shape: bool = False

    gt_texture_path: str = None

    gt_track_path: str = None
    augment_gt_track: bool = False

    # Input Data
    generate_synthetic_observations_if_possible: bool = True
    frame_provider: str = 'synthetic'  # 'precomputed' or 'synthetic'
    segmentation_provider: str = 'SAM2'  # 'precomputed', 'SAM2' or 'synthetic'

    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    # long_flow_model: str = 'MFT_IQ'   # 'MFT', 'MFT_IQ', 'MFT_SynthFlow' None
    long_flow_model: str = 'MFT'
    # long_flow_model: str = 'MFT_Synth'
    MFT_synth_add_noise: bool = False
    MFT_synth_noise_sigma: float = 0.3
    MFT_synth_noise_mu: float = 0.0
    MFT_backbone_cfg: str = 'MFT_RoMa_direct_cfg'
    # MFT_backbone_cfg = 'MFTIQ_SYNTHETIC_bs3_bce_200k_kubric_binary_cfg'

    # Optical flow settings
    segmentation_mask_erosion_iters: int = 3
    # Pre-initialization method: One of 'levenberg-marquardt', 'gradient_descent', 'coordinate_descent',
    #                                   'essential_matrix_decomposition' or None
    preinitialization_method: str = 'essential_matrix_decomposition'

    # RANSAC settings
    ransac_inlier_filter: str = 'pygcransac'  # 'magsac++', 'ransac', '8point', 'pygcransac', 'pnp_ransac'

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

    sift_mapping_num_feats = 8192,
    sift_mapping_min_matches = 15,
    sift_mapping_single_camera = True

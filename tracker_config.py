from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TrackerConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # General settings
    device = 'cuda'
    features: str = 'deep'
    features_channels = 64
    verbose: bool = True
    matching_visualization_type = 'matching'  # Either 'dots' or 'matching'

    # Visualization

    write_results: bool = True
    write_to_rerun_rather_than_disk: bool = True
    write_results_frequency: int = 1
    plot_mft_flow_kde_error_plot: bool = True
    analyze_ransac_matchings: bool = True
    analyze_ransac_matching_errors: bool = False
    visualize_outliers_distribution: bool = False

    analyze_ransac_matchings_frequency: int = 5
    mft_flow_kde_error_plot_frequency: int = 10

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 1
    all_frames_keyframes: bool = False

    # Mesh settings
    mesh_normalize: bool = False
    texture_size: int = 1000
    use_lights: bool = False

    # Camera settings
    camera_position: Tuple[float] = (0, 0, 5.0)
    camera_up: Tuple[float] = (0, 1, 0)
    max_width: int = 500
    image_downsample: float = 1.0

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
    initial_mesh_path: str = 'prototypes/sphere.obj'
    gt_mesh_path: str = None
    optimize_shape: bool = False

    gt_texture_path: str = None
    optimize_texture: bool = False

    gt_track_path: str = None
    augment_gt_track: bool = False
    optimize_pose: bool = True

    # Input Data
    generate_synthetic_observations_if_possible: bool = True
    segmentation_tracker: str = 'precomputed'  # 'precomputed', 'SAM', 'SAM2' or 'XMem'

    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    # long_flow_model: str = 'MFT_IQ'   # 'MFT', 'MFT_IQ', 'MFT_SynthFlow' None
    long_flow_model: str = 'MFT'
    # long_flow_model: str = 'MFT_Synth'
    MFT_synth_add_noise: bool = False
    MFT_synth_noise_sigma: float = 0.3
    MFT_synth_noise_mu: float = 0.0
    MFT_backbone_cfg: str = 'MFT_RoMa_direct_cfg'
    # MFT_backbone_cfg = 'MFTIQ_SYNTHETIC_bs3_bce_200k_kubric_binary_cfg'

    matching_target_to_backview: bool = False

    # Optical flow settings
    add_flow_arcs_strategy: str = 'single-previous'  # One of 'all-previous', 'single-previous' and 'absolute'
    # The 'all-previous' strategy for current frame i adds arcs (j, i) forall frames j < i, while 'single-previous' adds
    # only arc (i - 1, i).N
    segmentation_mask_erosion_iters: int = 3
    # Pre-initialization method: One of 'levenberg-marquardt', 'gradient_descent', 'coordinate_descent',
    #                                   'essential_matrix_decomposition' or None
    preinitialization_method: str = 'essential_matrix_decomposition'

    # RANSAC settings
    ransac_inlier_filter: str = 'pygcransac'  # 'magsac++', 'ransac', '8point', 'pygcransac', 'pnp_ransac'
    ransac_inlier_pose_method: str = 'zaragoza'  # 'zaragoza', '8point', 'numerical_E_optimization'
    ransac_refine_E_numerically: bool = False

    long_short_flow_chaining_pixel_level_verification: bool = False

    ransac_outlier_threshold: float = 0.01
    ransac_min_iters: int = 10000
    ransac_confidence: float = 0.9999

    ransac_use_gt_occlusions_and_segmentation: bool = False
    ransac_dilate_occlusion: bool = False
    ransac_erode_segmentation: bool = False

    ransac_use_dust3r: bool = False

    ransac_sample_points: bool = True
    ransac_sampled_points_number: int = 100

    ransac_feed_only_inlier_flow: bool = False
    ransac_feed_only_inlier_flow_epe_threshold: float = 1.0

    ransac_replace_mft_flow_with_gt_flow: bool = False
    ransac_feed_gt_flow_percentage: float = 0.75
    ransac_feed_gt_flow_add_gaussian_noise: bool = False
    ransac_feed_gt_flow_add_gaussian_noise_use_mft_errors: bool = False
    ransac_feed_gt_flow_add_gaussian_noise_sigma: float = 2.5
    ransac_feed_gt_flow_add_gaussian_noise_mean: float = 0.5

    ransac_distant_pixels_sampling: bool = False
    ransac_distant_pixels_sample_size: int = 1000

    ransac_confidences_from_occlusion: bool = False

    roma_sample_size: int = 10000

    # Icosphere templates
    icosphere_use_gt_long_jumps: bool = True
    icosphere_trust_region_degrees = 20

    flow_reliability_threshold: float = 0.5
    frame_filter_when_lost_algorithm = None
    frame_reconstruction_algorithm: str = 'glomap'
    matcher: str = 'RoMa'

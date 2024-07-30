from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrackerConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # General settings
    tracker_type: str = 'd3s'
    features: str = 'deep'
    features_channels = 64
    verbose: bool = True
    matching_visualization_type = 'matching'  # Either 'dots' or 'matching'

    # Visualization
    write_results: bool = True
    write_to_rerun_rather_than_disk: bool = True
    write_results_frequency: int = 1
    visualize_loss_landscape: bool = False
    plot_mft_flow_kde_error_plot: bool = True
    dump_correspondences: bool = False
    save_3d_model: bool = False
    visualize_point_clouds_from_ransac: bool = False
    analyze_ransac_matchings: bool = True
    analyze_ransac_matching_errors: bool = False
    visualize_outliers_distribution: bool = False

    analyze_ransac_matchings_frequency: int = 5
    loss_landscape_visualization_frequency: int = 18
    training_print_status_frequency = 20
    mft_flow_kde_error_plot_frequency: int = 10

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 1
    all_frames_keyframes: bool = False
    fmo_steps: int = 1

    # Mesh settings
    mesh_size: int = 11
    mesh_normalize: bool = False
    texture_size: int = 1000
    use_lights: bool = False

    # Camera settings
    camera_position: Tuple[float] = (0, 0, 5.0)
    camera_up: Tuple[float] = (0, 1, 0)
    max_width: int = 500
    image_downsample: float = 1.0

    # Learning rates
    learning_rate: float = 0.1
    quaternion_learning_rate_coef: float = 1.0
    translation_learning_rate_coef: float = 1.0

    # Tracking settings
    tran_init: Tuple[float] = (0., 0., 0.)
    rot_init: Tuple[float] = (0, 0, 0)
    iterations: int = 100
    stop_value: float = 0.05
    rgb_iters: int = 10
    project_coin: bool = False
    connect_frames: bool = False
    accumulate: bool = False
    mot_opt_all: bool = True
    motion_only_last: bool = True

    # Loss function coefficients
    loss_laplacian_weight: float = 1000.0
    loss_tv_weight: float = 0.001
    loss_iou_weight: float = 1.0
    loss_dist_weight: float = 0
    loss_q_weight: float = 1.0
    loss_t_weight: float = 1.0
    loss_rgb_weight: float = 1.0
    loss_flow_weight: float = 10.0
    loss_fl_obs_and_rend_weight: float = None
    loss_fl_not_obs_rend_weight: float = None
    loss_fl_obs_not_rend_weight: float = None
    occlusion_coef_threshold = 0.95  # Above this value, points will be considered as occluded
    segmentation_mask_threshold = 0.99

    # Additional settings
    sigmainv: float = 7000
    factor: float = 1
    mask_iou_th: float = 0
    rotation_divide: int = 8
    sequence: str = None
    experiment_name: str = None
    max_rendering_batch_size: int = 4

    # Ground truths
    initial_mesh_path: str = 'prototypes/sphere.obj'
    gt_mesh_path: str = None
    optimize_shape: bool = False

    gt_texture_path: str = None
    optimize_texture: bool = False
    # Either 'sgd' for stochastic gradient descent learning or 'sticking' for using the rendered texture uv coordinates
    texture_estimation: str = 'sgd'

    gt_track_path: str = None
    augment_gt_track: bool = False
    optimize_pose: bool = True

    # Input Data
    generate_synthetic_observations_if_possible: bool = True
    segmentation_tracker: str = 'SAM2'  # 'precomputed', 'SAM', 'SAM2' or 'XMem'

    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    short_flow_model: str = 'RAFT'  # 'RAFT' 'GMA'
    # long_flow_model: str = 'MFT_IQ'   # 'MFT', 'MFT_IQ', 'MFT_SynthFlow' None
    long_flow_model: str = 'MFT'
    # long_flow_model: str = 'MFT_Synth'
    MFT_synth_add_noise: bool = False
    MFT_synth_noise_sigma: float = 0.3
    MFT_synth_noise_mu: float = 0.0
    # MFT_backbone_cfg: str = 'MFTIQ_ROMA_bs3_bce_200k_kubric_binary_cfg'
    MFT_backbone_cfg: str = 'MFT_RoMa_direct_cfg'
    # MFT_backbone_cfg = 'MFTIQ_SYNTHETIC_bs3_bce_200k_kubric_binary_cfg'

    matching_target_to_backview: bool = False

    # Optimization
    allow_break_sgd_after = 30
    break_sgd_after_iters_with_no_change = 20
    optimize_non_positional_params_after = 70
    levenberg_marquardt_max_ter = 15
    use_lr_scheduler = False
    lr_scheduler_patience = 5
    run_main_optimization_loop: bool = False

    # Optical flow settings
    add_flow_arcs_strategy: str = None  # One of 'all-previous', 'single-previous' and 'absolute'
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

    ransac_outlier_threshold: float = 0.01
    ransac_min_iters: int = 1000
    ransac_confidence: float = 0.9999

    ransac_use_gt_occlusions_and_segmentation: bool = False
    ransac_dilate_occlusion: bool = False
    ransac_erode_segmentation: bool = True

    ransac_use_dust3r: bool = False

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

    levenberg_marquardt_implementation: str = 'custom'  # Either 'custom' or 'ceres'
    use_custom_jacobian: bool = False
    flow_sgd: bool = True
    flow_sgd_n_samples: int = 100

    # Icosphere templates
    icosphere_trust_region_degrees = 45
    icosphere_add_inplane_rotatiosn = False

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
    verbose: bool = True
    write_results: bool = True
    write_intermediate: bool = False
    visualize_loss_landscape: bool = True
    render_just_bounding_box: bool = False
    training_print_status_frequency = 20

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 8
    all_frames_keyframes: bool = False
    fmo_steps: int = 1
    stochastically_add_keyframes: bool = False

    # Mesh settings
    mesh_size: int = 11
    mesh_normalize: bool = False
    texture_size: int = 300
    use_lights: bool = False

    # Camera settings
    camera_distance: float = 5
    max_width: int = 500
    image_downsample: float = 1.0

    # Learning rates
    learning_rate: float = 0.1
    quaternion_learning_rate_coef: float = 1.0
    translation_learning_rate_coef: float = 1.0

    # Tracking settings
    tran_init: float = 0.0
    rot_init: Tuple[float] = (0, 0, 0)
    inc_step: int = 5
    iterations: int = 500
    stop_value: float = 0.05
    rgb_iters: int = 200
    project_coin: bool = False
    connect_frames: bool = False
    accumulate: bool = False
    weight_by_gradient: bool = False
    mot_opt_all: bool = True
    motion_only_last: bool = True

    # Loss function coefficients
    loss_laplacian_weight: float = 1000.0
    loss_tv_weight: float = 0.001
    loss_iou_weight: float = 1.0
    loss_dist_weight: float = 0
    loss_q_weight: float = 1.0
    loss_texture_change_weight: float = 0
    loss_t_weight: float = 1.0
    loss_rgb_weight: float = 1.0
    loss_flow_weight: float = 10.0
    loss_fl_obs_and_rend_weight: float = None
    loss_fl_not_obs_rend_weight: float = None
    loss_fl_obs_not_rend_weight: float = None

    # Additional settings
    sigmainv: float = 7000
    factor: float = 1
    mask_iou_th: float = 0
    erode_renderer_mask: int = 3
    rotation_divide: int = 8
    sequence: str = None

    # Ground truths
    initial_mesh_path: str = 'prototypes/sphere.obj'
    gt_mesh_path: str = None
    optimize_shape: bool = True

    gt_texture_path: str = None
    optimize_texture: bool = True

    gt_track_path: str = None
    optimize_pose: bool = True

    generate_synthetic_observations_if_possible: bool = True

    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    short_flow_model: str = 'GMA'  # 'RAFT' 'GMA'
    long_flow_model: str = 'MFT'   # 'MFT' or None

    # Optimization
    allow_break_sgd_after = 30
    break_sgd_after_iters_with_no_change = 20
    optimize_non_positional_params_after = 70
    levenberg_marquardt_max_ter = allow_break_sgd_after
    use_lr_scheduler = False
    lr_scheduler_patience = 5

    # Optical flow settings
    add_flow_arcs_strategy: str = 'single-previous'  # One of 'all-previous', 'single-previous' and 'absolute'
    # The 'all-previous' strategy for current frame i adds arcs (j, i) forall frames j < i, while 'single-previous' adds
    # only arc (i - 1, i).
    segmentation_mask_erosion_iters: int = 0
    # Pre-initialization method: One of 'levenberg-marquardt', 'gradient_descent', 'coordinate_descent' and 'lbfgs'
    preinitialization_method: str = 'levenberg-marquardt'
    flow_sgd: bool = False
    flow_sgd_n_samples: int = 100

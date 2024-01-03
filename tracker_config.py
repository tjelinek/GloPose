from dataclasses import dataclass
from typing import List


@dataclass
class TrackerConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # General settings
    tracker_type: str = 'ostrack'
    features: str = 'deep'
    verbose: bool = True
    write_results: bool = True
    write_intermediate: bool = False
    visualize_loss_landscape: bool = True
    render_just_bounding_box: bool = False
    training_print_status_frequency = 1

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 0
    keyframes: int = None
    all_frames_keyframes: bool = False
    fmo_steps: int = 1
    stochastically_add_keyframes: bool = False

    # Mesh settings
    mesh_size: int = None
    mesh_normalize: bool = None
    texture_size: int = None
    use_lights: bool = None

    # Camera settings
    camera_distance: float = None
    max_width: int = None
    image_downsample: float = None

    # Learning rates
    learning_rate: float = None
    quaternion_learning_rate_coef: float = 1.0
    translation_learning_rate_coef: float = 1.0

    # Tracking settings
    tran_init: float = None
    rot_init: List[float] = None
    inc_step: 0 = 0
    iterations: int = None
    stop_value: float = None
    rgb_iters: int = None
    project_coin: bool = None
    connect_frames: bool = None
    accumulate: bool = None
    weight_by_gradient: bool = None
    mot_opt_all: bool = None
    motion_only_last: bool = None

    # Loss function coefficients
    loss_laplacian_weight: float = None
    loss_tv_weight: float = None
    loss_iou_weight: float = None
    loss_dist_weight: float = None
    loss_q_weight: float = None
    loss_texture_change_weight: float = None
    loss_t_weight: float = None
    loss_rgb_weight: float = None
    loss_flow_weight: float = None
    loss_fl_obs_and_rend_weight: float = None
    loss_fl_not_obs_rend_weight: float = None
    loss_fl_obs_not_rend_weight: float = None

    # Additional settings
    sigmainv: float = None
    factor: float = None
    mask_iou_th: float = None
    erode_renderer_mask: int = None
    rotation_divide: int = None
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

    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic', 'FromFiles'
    flow_model: str = 'GMA'  # 'RAFT' 'GMA' and 'MFT'

    # Optimization
    allow_break_sgd_after = 30
    break_sgd_after_iters_with_no_change = 20
    optimize_non_positional_params_after = 70
    levenberg_marquardt_max_ter = allow_break_sgd_after
    use_lr_scheduler = False
    lr_scheduler_patience = 5

    # Optical flow settings
    add_flow_arcs_strategy: str = 'single-previous'  # One of 'all-previous' and 'single-previous'
    # The 'all-previous' strategy for current frame i adds arcs (j, i) forall frames j < i, while 'single-previous' adds
    # only arc (i - 1, i).
    segmentation_mask_erosion_iters: int = 0
    # Pre-initialization method: One of 'levenberg-marquardt', 'gradient_descent', 'coordinate_descent' and 'lbfgs'
    preinitialization_method: str = 'levenberg-marquardt'
    flow_sgd: bool = True
    flow_sgd_n_samples: int = 100

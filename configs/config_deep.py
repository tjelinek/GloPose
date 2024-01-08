from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    # General settings
    cfg.tracker_type = 'd3s'
    cfg.features = 'deep'
    cfg.verbose = True
    cfg.write_results = True
    cfg.write_intermediate = False
    cfg.visualize_loss_landscape = True
    cfg.render_just_bounding_box = False
    cfg.training_print_status_frequency = 20

    # Frame and keyframe settings
    cfg.input_frames = 0
    cfg.max_keyframes = 8
    cfg.all_frames_keyframes = False
    cfg.fmo_steps = 1
    cfg.stochastically_add_keyframes = False

    # Mesh settings
    cfg.mesh_size = 11
    cfg.mesh_normalize = False
    cfg.texture_size = 300
    cfg.use_lights = False

    # Camera settings
    cfg.camera_distance = 5
    cfg.max_width = 500
    cfg.image_downsample = 1.0

    # Learning rates
    cfg.learning_rate = 0.1
    cfg.quaternion_learning_rate_coef = 1.0
    cfg.translation_learning_rate_coef = 1.0

    # Tracking settings
    cfg.tran_init = 0.0
    cfg.rot_init = (0, 0, 0)
    cfg.inc_step = 5
    cfg.iterations = 500
    cfg.stop_value = 0.05
    cfg.rgb_iters = 200
    cfg.project_coin = False
    cfg.connect_frames = False
    cfg.accumulate = False
    cfg.weight_by_gradient = False
    cfg.mot_opt_all = True
    cfg.motion_only_last = True

    # Loss function coefficients
    cfg.loss_laplacian_weight = 1000.0
    cfg.loss_tv_weight = 0.001
    cfg.loss_iou_weight = 1.0
    cfg.loss_dist_weight = 0
    cfg.loss_q_weight = 1.0
    cfg.loss_texture_change_weight = 0
    cfg.loss_t_weight = 1.0
    cfg.loss_rgb_weight = 1.0
    cfg.loss_flow_weight = 10.0
    cfg.loss_fl_obs_and_rend_weight = None
    cfg.loss_fl_not_obs_rend_weight = None
    cfg.loss_fl_obs_not_rend_weight = None

    # Additional settings
    cfg.sigmainv = 7000
    cfg.factor = 1
    cfg.mask_iou_th = 0
    cfg.erode_renderer_mask = 3
    cfg.rotation_divide = 8
    cfg.sequence = None

    # Ground truths
    cfg.initial_mesh_path = 'prototypes/sphere.obj'
    cfg.gt_mesh_path = None
    cfg.optimize_shape = True

    cfg.gt_texture_path = None
    cfg.optimize_texture = True

    cfg.gt_track_path = None
    cfg.optimize_pose = True

    cfg.generate_synthetic_observations_if_possible = True

    cfg.gt_flow_source = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    cfg.short_flow_model = 'GMA'  # 'RAFT' 'GMA'
    cfg.long_flow_model = 'MFT'  # 'MFT' or None

    # Optimization
    cfg.allow_break_sgd_after = 30
    cfg.break_sgd_after_iters_with_no_change = 20
    cfg.optimize_non_positional_params_after = 70
    cfg.levenberg_marquardt_max_ter = cfg.allow_break_sgd_after
    cfg.use_lr_scheduler = False
    cfg.lr_scheduler_patience = 5

    # Optical flow settings
    cfg.add_flow_arcs_strategy = 'single-previous'  # One of 'all-previous', 'single-previous' and 'absolute'
    # The 'all-previous' strategy for current frame i adds arcs (j, i) forall frames j < i, while 'single-previous' adds
    # only arc (i - 1, i).
    cfg.segmentation_mask_erosion_iters = 0
    # Pre-initialization method of 'levenberg-marquardt', 'gradient_descent', 'coordinate_descent' and 'lbfgs'
    cfg.preinitialization_method = 'levenberg-marquardt'
    cfg.flow_sgd = False
    cfg.flow_sgd_n_samples = 100

    return cfg

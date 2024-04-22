from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False
    cfg.all_frames_keyframes = False

    cfg.matching_target_to_backview = True

    cfg.run_main_optimization_loop = False
    cfg.preinitialization_method = 'essential_matrix_decomposition'
    cfg.gt_flow_source = 'FlowNetwork'
    cfg.essential_matrix_algorithm = 'pygcransac'

    cfg.ransac_feed_only_inlier_flow = True
    cfg.ransac_feed_only_inlier_flow_epe_threshold = 3.0

    return cfg

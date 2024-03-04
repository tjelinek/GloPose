from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False
    cfg.all_frames_keyframes = False

    cfg.matching_target_to_backview = False

    cfg.run_main_optimization_loop = True
    cfg.preinitialization_method = 'essential_matrix_decomposition'
    cfg.gt_flow_source = 'GenerateSynthetic'
    cfg.essential_matrix_algorithm = 'ransac'

    return cfg

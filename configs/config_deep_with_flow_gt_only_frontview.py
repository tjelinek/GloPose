from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False
    cfg.all_frames_keyframes = False

    cfg.matching_target_to_backview = False

    return cfg

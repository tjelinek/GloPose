from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False

    cfg.frame_filter = 'RoMa'

    cfg.min_roma_certainty_threshold = 0.95
    cfg.flow_reliability_threshold = 0.75
    cfg.min_number_of_reliable_matches = 1000

    return cfg

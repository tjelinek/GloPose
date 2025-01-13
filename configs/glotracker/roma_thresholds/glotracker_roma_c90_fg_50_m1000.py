from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.frame_filter = 'RoMa'

    cfg.min_roma_certainty_threshold = 0.9
    cfg.flow_reliability_threshold = 0.5
    cfg.min_number_of_reliable_matches = 1000

    return cfg

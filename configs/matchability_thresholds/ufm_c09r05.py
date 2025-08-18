from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.min_roma_certainty_threshold = 0.9
    cfg.flow_reliability_threshold = 0.5
    cfg.frame_filter = 'dense_matching'
    cfg.frame_filter_matcher = 'UFM'

    return cfg

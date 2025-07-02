from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.min_roma_certainty_threshold: float = 0.5
    cfg.flow_reliability_threshold: float = 0.25

    return cfg

from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False

    cfg.matcher = 'RoMa'

    cfg.min_roma_certainty_threshold = 0.8
    cfg.flow_reliability_threshold = 0.25
    cfg.min_number_of_reliable_matches = 250

    return cfg

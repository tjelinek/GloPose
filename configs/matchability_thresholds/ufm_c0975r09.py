from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.0.975
    cfg.flow_reliability_threshold = 0.9
    cfg.frame_filter = 'dense_matching'
    cfg.dense_matching = 'UFM'

    return cfg

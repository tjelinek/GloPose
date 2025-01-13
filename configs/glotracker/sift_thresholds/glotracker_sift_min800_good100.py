from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.frame_filter = 'SIFT'

    cfg.sift_filter_min_matches = 100
    cfg.sift_filter_good_to_add_matches = 450

    return cfg

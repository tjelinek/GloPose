from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.frame_filter = 'passthrough'
    cfg.roma_allow_disk_cache = False
    cfg.skip_indices = 2

    return cfg

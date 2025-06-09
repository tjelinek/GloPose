from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.evaluate_sam2_only = True
    # cfg.skip_indices = 1
    # cfg.per_dataset_skip_indices = False

    return cfg

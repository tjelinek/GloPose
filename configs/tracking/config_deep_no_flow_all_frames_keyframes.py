from configs.tracking.config_deep import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.loss_flow_weight = 0
    cfg.all_frames_keyframes = True

    return cfg

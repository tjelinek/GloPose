from configs.config_deep import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.loss_laplacian_weight = 0
    cfg.loss_tv_weight = 0
    cfg.loss_iou_weight = 0
    cfg.loss_dist_weight = 0
    cfg.loss_texture_change_weight = 0
    cfg.loss_rgb_weight = 0
    cfg.loss_q_weight = 0
    cfg.loss_t_weight = 0
    cfg.loss_flow_weight = 10.0

    cfg.optimize_shape = False
    cfg.max_keyframes = 2


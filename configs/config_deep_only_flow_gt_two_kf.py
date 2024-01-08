from dataclasses import dataclass

from configs.config_deep import ConfigDeep


@dataclass
class ConfigDeep(ConfigDeep):

    loss_laplacian_weight: float = 0
    loss_tv_weight: float = 0
    loss_iou_weight: float = 0
    loss_dist_weight: float = 0
    loss_texture_change_weight: float = 0
    loss_rgb_weight: float = 0
    loss_q_weight: float = 0
    loss_t_weight: float = 0
    loss_flow_weight: float = 10.0

    optimize_shape: bool = False
    max_keyframes: int = 2


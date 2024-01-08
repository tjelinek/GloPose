from dataclasses import dataclass

from configs.config_deep import ConfigDeep


@dataclass
class ConfigDeep(ConfigDeep):

    loss_flow_weight: float = 0
    all_frames_keyframes: bool = True

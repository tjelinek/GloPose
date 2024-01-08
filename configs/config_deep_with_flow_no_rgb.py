from dataclasses import dataclass, field
from typing import List

from tracker_config import TrackerConfig


@dataclass
class ConfigDeep(TrackerConfig):

    loss_rgb_weight = 0

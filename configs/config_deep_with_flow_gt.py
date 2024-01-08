from dataclasses import dataclass, field
from typing import List

from tracker_config import TrackerConfig


@dataclass
class ConfigDeep(TrackerConfig):

    optimize_shape = False
    all_frames_keyframes = True

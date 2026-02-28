from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


@dataclass
class BaseUFMConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__

        self.use_custom_weights: bool = False
        self.backward: bool = True

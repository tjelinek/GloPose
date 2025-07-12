from dataclasses import dataclass


@dataclass
class BaseFrameFilterConfig:

    pass

    def __post_init__(self):
        self.config_name = self.__class__.__name__

from dataclasses import dataclass


@dataclass
class BasePoseEstimationConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.config_name: str = self.__class__.__name__

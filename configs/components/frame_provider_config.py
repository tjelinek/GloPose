from dataclasses import dataclass


@dataclass
class BaseFrameProviderConfig:

    erode_segmentation: bool = False
    erode_segmentation_iters: int = 2
    background_color: str = 'original'  # 'original' | 'black' | 'white'

    def __post_init__(self):
        self.config_name = self.__class__.__name__

import sys
from pathlib import Path

import torch

from utils.general import pad_to_multiple, unpad_image

sys.path.append('repositories/Depth-Anything-V2/')
from repositories.DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2


class DepthAnythingProvider:

    def __init__(self):
        model_configs = {
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
        }

        encoder = 'vitl'  # or 'vitb', 'vits'
        self.depth_anything = DepthAnythingV2(**model_configs[encoder])
        base_dir = Path('/mnt/personal/jelint19/weights/DepthAnythingV2/')

        path_to_weights = base_dir / f'depth_anything_v2_{encoder}.pth'

        self.depth_anything.load_state_dict(torch.load(path_to_weights, map_location='cpu'))
        self.depth_anything.eval()

    def infer_depth_anything(self, image: torch.Tensor) -> torch.Tensor:

        padded_image, pad_h, pad_w = pad_to_multiple(image, 14)

        depth_image = self.depth_anything(padded_image.to('cpu'))
        depth_image = unpad_image(depth_image, pad_h, pad_w)

        depth_image = depth_image.cuda()

        return depth_image


def depth_to_point_cloud(depth_image, f_x: float, f_y: float, c_x: float, c_y: float):
    pass

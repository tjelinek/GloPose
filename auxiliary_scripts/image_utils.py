from collections import namedtuple
from pathlib import Path

import cv2
import imageio
from scipy.ndimage import uniform_filter
from torch.nn import functional as F
from torchvision import transforms

ImageShape = namedtuple('ImageShape', ['width', 'height'])


def get_shape(image_path: Path) -> ImageShape:
    image = imageio.v3.imread(image_path)

    return ImageShape(width=image.shape[1], height=image.shape[0])


def pad_image(image):
    W, H = image.shape[-2:]
    max_size = max(H, W)
    pad_h = max_size - H
    pad_w = max_size - W
    padding = [(pad_h + 1) // 2, pad_h // 2, pad_w // 2, (pad_w + 1) // 2]  # (top, bottom, left, right)

    image = F.pad(image, padding, mode='constant', value=0)

    return image


def resize_and_filter_image(image, new_width, new_height):
    image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    image = uniform_filter(image, size=(3, 3, 1))

    image_tensor = transforms.ToTensor()(image / 255.0)
    image = image_tensor.unsqueeze(0).float()

    return image

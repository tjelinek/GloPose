from collections import namedtuple
from pathlib import Path

import cv2
import imageio
import numpy as np
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


def overlay_occlusion(image: np.ndarray, occlusion_mask: np.ndarray, alpha: float = 0.2):
    """
    Overlay an occlusion mask on an image.

    Args:
    - image: The original image as a numpy array of shape (H, W, C).
    - occlusion_mask: The occlusion mask as a numpy array of shape (H, W, 1), values in [0, 1].
    - alpha: The alpha value for the overlay, where 0 means no overlay and 1 means full overlay.

    Returns:
    - The image with the occlusion mask overlay as a numpy array of shape (H, W, C).
    """
    occlusion_mask = occlusion_mask.squeeze()  # Remove the singleton dimension if present
    occlusion_mask = occlusion_mask * alpha  # Apply the alpha value to the occlusion mask

    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # Grayscale or single-channel
        image = np.dstack([image] * 3)  # Convert to 3-channel for coloring

    white_overlay = np.ones_like(image) * 255
    overlay_image = (1 - occlusion_mask[..., np.newaxis]) * image + occlusion_mask[..., np.newaxis] * white_overlay

    return overlay_image.astype(image.dtype)

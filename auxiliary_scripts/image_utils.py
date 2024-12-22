from collections import namedtuple
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from kornia.image import ImageSize
from scipy.ndimage import uniform_filter
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image, ExifTags

def get_shape(image_path: Path, image_downsample: float=1.0) -> ImageSize:
    image = imageio.v3.imread(image_path)

    return ImageSize(width=int(image_downsample * image.shape[1]),
                     height=int(image_downsample * image.shape[0]))


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


def get_intrinsics_from_exif(image_path: Path, err_on_default=False) -> torch.tensor:
    image = Image.open(image_path)
    max_size = max(image.size)
    width, height = image.size

    exif = image.getexif()
    focal = None
    sensor_width_mm = None
    focal_35mm = None
    focal_mm = None

    if exif is not None:
        for tag, value in exif.items():
            tag_name = ExifTags.TAGS.get(tag, None)
            if tag_name == 'FocalLength':
                focal_mm = float(value[0]) / float(value[1])  # Handle rational values
            elif tag_name == 'FocalLengthIn35mmFilm':
                focal_35mm = float(value)
            elif tag_name == 'SensorWidth':
                sensor_width_mm = float(value)

        # If FocalLengthIn35mmFilm is available, use it as a fallback
        if focal_35mm and sensor_width_mm:
            focal = focal_35mm / 35.0 * max_size
        elif focal_mm and sensor_width_mm:
            focal = focal_mm * (width / sensor_width_mm)

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length in EXIF data")

        # Fallback to a prior focal length approximation
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    # Compute intrinsics
    fx = focal
    fy = focal  # Assuming square pixels
    cx = width / 2
    cy = height / 2

    return torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

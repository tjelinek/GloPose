import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from kornia.morphology import erosion, dilation
from skimage.measure import label, regionprops
import torch.nn.functional as F

from tracker_config import TrackerConfig


def erode_segment_mask2(erosion_iterations, segment_masks):
    """

    :param erosion_iterations: int - iterations of erosion by 3x3 kernel
    :param segment_masks: Tensor of shape (N, 1, H, W)
    :return: Eroded segment mask of the same shape
    """

    kernel = torch.ones(3, 3).to(segment_masks.device)
    eroded_segment_masks = segment_masks.clone()

    for _ in range(erosion_iterations):
        eroded_segment_masks = erosion(eroded_segment_masks, kernel)
    return eroded_segment_masks


def dilate_mask(dilation_iterations, mask_tensor):
    """

    :param dilation_iterations: int - iterations of erosion by 3x3 kernel
    :param mask_tensor: Tensor of shape (1, N, 1, H, W)
    :return: Eroded segment mask of the same shape
    """

    kernel = torch.ones(3, 3).to(mask_tensor.device)
    dilated_segment_masks = mask_tensor.clone()[0]

    for _ in range(dilation_iterations):
        dilated_segment_masks = dilation(dilated_segment_masks, kernel)
    return dilated_segment_masks[None]


def load_config(config_path) -> TrackerConfig:
    config_path = Path(config_path)

    spec = importlib.util.spec_from_file_location("module.name", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config_module
    spec.loader.exec_module(config_module)

    config_instance: TrackerConfig = config_module.get_config()

    return config_instance


def imread(name):
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        return img / 255
    elif img.shape[2] == 3:
        return img[:, :, [2, 1, 0]] / 255
    else:
        return img[:, :, [2, 1, 0, 3]] / 65535


def fmo_detect_maxarea(I, B):
    dI = (np.sum(np.abs(I - B), 2) > 0.1).astype(float)
    labeled = label(dI)
    regions = regionprops(labeled)
    ind = -1
    maxarea = 0
    for ki in range(len(regions)):
        if regions[ki].area > maxarea:
            ind = ki
            maxarea = regions[ki].area
    if ind == -1:
        return [], 0
    bbox = np.array(regions[ind].bbox).astype(int)
    return bbox, regions[ind].minor_axis_length


def crop_resize(Is, bbox, res):
    if Is is None:
        return None
    rev_axis = False
    if len(Is.shape) == 3:
        rev_axis = True
        Is = Is[:, :, :, np.newaxis]
    imr = np.zeros((res[1], res[0], Is.shape[2], Is.shape[3]))
    for kk in range(Is.shape[3]):
        im = Is[bbox[0]:bbox[2], bbox[1]:bbox[3], :, kk]
        imr[:, :, :, kk] = cv2.resize(im, res, interpolation=cv2.INTER_CUBIC)
    if rev_axis:
        imr = imr[:, :, :, 0]
    return imr

def mesh_normalize(vertices):
    mesh_max = torch.max(vertices, dim=1, keepdim=True)[0]
    mesh_min = torch.min(vertices, dim=1, keepdim=True)[0]
    mesh_middle = (mesh_min + mesh_max) / 2
    vertices = vertices - mesh_middle
    bs = vertices.shape[0]
    mesh_biggest = torch.max(vertices.view(bs, -1), dim=1)[0]
    vertices = vertices / mesh_biggest.view(bs, 1, 1)  # * 0.45
    return vertices


def comp_tran_diff(vect):
    vdiff = (vect[1:] - vect[:-1]).abs()
    vdiff[vdiff < 0.2] = 0
    return torch.cat((0 * vdiff[:1], vdiff), 0).norm(dim=1)


def normalize_vertices(vertices: torch.Tensor):
    vertices = vertices - vertices.mean(axis=0)
    max_abs_val = torch.max(torch.abs(vertices))
    magnification = 1.0 / max_abs_val if max_abs_val != 0 else 1.0
    vertices *= magnification

    return vertices


def get_foreground_and_segment_mask(observed_occlusion, observed_segmentation,
                                    occlusion_threshold: float, segmentation_threshold: float):
    not_occluded_binary_mask: torch.Tensor = (observed_occlusion <= occlusion_threshold)
    segmentation_binary_mask: torch.Tensor = (observed_segmentation > segmentation_threshold)
    not_occluded_foreground_mask = (not_occluded_binary_mask * segmentation_binary_mask).squeeze()

    return not_occluded_binary_mask, segmentation_binary_mask, not_occluded_foreground_mask


def get_not_occluded_foreground_points(observed_occlusion, observed_segmentation, occlusion_threshold,
                                       segmentation_threshold):

    _, _, not_occluded_foreground_mask = get_foreground_and_segment_mask(observed_occlusion, observed_segmentation,
                                                                         occlusion_threshold, segmentation_threshold)

    src_pts_yx = torch.nonzero(not_occluded_foreground_mask).to(torch.float32)

    return src_pts_yx, not_occluded_foreground_mask


def print_cuda_occupied_memory(device='cuda:0'):
    # Set the device
    torch.cuda.set_device(device)

    # Get the current memory usage and the total memory.
    allocated_memory = torch.cuda.memory_allocated(device)
    reserved_memory = torch.cuda.memory_reserved(device)

    print(f"Allocated Memory: {allocated_memory / 1024 ** 2:.2f} MB")
    print(f"Reserved Memory: {reserved_memory / 1024 ** 2:.2f} MB")


def tensor_index_to_coordinates_xy(src_pts_yx):
    src_pts_xy = src_pts_yx.clone()
    src_pts_xy[:, [0, 1]] = src_pts_yx[:, [1, 0]]

    return src_pts_xy


def coordinates_xy_to_tensor_index(src_pts_xy):
    src_pts_yx = src_pts_xy.clone()
    src_pts_yx[:, [0, 1]] = src_pts_xy[:, [1, 0]]

    return src_pts_yx


def homogenize_3x4_transformation_matrix(T_3x4):
    T_4x4 = torch.eye(4, dtype=T_3x4.dtype).to(T_3x4.device).expand(*T_3x4.shape[:-2], 4, 4)
    T_4x4[..., :3, :] = T_3x4

    return T_4x4


def pad_to_multiple(image, multiple):

    height, width = image.shape[-2:]
    pad_h = multiple - (height % multiple)
    pad_w = multiple - (width % multiple)
    padded_image = F.pad(image, (0, pad_w, 0, pad_h))

    return padded_image, pad_h, pad_w


def unpad_image(image, pad_h, pad_w):
    if pad_h > 0:
        image = image[:, :-pad_h, :]
    if pad_w > 0:
        image = image[:, :, :-pad_w]
    return image

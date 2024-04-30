import importlib
import sys
from pathlib import Path

import cv2
import math
import numpy as np
import torch
import yaml
from kornia.morphology import erosion
from skimage.measure import label, regionprops

from main_settings import tmp_folder
from tracker_config import TrackerConfig


def segment2bbox(segment):
    inds = segment.nonzero(as_tuple=False)
    bbox = [int(inds[:, 1].min()), int(inds[:, 0].min()), int(inds[:, 1].max()), int(inds[:, 0].max())]
    return bbox


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


def write_video(array4d, path, fps=6):
    """

    :param array4d: Input Tensor of Shape (1, N, C, H, W)
    :param path: Output path
    :param fps: Frames per second
    :return:
    """
    array4d[array4d < 0] = 0
    array4d[array4d > 1] = 1
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (array4d.shape[-1], array4d.shape[-2]), True)
    for ki in range(array4d.shape[1]):
        out.write((array4d[:, ki, [2, 1, 0]] * 255).astype(np.uint8))
    out.release()


def calciou_masks(mask1, mask2):
    A_inter_B = mask1 * mask2
    A_union_B = (mask1 + mask2 - A_inter_B)
    iou = np.sum(A_inter_B) / np.sum(A_union_B)
    return iou


def load_config_yaml(config_name) -> TrackerConfig:
    with open(config_name) as file:
        config = yaml.safe_load(file)
    tracker_config = TrackerConfig(**config)
    return tracker_config


def load_config(config_path) -> TrackerConfig:
    config_path = Path(config_path)
    if config_path.suffix == 'yaml':
        load_config_yaml(config_path)

    spec = importlib.util.spec_from_file_location("module.name", config_path)
    config_module = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = config_module
    spec.loader.exec_module(config_module)

    config_instance: TrackerConfig = config_module.get_config()

    return config_instance


def fmo_detect(I, B):
    # simulate FMO detector -> find approximate location of FMO
    dI = (np.sum(np.abs(I - B), 2) > 0.05).astype(float)
    labeled = label(dI)
    regions = regionprops(labeled)
    ind = -1
    maxsol = 0
    for ki in range(len(regions)):
        if 100 < regions[ki].area < 0.01 * np.prod(dI.shape):
            if regions[ki].solidity > maxsol:
                ind = ki
                maxsol = regions[ki].solidity
    if ind == -1:
        return [], 0

    # pdb.set_trace()
    bbox = np.array(regions[ind].bbox).astype(int)
    return bbox, regions[ind].minor_axis_length


def imread(name):
    img = cv2.imread(name, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 2:
        return img / 255
    elif img.shape[2] == 3:
        return img[:, :, [2, 1, 0]] / 255
    else:
        return img[:, :, [2, 1, 0, 3]] / 65535


def imwrite(im, name=tmp_folder + 'tmp.png'):
    im[im < 0] = 0
    im[im > 1] = 1
    cv2.imwrite(name, im[:, :, [2, 1, 0]] * 255)


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


def deg_to_rad(deg):
    return math.pi * deg / 180.0


def rad_to_deg(rad):
    return 180 * rad / math.pi


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


def qnorm(q1):
    return q1 / q1.norm()


def qnorm_vectorized(quaternions):
    return quaternions / quaternions.norm(dim=-1).unsqueeze(-1)


def qmult(q1, q0):  # q0, then q1, you get q3
    w0, x0, y0, z0 = q0[0]
    w1, x1, y1, z1 = q1[0]
    q3 = torch.cat(((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)[None, None],
                    (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0)[None, None],
                    (-x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0)[None, None],
                    (x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)[None, None]), 1)
    return q3


def qdist(q1, q2):
    return 1 - (q1 * q2).sum() ** 2


def qdifference(q1, q2):  # how to get from q1 to q2
    q1conj = -q1
    q1conj[0, 0] = q1[0, 0]
    q1inv = q1conj / q1.norm()
    diff = qmult(q2, q1inv)
    return diff


def quaternion_angular_difference(quaternions1, quaternions2):
    angles = torch.zeros(quaternions1.shape[1])

    for i in range(angles.shape[0]):
        diff = qnorm(qdifference(quaternions1[:, i], quaternions2[:, i]))
        ang = float(2 * torch.atan2(diff[:, 1:].norm(), diff[:, 0])) * 180 / np.pi
        angles[i] = ang
    return angles


def consecutive_quaternions_angular_difference(quaternion):
    angs = []
    for qi in range(quaternion.shape[1] - 1):
        diff = qnorm(qdifference(quaternion[:, qi], quaternion[:, qi + 1]))
        angs.append(float(2 * torch.atan2(diff[:, 1:].norm(), diff[:, 0])) * 180 / np.pi)
    return np.array(angs)


def consecutive_quaternions_angular_difference2(quaternion):
    angs = []
    for qi in range(quaternion.shape[1] - 1):
        ang = float(torch.acos(torch.dot(quaternion[0, qi], quaternion[0, qi + 1]) /
                               (quaternion[0, qi].norm() * quaternion[0, qi].norm()))) * 180.0 / np.pi
        angs.append(ang)
    return np.array(angs)


def normalize_rendered_flows(rendered_flows, rendering_width, rendering_height, original_width,
                             original_height):
    rendered_flows[..., 0] = rendered_flows[..., 0] * (rendering_width / original_width)
    rendered_flows[..., 1] = rendered_flows[..., 1] * (rendering_height / original_height)

    return rendered_flows


def normalize_vertices(vertices: torch.Tensor):
    vertices = vertices - vertices.mean(axis=0)
    max_abs_val = torch.max(torch.abs(vertices))
    magnification = 1.0 / max_abs_val if max_abs_val != 0 else 1.0
    vertices *= magnification

    return vertices


def get_foreground_and_segment_mask(observed_occlusion: torch.Tensor, observed_segmentation: torch.Tensor,
                                    occlusion_threshold: float, segmentation_threshold: float):
    not_occluded_binary_mask = (observed_occlusion <= occlusion_threshold)
    segmentation_binary_mask = (observed_segmentation > segmentation_threshold)
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

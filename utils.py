import cv2
import math
import numpy as np
import torch
import yaml
from scipy.ndimage import center_of_mass
from skimage.measure import label, regionprops
from torch import Tensor
from typing import Tuple

from main_settings import tmp_folder


def segment2bbox(segment):
    inds = segment.nonzero(as_tuple=False)
    bbox = [int(inds[:, 1].min()), int(inds[:, 0].min()), int(inds[:, 1].max()), int(inds[:, 0].max())]
    return bbox


def erode_segment_mask(erosion_iterations, segment_masks):
    """

    :param erosion_iterations: int - iterations of erosion by 3x3 kernel
    :param segment_masks: Tensor of shape (N, 1, H, W)
    :return: Eroded segment mask of the same shape
    """

    eroded_segment_masks = segment_masks.clone()
    conv_kernel = torch.Tensor([[0, 1, 0],
                                [1, 1, 1],
                                [0, 1, 0]]).to(torch.float64)
    conv_kernel = conv_kernel[None][None].to(segment_masks.device)

    for _ in range(erosion_iterations):
        eroded_segment_masks = torch.nn.functional.conv2d(eroded_segment_masks, weight=conv_kernel, padding=(1, 1))
        # The sum of the conv weights is 5
        eroded_segment_masks = (eroded_segment_masks >= 5).to(torch.float64) * 1.0
    return eroded_segment_masks


def write_video(array4d, path, fps=6):
    array4d[array4d < 0] = 0
    array4d[array4d > 1] = 1
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (array4d.shape[1], array4d.shape[0]), True)
    for ki in range(array4d.shape[3]):
        out.write((array4d[:, :, [2, 1, 0], ki] * 255).astype(np.uint8))
    out.release()


def calciou_masks(mask1, mask2):
    A_inter_B = mask1 * mask2
    A_union_B = (mask1 + mask2 - A_inter_B)
    iou = np.sum(A_inter_B) / np.sum(A_union_B)
    return iou


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


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


def montageF(F):
    return np.reshape(np.transpose(F, (0, 1, 3, 2)), (F.shape[0], -1, F.shape[2]), 'F')


def montageH(Hs):
    return np.concatenate((np.sum(Hs[:, :, ::3], 2)[:, :, np.newaxis], np.sum(Hs[:, :, 1::3], 2)[:, :, np.newaxis],
                           np.sum(Hs[:, :, 2::3], 2)[:, :, np.newaxis]), 2)


def diskMask(rad):
    sz = 2 * np.array([rad, rad])

    ran1 = np.arange(-(sz[1] - 1) / 2, ((sz[1] - 1) / 2) + 1, 1.0)
    ran2 = np.arange(-(sz[0] - 1) / 2, ((sz[0] - 1) / 2) + 1, 1.0)
    xv, yv = np.meshgrid(ran1, ran2)
    mask = np.square(xv) + np.square(yv) <= rad * rad
    M = mask.astype(float)
    return M


def boundingBox(img, pads=None):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    if pads is not None:
        rmin = max(rmin - pads[0], 0)
        rmax = min(rmax + pads[0], img.shape[0])
        cmin = max(cmin - pads[1], 0)
        cmax = min(cmax + pads[1], img.shape[1])
    return rmin, rmax, cmin, cmax


def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, i)
    size = round(size_bytes / power, 2)
    return "{} {}".format(size, size_name[i])


def calc_tiou(gt_traj, traj, rad):
    ns = gt_traj.shape[1]
    est_traj = np.zeros(gt_traj.shape)
    if traj.shape[0] == 4:
        for ni, ti in zip(range(ns), np.linspace(0, 1, ns)):
            est_traj[:, ni] = traj[[1, 0]] * (1 - ti) + ti * traj[[3, 2]]
    else:
        bline = (np.abs(traj[3] + traj[7]) > 1.0).astype(float)
        if bline:
            len1 = np.linalg.norm(traj[[5, 1]])
            len2 = np.linalg.norm(traj[[7, 3]])
            v1 = traj[[5, 1]] / len1
            v2 = traj[[7, 3]] / len2
            piece = (len1 + len2) / (ns - 1)
            for ni in range(ns):
                est_traj[:, ni] = traj[[4, 0]] + np.min([piece * ni, len1]) * v1 + np.max([0, piece * ni - len1]) * v2
        else:
            for ni, ti in zip(range(ns), np.linspace(0, 1, ns)):
                est_traj[:, ni] = traj[[4, 0]] + ti * traj[[5, 1]] + ti * ti * traj[[6, 2]]

    est_traj2 = est_traj[:, -1::-1]

    ious = calciou(gt_traj, est_traj, rad)
    ious2 = calciou(gt_traj, est_traj2, rad)
    return np.max([np.mean(ious), np.mean(ious2)])


def calciou(p1, p2, rad):
    dists = np.sqrt(np.sum(np.square(p1 - p2), 0))
    dists[dists > 2 * rad] = 2 * rad

    theta = 2 * np.arccos(dists / (2 * rad))
    A = ((rad * rad) / 2) * (theta - np.sin(theta))
    intersection = 2 * A
    union = 2 * np.pi * rad * rad - intersection
    iou = intersection / union
    return iou


def generate_lowFPSvideo(V, k=8, gamma_coef=0.4, do_WB=True):
    newk = int(np.floor(V.shape[3] / k))
    Vk = np.reshape(V[:, :, :, :newk * k], (V.shape[0], V.shape[1], V.shape[2], newk, k)).mean(-1)
    if do_WB:
        WB = np.expand_dims(np.array([2, 1, 2]), [0, 1, 3])
        Vk_WB = ((Vk * WB) / WB.max()) ** gamma_coef
        WB = np.expand_dims(np.array([2, 1, 2]), [0, 1, 3])
    else:
        Vk_WB = Vk ** gamma_coef
    return Vk_WB


def extend_bbox_nonuniform(bbox, ext, shp):
    bbox[0] -= ext[0]
    bbox[2] += ext[0]
    bbox[1] -= ext[1]
    bbox[3] += ext[1]
    bbox[bbox < 0] = 0
    bbox[2] = np.min([bbox[2], shp[0] - 1])
    bbox[3] = np.min([bbox[3], shp[1] - 1])
    return bbox


def rgba2hs(rgba, bgr):
    return rgba[:, :, :3] * rgba[:, :, 3:] + bgr[:, :, :, None] * (1 - rgba[:, :, 3:])


def rgba2rgb(rgba):
    return rgba[:, :, :3] * rgba[:, :, 3:] + 1 * (1 - rgba[:, :, 3:])


def sync_directions_smooth(est_hs_crop, est_traj, est_traj_prev, radius):
    if est_traj_prev is not None:
        dist = np.min([np.linalg.norm(est_traj[:, 0] - est_traj_prev[:, 0]),
                       np.linalg.norm(est_traj[:, 0] - est_traj_prev[:, -1])])
        dist2 = np.min([np.linalg.norm(est_traj[:, -1] - est_traj_prev[:, 0]),
                        np.linalg.norm(est_traj[:, -1] - est_traj_prev[:, -1])])
        flip_due_to_newobj = np.min([dist, dist2]) > 2 * radius and est_traj[1, -1] > est_traj[1, 0]
        flip_due_to_smoothness = dist2 < dist
        do_flip = flip_due_to_newobj or flip_due_to_smoothness
    else:
        do_flip = est_traj[1, -1] > est_traj[1, 0]
    if do_flip:
        est_hs_crop = est_hs_crop[:, :, :, ::-1]
        est_traj = est_traj[:, ::-1]
    return est_hs_crop, est_traj, do_flip


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


def quaternion_from_euler(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Convert Euler angles to quaternion coefficients.

    Euler angles are assumed to be in radians in XYZ convention.

    Args:
        roll: the roll euler angle.
        pitch: the pitch euler angle.
        yaw: the yaw euler angle.

    Return:
        A tuple with quaternion coefficients in order of `wxyz`.
    """

    roll_half = roll * 0.5
    pitch_half = pitch * 0.5
    yaw_half = yaw * 0.5

    cy = yaw_half.cos()
    sy = yaw_half.sin()
    cp = pitch_half.cos()
    sp = pitch_half.sin()
    cr = roll_half.cos()
    sr = roll_half.sin()

    qw = cy * cp * cr + sy * sp * sr
    qx = cy * cp * sr - sy * sp * cr
    qy = sy * cp * sr + cy * sp * cr
    qz = sy * cp * cr - cy * sp * sr

    return qw, qx, qy, qz


def euler_from_quaternion(w: Tensor, x: Tensor, y: Tensor, z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Convert a quaternion coefficients to Euler angles.

    Returned angles are in radians in XYZ convention.

    Args:
        w: quaternion :math:`q_w` coefficient.
        x: quaternion :math:`q_x` coefficient.
        y: quaternion :math:`q_y` coefficient.
        z: quaternion :math:`q_z` coefficient.

    Return:
        A tuple with euler angles`roll`, `pitch`, `yaw`.
    """

    yy = y * y

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + yy)
    roll = sinr_cosp.atan2(cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = sinp.clamp(min=-1.0, max=1.0)
    pitch = sinp.asin()

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (yy + z * z)
    yaw = siny_cosp.atan2(cosy_cosp)

    return roll, pitch, yaw


def deg_to_rad(deg):
    return math.pi * deg / 180.0


def rad_to_deg(rad):
    return 180 * rad / math.pi


def compute_trandist(renders):
    masks = renders[0, :, :, -1].cpu().numpy()
    centers = []
    for mi in range(masks.shape[0]):
        centers.append(center_of_mass(masks[mi])[1:])
    Tdist = (np.diff(np.stack(centers).T, 1).T ** 2).sum(1) ** 0.5
    return Tdist


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


def comp_diff(vect):
    vdiff = vect[1:] - vect[:-1]
    v2diff = vdiff - torch.cat((vdiff[:1], vdiff[:-1]), 0)
    return torch.cat((0 * v2diff[:1], v2diff), 0).norm(dim=1)


def comp_2diff(vdiff):
    v2diff = vdiff - torch.cat((vdiff[:1], vdiff[:-1]), 0)
    return torch.cat((0 * v2diff[:1], v2diff), 0).abs()


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


def normalize_vertices(vertices):
    vertices = vertices - vertices[0].mean(0)
    magnification = 1 / (vertices.max() - vertices.mean()) * 1.0
    vertices *= magnification

    return vertices

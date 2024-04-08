import cv2
import numpy as np
import pygcransac
import torch
from kornia.geometry import (rotation_matrix_to_axis_angle, motion_from_essential_choose_solution,
                             euler_from_quaternion, axis_angle_to_quaternion)

from utils import tensor_index_to_coordinates_xy


def estimate_pose_using_dense_correspondences(src_pts_yx: torch.Tensor, dst_pts_yx: torch.Tensor, K1: torch.Tensor,
                                              K2: torch.Tensor, width: int, height: int,
                                              ransac_conf=0.99, method='magsac++'):
    # Convert to x, y order
    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

    src_pts_np = src_pts_xy.numpy(force=True)
    dst_pts_np = dst_pts_xy.numpy(force=True)

    if method == 'pygcransac':
        correspondences = np.ascontiguousarray(np.concatenate([src_pts_np, dst_pts_np], axis=1))
        # , sampler_id=1
        K1_np = K1.numpy(force=True)
        K2_np = K2.numpy(force=True)
        E, mask = pygcransac.findEssentialMatrix(correspondences, K1_np, K2_np, height, width, height, width,
                                                 ransac_conf, threshold=1.)
    else:
        methods = {'magsac++': cv2.USAC_MAGSAC,
                   'ransac': cv2.RANSAC,
                   '8point': cv2.USAC_FM_8PTS}

        chosen_method = methods[method]
        K1_np = K1.numpy(force=True)
        E, mask = cv2.findEssentialMat(src_pts_np, dst_pts_np, K1_np, method=chosen_method, threshold=1.,
                                       prob=ransac_conf)
        mask = mask[:, 0].astype(np.bool_)

    E_tensor = torch.from_numpy(E).cuda().to(torch.float32)
    mask_tensor = torch.from_numpy(mask).cuda()
    R, t_cam, triangulated_points = motion_from_essential_choose_solution(E_tensor, K1, K2, src_pts_yx, dst_pts_yx,
                                                                          mask_tensor)

    t_cam = t_cam.squeeze()

    r_cam = rotation_matrix_to_axis_angle(R.contiguous()).squeeze()

    r_cam_deg = torch.rad2deg(torch.stack(euler_from_quaternion(*axis_angle_to_quaternion(r_cam))))

    print('----------------------------------------')
    print("---t_cam", t_cam.squeeze().round(decimals=3))
    print("---r_cam", r_cam_deg.squeeze().round(decimals=3))
    print('----------------------------------------')

    return r_cam, t_cam, mask_tensor, triangulated_points

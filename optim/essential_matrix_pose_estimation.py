import cv2
import numpy as np
import pygcransac
import torch
from kornia.geometry import (rotation_matrix_to_axis_angle, motion_from_essential_choose_solution,
                             euler_from_quaternion, axis_angle_to_quaternion, axis_angle_to_rotation_matrix,
                             Rt_to_matrix4x4, relative_transformation, triangulate_points)

from utils import tensor_index_to_coordinates_xy


def estimate_pose_using_dense_correspondences(src_pts_yx: torch.Tensor, dst_pts_yx: torch.Tensor, K1: torch.Tensor,
                                              K2: torch.Tensor, width: int, height: int,
                                              ransac_conf=0.9999, method='magsac++'):
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
                                                 ransac_conf, threshold=0.01, min_iters=10000)

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

    # r_cam_deg = torch.rad2deg(torch.stack(euler_from_quaternion(*axis_angle_to_quaternion(r_cam))))
    # print('----------------------------------------')
    # print("---t_cam", t_cam.squeeze().round(decimals=3))
    # print("---r_cam", r_cam_deg.squeeze().round(decimals=3))
    # print('----------------------------------------')

    return r_cam, t_cam, mask_tensor, triangulated_points


def relative_scale_recovery(essential_matrix_data, flow_arc, K1):
    flow_source, flow_target = flow_arc

    flow_arc_prev = (flow_source, flow_target - 1)

    inlier_mask1 = essential_matrix_data.inlier_mask[flow_arc_prev]
    inlier_mask2 = essential_matrix_data.inlier_mask[flow_arc]
    common_inlier_mask = (inlier_mask1 & inlier_mask2)
    common_inlier_indices = torch.nonzero(common_inlier_mask, as_tuple=True)

    inliers_src_pts1 = essential_matrix_data.source_points[flow_arc][common_inlier_indices]
    inliers_dst_pts1 = essential_matrix_data.target_points[flow_arc][common_inlier_indices]
    inliers_src_pts2 = inliers_dst_pts1
    inliers_dst_pts2 = essential_matrix_data.target_points[flow_arc_prev][common_inlier_indices]

    R_1i = axis_angle_to_rotation_matrix(essential_matrix_data.camera_rotations[flow_arc_prev].unsqueeze(0))
    t_1i = essential_matrix_data.camera_translations[flow_arc_prev].unsqueeze(0).unsqueeze(-1)

    R_1j = axis_angle_to_rotation_matrix(essential_matrix_data.camera_rotations[flow_arc].unsqueeze(0))
    t_1j = essential_matrix_data.camera_translations[flow_arc].unsqueeze(0).unsqueeze(-1)

    T_1i = Rt_to_matrix4x4(R_1i, t_1i)
    T_1j = Rt_to_matrix4x4(R_1j, t_1j)
    T_ij = relative_transformation(T_1i, T_1j)

    # T_13x4 = R
    T_1i3x4 = T_1i[..., :3, :4][0]
    T_1j3x4 = T_1i[..., :3, :4][0]
    T_ij3x4 = T_ij[..., :3, :4][0]

    P_1i_3x4 = torch.mm(K1, T_1i3x4)
    P_1j_3x4 = torch.mm(K1, T_1i3x4)
    P_ij3x4 = torch.mm(K1, T_ij3x4)

    triangulated_points_1 = triangulate_points(P_1i_3x4, inliers_src_pts1, inliers_dst_pts1)
    triangulated_points_2 = triangulate_points(P_ij3x4, inliers_src_pts2, inliers_dst_pts2)

    N_point = triangulated_points_1.shape[0]

    pairs_N = min(100, N_point)

    random_pairs_indices = torch.randperm(N_point)[:pairs_N * 2].view(-1, 2)

    pts1_1i = triangulated_points_1[random_pairs_indices[:, 0]]
    pts2_2i = triangulated_points_1[random_pairs_indices[:, 1]]

    pts1_ij = triangulated_points_2[random_pairs_indices[:, 0]]
    pts2_ij = triangulated_points_2[random_pairs_indices[:, 1]]

    ratio = torch.linalg.norm(triangulated_points_1 - t)
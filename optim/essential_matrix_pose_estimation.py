import cv2
import numpy as np
import pygcransac
import torch
from kornia.geometry import rotation_matrix_to_axis_angle, motion_from_essential_choose_solution, Rt_to_matrix4x4, \
    matrix4x4_to_Rt, inverse_transformation, compose_transformations

from flow import source_coords_to_target_coords


def estimate_pose_using_dense_correspondences(dense_correspondences: torch.Tensor,
                                              dense_correspondences_mask: torch.Tensor, W_world_to_cam,
                                              K1, K2, width: int, height: int, ransac_conf=0.99, method='magsac++'):
    # src and target in y, x order
    src_pts_yx = torch.nonzero(dense_correspondences_mask).to(torch.float32)
    dst_pts_yx = source_coords_to_target_coords(src_pts_yx.permute(1, 0), dense_correspondences).permute(1, 0)

    # Convert to x, y order
    src_pts_xy = src_pts_yx.clone()
    dst_pts_xy = dst_pts_yx.clone()
    src_pts_xy[:, [0, 1]] = src_pts_yx[:, [1, 0]]
    dst_pts_xy[:, [0, 1]] = dst_pts_yx[:, [1, 0]]
    # dst_pts_xy[:, 0] = height - dst_pts_xy[:, 0]
    # TODO When converting to numpy, I may need to change the source coord to its mirror in the y axis

    src_pts_np = src_pts_xy.numpy(force=True)
    dst_pts_np = dst_pts_xy.numpy(force=True)

    if method == 'pygcransac':
        correspondences = np.ascontiguousarray(np.concatenate([src_pts_np, dst_pts_np], axis=1))
        E, mask = pygcransac.findEssentialMatrix(correspondences, K1, K2, height, width, height, width,
                                                 ransac_conf)
    else:
        methods = {'magsac++': cv2.USAC_MAGSAC,
                   'ransac': cv2.RANSAC,
                   '8point': cv2.USAC_FM_8PTS}

        chosen_method = methods[method]
        E, mask = cv2.findEssentialMat(src_pts_np, dst_pts_np, K1, method=chosen_method, threshold=1.,
                                       prob=ransac_conf)
        mask = mask[:, 0]

    inlier_src_pts = src_pts_yx[torch.nonzero(torch.from_numpy(mask), as_tuple=True)]
    outlier_src_pts = src_pts_yx[torch.nonzero(~torch.from_numpy(mask), as_tuple=True)]

    E_tensor = torch.from_numpy(E).cuda().to(torch.float32)
    K1_tensor = torch.from_numpy(K1).cuda()
    K2_tensor = torch.from_numpy(K2).cuda()
    mask_tensor = torch.from_numpy(mask).cuda()
    R, t, triangulated_points = motion_from_essential_choose_solution(E_tensor, K1_tensor, K2_tensor,
                                                                      src_pts_yx, dst_pts_yx, mask_tensor)

    T = Rt_to_matrix4x4(R.unsqueeze(0), t.unsqueeze(0))

    # Inverse of world -> camera matrix
    W_cam_to_world = inverse_transformation(W_world_to_cam)

    # Get the transformation matrices in world space
    T_world = compose_transformations(W_cam_to_world, compose_transformations(T, W_world_to_cam))
    R_world, t_world = matrix4x4_to_Rt(T_world)

    t_world = t_world.squeeze(-1)
    r_world = rotation_matrix_to_axis_angle(R_world.contiguous()).squeeze()

    r_world_deg = torch.rad2deg(r_world) % 180


    return r_world, t_world, inlier_src_pts, outlier_src_pts

import cv2
import numpy as np
import pygcransac
import torch
from kornia.geometry import rotation_matrix_to_axis_angle

from flow import source_coords_to_target_coords


def estimate_pose_using_dense_correspondences(dense_correspondences: torch.Tensor,
                                              dense_correspondences_mask: torch.Tensor, W_world_to_cam,
                                              K1, K2, width: int, height: int, ransac_conf=0.99, method='magsac++'):
    # src and target in y, x order
    src_pts = torch.nonzero(dense_correspondences_mask).to(torch.float32)
    dst_pts = source_coords_to_target_coords(src_pts.permute(1, 0), dense_correspondences).permute(1, 0)

    # Convert to x, y order
    src_pts[:, [0, 1]] = src_pts[:, [1, 0]]
    dst_pts[:, [0, 1]] = dst_pts[:, [1, 0]]

    src_pts_np = src_pts.numpy(force=True)
    dst_pts_np = dst_pts.numpy(force=True)

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

    inlier_src_pts = src_pts[torch.nonzero(torch.from_numpy(mask), as_tuple=True)]
    outlier_src_pts = src_pts[torch.nonzero(~torch.from_numpy(mask), as_tuple=True)]

    E_tensor = torch.from_numpy(E).cuda().to(torch.float32)
    K1_tensor = torch.from_numpy(K1).cuda()
    K2_tensor = torch.from_numpy(K2).cuda()
    mask_tensor = torch.from_numpy(mask).cuda()
    R, t, triangulated_points = motion_from_essential_choose_solution(E_tensor, K1_tensor, K2_tensor,
                                                                      src_pts, dst_pts, mask_tensor)

    T = Rt_to_matrix4x4(R.unsqueeze(0), t.unsqueeze(0))

    # Inverse of world -> camera matrix
    W_cam_to_world = inverse_transformation(W_world_to_cam)

    # Get the transformation matrices in world space
    T1_world = W_inv @ T1 @ W_hom
    T2_world = W_inv @ T2 @ W_hom

    t1_world = T1_world[:, 3:, 0:3]
    t2_world = T2_world[:, 3:, 0:3]
    R1_world = T1_world[:, 0:3, 0:3]
    R2_world = T2_world[:, 0:3, 0:3]

    r1_world = rotation_matrix_to_axis_angle(R1_world.contiguous())
    r2_world = rotation_matrix_to_axis_angle(R2_world.contiguous())

    r1_world_deg = torch.rad2deg(r1_world) % 180
    r2_world_deg = torch.rad2deg(r2_world) % 180

    print("Translation", t_world.squeeze().round(decimals=3))
    print("r1", r1_world_deg.squeeze().round(decimals=3))
    print("r2", r2_world_deg.squeeze().round(decimals=3))
    print("r1_cam", r1_deg.round(decimals=3))
    print("r2_cam", r2_deg.round(decimals=3))

    r1_world[:, [0, 1]] = r1_world[:, [1, 0]]
    r2_world[:, [0, 1]] = r2_world[:, [1, 0]]
    t1_world[:, :, [0, 1]] = t1_world[:, :, [1, 0]]
    t2_world[:, :, [0, 1]] = t2_world[:, :, [1, 0]]

    return r1_world, r2_world, t1_world, t2_world, inlier_src_pts, outlier_src_pts

    t_world = t_world.squeeze(-1)
    r_world = rotation_matrix_to_axis_angle(R_world.contiguous()).squeeze()

    r_world_deg = torch.rad2deg(r_world) % 180


    return r_world, t_world, inlier_src_pts, outlier_src_pts

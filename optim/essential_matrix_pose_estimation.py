import cv2
import numpy as np
import pygcransac
import torch
from kornia.geometry import rotation_matrix_to_angle_axis


def estimate_pose_using_dense_correspondences(dense_correspondences, dense_correspondences_mask,
                                              camera_transformation_matrix, K1, K2, width, height):

    src_pts = torch.nonzero(dense_correspondences_mask)
    dst_pts = dense_correspondences[:, src_pts[:, 0], src_pts[:, 1]].permute(1, 0) + src_pts

    src_pts_np = src_pts.numpy(force=True).astype(np.float64)
    dst_pts_np = dst_pts.numpy(force=True).astype(np.float64)
    correspondences = np.ascontiguousarray(np.concatenate([src_pts_np, dst_pts_np], axis=1))

    E, mask = pygcransac.findEssentialMatrix(correspondences, K1, K2, height, width, height, width, 1e-4)

    inlier_src_pts = src_pts[torch.nonzero(torch.from_numpy(mask), as_tuple=True)]
    outlier_src_pts = src_pts[torch.nonzero(~torch.from_numpy(mask), as_tuple=True)]

    R1, R2, t = cv2.decomposeEssentialMat(E)
    r1 = rotation_matrix_to_angle_axis(torch.from_numpy(R1))
    r2 = rotation_matrix_to_angle_axis(torch.from_numpy(R2))

    r1_deg = torch.rad2deg(r1) % 180
    r2_deg = torch.rad2deg(r2) % 180

    T1 = transformation_matrix(torch.from_numpy(R1).cuda(), torch.from_numpy(t).cuda())
    T2 = transformation_matrix(torch.from_numpy(R2).cuda(), torch.from_numpy(t).cuda())

    # W: World coords -> camera coords
    W = camera_transformation_matrix

    W_hom = homogenize_transformation_matrix(W)

    # Inverse of world -> camera matrix
    W_inv = invert_transformation_matrix(W)

    # Get the transformation matrices in world space
    T1_world = W_inv @ T1 @ W_hom
    T2_world = W_inv @ T2 @ W_hom

    t_world = T1_world[:, 3:, 0:3]
    R1_world = T1_world[:, 0:3, 0:3]
    R2_world = T2_world[:, 0:3, 0:3]

    r1_world = rotation_matrix_to_angle_axis(R1_world.contiguous())
    r2_world = rotation_matrix_to_angle_axis(R2_world.contiguous())

    r1_world_deg = torch.rad2deg(r1_world) % 180
    r2_world_deg = torch.rad2deg(r2_world) % 180

    print("Translation", t_world.squeeze().round(decimals=3))
    print("r1", r1_world_deg.squeeze().round(decimals=3))
    print("r2", r2_world_deg.squeeze().round(decimals=3))
    print("r1_cam", r1_deg.round(decimals=3))
    print("r2_cam", r2_deg.round(decimals=3))

    return r1_world, r2_world, t_world, inlier_src_pts, outlier_src_pts


def homogenize_transformation_matrix(T):
    T_hom = torch.zeros(1, 4, 4).cuda()
    T_hom[:, 3, 3] = 1.
    T_hom[:, :, :3] = T
    return T_hom


def invert_transformation_matrix(T):

    T_rot = T[:, :3, :3]
    T_trans = T[:, [3], :3]

    T_inv = torch.zeros(1, 4, 4).cuda()
    T_inv[0, 3, 3] = 1.0
    T_inv[:, :3, :3] = T_rot.transpose(1, 2)
    T_inv[:, 3:, :3] = (-T_rot.transpose(1, 2) @ T_trans.transpose(1, 2)).transpose(1, 2)
    return T_inv


def transformation_matrix(R, t):
    T = torch.zeros(1, 4, 4).cuda()
    T[:, 3, 3] = 1.
    T[0, 0:3, 0:3] = R
    T[0, 3:, 0:3] = t.T
    return T

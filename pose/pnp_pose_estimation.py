import cv2
import numpy as np
import torch

from tracker_config import TrackerConfig
from utils import tensor_index_to_coordinates_xy


def estimate_pose_using_PnP_solver(src_pts_yx: torch.Tensor, dst_pts_yx: torch.Tensor, K1: torch.Tensor,
                                   K2: torch.Tensor, coords_c1_3d: torch.Tensor, width: int, height: int,
                                   ransac_config: TrackerConfig, confidences=None):
    # Convert to x, y order
    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

    src_pts_np = src_pts_xy.numpy(force=True)
    dst_pts_np = dst_pts_xy.numpy(force=True)

    coords_c1_3d_fg = coords_c1_3d[0, 0, :, src_pts_yx[:, 1].long(), src_pts_yx[:, 0].long()]
    coords_c1_3d_fg = coords_c1_3d_fg[[1, 0, 2]]
    coords_c1_3d_numpy_fg = coords_c1_3d_fg.permute(1, 0).numpy(force=True)

    K1_np = K1.numpy(force=True)
    K2_np = K2.numpy(force=True)

    N_matches = src_pts_xy.shape[0]
    min_matches_for_ransac = 6

    if N_matches > min_matches_for_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(coords_c1_3d_numpy_fg, dst_pts_np, K2_np, None)

        rvec = torch.from_numpy(rvec).cuda().squeeze().to(torch.float32)
        tvec = torch.from_numpy(tvec).cuda().to(torch.float32)
        inliers = torch.from_numpy(inliers).cuda().to(torch.bool)[:, 0]
    else:
        tvec = torch.zeros(3).to(torch.float32).cuda().unsqueeze(-1)
        rvec = torch.zeros(3).to(torch.float32).cuda()

    inliers = torch.ones(src_pts_yx.shape[0], dtype=torch.bool)  # TODO remove me
    triangulated_points = torch.zeros(inliers.shape[0], 3).to(torch.float32)  # TODO remove me

    rvec[1] -= np.pi
    rvec *= -1.  # TODO properly align the poses

    return rvec, tvec, inliers, triangulated_points

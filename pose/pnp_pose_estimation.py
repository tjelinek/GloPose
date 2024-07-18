import cv2
import torch


def estimate_pose_using_PnP_solver(src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor, K1: torch.Tensor,
                                   K2: torch.Tensor, coords_c1_3d_xy: torch.Tensor):

    src_pts_np = src_pts_xy.numpy(force=True)
    dst_pts_np = dst_pts_xy.numpy(force=True)

    coords_c1_3d_xy_fg = coords_c1_3d_xy[:, src_pts_xy[:, 0].long(), src_pts_xy[:, 1].long()]
    coords_c1_3d_xy_numpy_fg = coords_c1_3d_xy_fg.permute(1, 0).numpy(force=True)

    K1_np = K1.numpy(force=True)
    K2_np = K2.numpy(force=True)

    N_matches = src_pts_xy.shape[0]
    min_matches_for_ransac = 6

    if N_matches < min_matches_for_ransac:
        raise ValueError("Not enough matches.")

    success, camera_r, camera_t, inliers = cv2.solvePnPRansac(coords_c1_3d_xy_numpy_fg, dst_pts_np, K2_np, None)

    camera_r = torch.from_numpy(camera_r).cuda().squeeze().to(torch.float32)
    camera_t = torch.from_numpy(camera_t).cuda().to(torch.float32)
    inliers = torch.from_numpy(inliers).cuda().to(torch.bool)[:, 0]

    triangulated_points = coords_c1_3d_xy_numpy_fg[inliers]

    return camera_r, camera_t, inliers, triangulated_points

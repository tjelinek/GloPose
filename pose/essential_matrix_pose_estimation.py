import cv2
import kornia
import numpy as np
import pygcransac
import pymagsac
import torch
from kornia import vec_like, eye_like
from kornia.geometry import rotation_matrix_to_axis_angle, motion_from_essential_choose_solution, projection_from_KRt, \
    triangulate_points, axis_angle_to_rotation_matrix, vector_to_skew_symmetric_matrix
from nonmin_pose import C2P

from tracker_config import TrackerConfig
from utils import tensor_index_to_coordinates_xy


def estimate_pose_using_2D_2D_E_solver(src_pts_yx: torch.Tensor, dst_pts_yx: torch.Tensor, K1: torch.Tensor,
                                       K2: torch.Tensor, width: int, height: int, ransac_config: TrackerConfig,
                                       confidences=None):
    # Convert to x, y order
    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

    src_pts_np = src_pts_xy.numpy(force=True)
    dst_pts_np = dst_pts_xy.numpy(force=True)

    K1_np = K1.numpy(force=True)
    K2_np = K2.numpy(force=True)

    N_matches = src_pts_xy.shape[0]
    min_matches_for_ransac = 5

    E_method = ransac_config.ransac_essential_matrix_algorithm
    ransac_confidence = ransac_config.ransac_confidence
    if E_method == 'pygcransac' and N_matches >= min_matches_for_ransac:
        correspondences = np.ascontiguousarray(np.concatenate([src_pts_np, dst_pts_np], axis=1))

        if confidences is not None:
            confidences_np = torch.numpy(confidences)
            E, mask = pygcransac.findEssentialMatrix(correspondences, K1_np, K2_np, height, width, height, width,
                                                     confidences_np, threshold=0.1, min_iters=10000)
        else:
            E, mask = pygcransac.findEssentialMatrix(correspondences, K1_np, K2_np, height, width, height, width,
                                                     ransac_confidence, threshold=0.1, min_iters=10000)

    elif E_method is not None and N_matches >= min_matches_for_ransac:
        methods = {'magsac++': cv2.USAC_MAGSAC,
                   'ransac': cv2.RANSAC,
                   '8point': cv2.USAC_FM_8PTS}
        if E_method not in methods:
            raise ValueError("Unknown RANSAC method")

        chosen_method = methods[E_method]
        K1_np = K1.numpy(force=True)
        E, mask = cv2.findEssentialMat(src_pts_np, dst_pts_np, K1_np, method=chosen_method, threshold=1.,
                                       prob=ransac_confidence)
        mask = mask[:, 0].astype(np.bool_)
    else:
        mask = np.ones(src_pts_yx.shape[0], dtype=np.bool_)

    mask_tensor = torch.from_numpy(mask).cuda()

    if N_matches >= 8:
        if ransac_config.ransac_inlier_pose_method == '8point':

            src_pts_xy_inliers = src_pts_xy[mask_tensor]
            dst_pts_xy_inliers = dst_pts_xy[mask_tensor]

            F_mat = kornia.geometry.epipolar.find_fundamental(src_pts_xy_inliers[None], dst_pts_xy_inliers[None],
                                                              method='8POINT')
            E_inliers = kornia.geometry.epipolar.essential_from_fundamental(F_mat, K1[None], K2[None])
            E = (E_inliers / torch.norm(E_inliers)).squeeze().numpy(force=True)
        elif ransac_config.ransac_inlier_pose_method == 'zaragoza':
            src_pts_yx_inliers = src_pts_xy[mask_tensor]
            dst_pts_yx_inliers = dst_pts_xy[mask_tensor]

            rot_cam, t_cam, _, _ = estimate_pose_using_directly_zaragoza(src_pts_yx_inliers, dst_pts_yx_inliers,
                                                                         K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2])

            rot_cam = rot_cam[[1, 0, 2]]

            tx_cam = vector_to_skew_symmetric_matrix(t_cam[None, :, 0])
            R_cam = axis_angle_to_rotation_matrix(rot_cam[None])

            E = (tx_cam @ R_cam).squeeze().numpy(force=True)

        elif ransac_config.ransac_inlier_pose_method == 'bundle_adjustment':
            raise NotImplementedError("This is not implemented yet")
        else:
            assert E_method is not None  # At least one pose estimation method must run.
    else:
        # TODO think what to do with this - this is when we have 0 inliers, so output R=(id), t=(1, 0, 0)
        E = np.asarray([[0, 0, 0], [0, 0, -1], [0, 1, 0]]).astype(np.float32)
        mask_tensor = torch.zeros_like(mask_tensor, dtype=torch.bool)

    if ransac_config.refine_pose_using_numerical_optimization and N_matches >= 8:
        src_pts_xy_inliers_np = src_pts_xy.numpy(force=True).astype(np.float64)
        dst_pts_xy_inliers_np = dst_pts_xy.numpy(force=True).astype(np.float64)
        correspondences_inliers = np.ascontiguousarray(np.concatenate([src_pts_xy_inliers_np,
                                                                       dst_pts_xy_inliers_np], axis=1))
        # E_best = np.ones_like(E, dtype=np.float64)
        E_old = E.copy()
        E_best = E.astype(np.float64)
        mask_uint64 = mask.astype(np.uint64)
        E, mask = pymagsac.optimizeEssentialMatrix(correspondences_inliers, K1_np, K2_np, mask_uint64, E_best)
        # print(f"This is what optimizeEssentialMatrix gave\n{E}")
        if np.linalg.norm(E_old - E) < 0.01:
            breakpoint()

    E_tensor = torch.from_numpy(E).cuda().to(torch.float32)

    R, t_cam, triangulated_points = motion_from_essential_choose_solution(E_tensor, K1, K2, src_pts_yx, dst_pts_yx,
                                                                          mask_tensor)
    r_cam = rotation_matrix_to_axis_angle(R.contiguous())
    if ransac_config.refine_pose_using_numerical_optimization:
        # TODO this is a nasty thing, but it seems that in this implementation, the directions are interchanged
        r_cam = -r_cam

    # r_cam_deg = torch.rad2deg(torch.stack(euler_from_quaternion(*axis_angle_to_quaternion(r_cam))))
    # print('----------------------------------------')
    # print("---t_cam", t_cam.squeeze().round(decimals=3))
    # print("---r_cam", r_cam_deg.squeeze().round(decimals=3))
    # print('----------------------------------------')

    return r_cam, t_cam, mask_tensor, triangulated_points


def triangulate_points_from_Rt(R_cam: torch.Tensor, t_cam: torch.Tensor, src_pts_yx: torch.Tensor,
                               dst_pts_yx: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor) -> torch.Tensor:
    # set reference view pose and compute projection matrix
    R1 = eye_like(3, K1)  # Bx3x3
    t1 = vec_like(3, K1)  # Bx3x1

    # compute the projection matrices for first camera
    P1 = projection_from_KRt(K1, R1, t1)  # Bx3x4

    # compute the projection matrices for second camera
    R2 = R_cam
    t2 = t_cam

    P2 = projection_from_KRt(K2, R2, t2)  # Bx3x4

    X = triangulate_points(P1, P2, src_pts_yx, dst_pts_yx)

    return X


def relative_scale_recovery(essential_matrix_data, flow_arc, K1):
    flow_source, flow_target = flow_arc

    flow_arc_prev = (flow_source, flow_target - 1)

    inlier_mask1 = essential_matrix_data.inlier_mask[flow_arc_prev]
    inlier_mask2 = essential_matrix_data.inlier_mask[flow_arc]

    # extend_inlier_mask1(inlier_mask1, inlier_mask2, src_pts_yx_current, src_pts_yx_prev)

    # common_inlier_mask = (inlier_mask1 & inlier_mask2)
    # common_inlier_indices = torch.nonzero(common_inlier_mask, as_tuple=True)

    inliers_src_pts1 = essential_matrix_data.source_points[flow_arc_prev][inlier_mask1]
    inliers_dst_pts1 = essential_matrix_data.target_points[flow_arc_prev][inlier_mask1]
    inliers_src_pts2 = inliers_dst_pts1
    inliers_dst_pts2 = essential_matrix_data.target_points[flow_arc][inlier_mask2]

    triangulated_points_1 = essential_matrix_data.ransac_triangulated_points[flow_arc_prev]
    triangulated_points_2 = essential_matrix_data.ransac_triangulated_points[flow_arc]

    N_point = triangulated_points_1.shape[0]

    pairs_N = min(100, N_point)

    random_pairs_indices = torch.randperm(N_point)[:pairs_N * 2].view(-1, 2)

    pts1_1i = triangulated_points_1[random_pairs_indices[:, 0]]
    pts2_2i = triangulated_points_1[random_pairs_indices[:, 1]]

    pts1_ij = triangulated_points_2[random_pairs_indices[:, 0]]
    pts2_ij = triangulated_points_2[random_pairs_indices[:, 1]]

    # breakpoint()
    ratio = torch.linalg.norm(triangulated_points_1 - triangulated_points_2)


def compute_bearing_vectors(pts_xy, focal_x, focal_y, c_x, c_y):
    normalized_x = (pts_xy[:, 0] - c_x) / focal_x
    normalized_y = (pts_xy[:, 1] - c_y) / focal_y

    bearing_vectors = torch.stack([normalized_x, normalized_y, torch.ones_like(normalized_x)], dim=1)
    bearing_vectors = bearing_vectors / torch.norm(bearing_vectors, dim=1, keepdim=True)

    return bearing_vectors


def estimate_pose_using_directly_zaragoza(src_pts_yx: torch.Tensor, dst_pts_yx: torch.Tensor, focal_x: torch.Tensor,
                                          focal_y: torch.Tensor, c_x: torch.Tensor, c_y: torch.Tensor):
    configuration = {
        # threshold for the singular values of X to check if rank(X)\in[1,3] (condition for tightness),
        # where X is the SDP-solution submatrix corresponding to E, t, q and h.
        "th_rank_optimality": 1e-5,  # default
        # threshold for the slack variable "st" for detecting (near-)pure rotations.
        "th_pure_rot_sdp": 1e-3,  # default
        # threshold for "st" used for detecting noise-free pure rotations
        # and improving the numerical accuracy.
        "th_pure_rot_noisefree_sdp": 1e-4,  # default
    }

    solver = C2P(cfg=configuration)

    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

    src_pts_bearings_xy = compute_bearing_vectors(src_pts_xy, focal_x, focal_y, c_x, c_y)
    dst_pts_bearings_xy = compute_bearing_vectors(dst_pts_xy, focal_x, focal_y, c_x, c_y)

    src_pts_xy_bearings_np = src_pts_bearings_xy.numpy(force=True).T
    dst_pts_xy_bearings_np = dst_pts_bearings_xy.numpy(force=True).T

    mask_tensor = torch.ones(src_pts_yx.shape[0], dtype=torch.bool).cuda()  # TODO fill me with something meaningful
    triangulated_points = torch.zeros_like(src_pts_bearings_xy).cuda()  # TODO fill me with something meaningful

    if src_pts_yx.shape[0] < 5:
        R_cam = torch.eye(3).to(torch.float32).cuda()
        t_cam = torch.zeros(3).to(torch.float32).cuda().unsqueeze(-1)
    else:
        solution = solver(src_pts_xy_bearings_np, dst_pts_xy_bearings_np)

        R_cam = torch.from_numpy(solution["R01"]).to(torch.float32).cuda()
        t_cam = torch.from_numpy(solution["t01"]).to(torch.float32).cuda()

    r_cam = rotation_matrix_to_axis_angle(R_cam)

    return r_cam, t_cam, mask_tensor, triangulated_points


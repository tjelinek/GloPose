import cv2
import kornia
import numpy as np
import pygcransac
import torch
from kornia import vec_like, eye_like
from kornia.geometry import rotation_matrix_to_axis_angle, motion_from_essential_choose_solution, projection_from_KRt, \
    triangulate_points, Rt_to_matrix4x4, inverse_transformation, matrix4x4_to_Rt
from nonmin_pose import C2P
# from pymagsac import optimizeEssentialMatrix


def filter_inliers_using_ransac(src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor, K1: torch.Tensor, K2: torch.Tensor,
                                width: int, height: int, ransac_method, ransac_confidence, confidences=None,
                                ransac_refine_E_numerically: bool = False):
    src_pts_xy_np = src_pts_xy.numpy(force=True)
    dst_pts_xy_np = dst_pts_xy.numpy(force=True)

    K1_np = K1.numpy(force=True)
    K2_np = K2.numpy(force=True)

    N_matches = src_pts_xy.shape[0]
    min_matches_for_ransac = 5

    if ransac_method == 'pygcransac' and N_matches >= min_matches_for_ransac:
        correspondences = np.ascontiguousarray(np.concatenate([src_pts_xy_np, dst_pts_xy_np], axis=1))

        if confidences is not None:
            confidences_np = confidences.numpy(force=True)
            ordering = confidences_np.argsort()[::-1]

            confidences_np = confidences_np[ordering]
            correspondences = correspondences[ordering]

            # sampler = 1 is PROSAC
            E, mask = pygcransac.findEssentialMatrix(correspondences, K1_np, K2_np, height, width, height, width,
                                                     confidences_np, threshold=0.1, min_iters=1000, sampler=1)
        else:
            E, mask = pygcransac.findEssentialMatrix(correspondences, K1_np, K2_np, height, width, height, width,
                                                     ransac_confidence, threshold=0.1, min_iters=10000)

    elif ransac_method is not None and N_matches >= min_matches_for_ransac:
        methods = {'magsac++': cv2.USAC_MAGSAC,
                   'ransac': cv2.RANSAC,
                   '8point': cv2.USAC_FM_8PTS}
        if ransac_method not in methods:
            raise ValueError("Unknown RANSAC method")

        chosen_method = methods[ransac_method]
        K1_np = K1.numpy(force=True)
        E, mask = cv2.findEssentialMat(src_pts_xy_np, dst_pts_xy_np, K1_np, method=chosen_method, threshold=1.,
                                       prob=ransac_confidence)
        mask = mask[:, 0].astype(np.bool_)
    else:
        raise ValueError("Not enough points to run 5pt algorithm RANSAC")

    if ransac_refine_E_numerically:
        E = refine_pose_using_numerical_optimization(src_pts_xy_np, dst_pts_xy_np, E, K1_np, K2_np, mask)

    mask_tensor = torch.from_numpy(mask).cuda()
    E_tensor = torch.from_numpy(E).cuda().to(torch.float32)

    R, t_cam, triangulated_points = motion_from_essential_choose_solution(E_tensor, K1, K2, src_pts_xy, dst_pts_xy,
                                                                          mask_tensor)
    r_cam = rotation_matrix_to_axis_angle(R.contiguous())

    return r_cam, t_cam, mask_tensor, triangulated_points


def estimate_pose_using_8pt_algorithm(src_pts_xy_inliers: torch.Tensor, dst_pts_xy_inliers: torch.Tensor,
                                      K1: torch.Tensor, K2: torch.Tensor):
    N_matches = src_pts_xy_inliers.shape[0]

    if N_matches < 8:
        raise ValueError("Not Enough Correspondences")

    F_mat = kornia.geometry.epipolar.find_fundamental(src_pts_xy_inliers[None], dst_pts_xy_inliers[None],
                                                      method='8POINT')

    E_inliers = kornia.geometry.epipolar.essential_from_fundamental(F_mat, K1[None], K2[None])
    E = (E_inliers / torch.norm(E_inliers)).squeeze()

    R, t_cam, triangulated_points = motion_from_essential_choose_solution(E, K1, K2, src_pts_xy_inliers,
                                                                          dst_pts_xy_inliers)
    r_cam = rotation_matrix_to_axis_angle(R.contiguous())  # Clockwise->anti-clockwise rotation convention

    return r_cam, t_cam


def refine_pose_using_numerical_optimization(src_pts_xy: np.ndarray, dst_pts_xy: np.ndarray, E: np.ndarray,
                                             K1: np.ndarray, K2: np.ndarray, mask: np.ndarray):
    src_pts_xy_inliers_np = src_pts_xy.astype(np.float64)
    dst_pts_xy_inliers_np = dst_pts_xy.astype(np.float64)
    correspondences_inliers = np.ascontiguousarray(np.concatenate([src_pts_xy_inliers_np,
                                                                   dst_pts_xy_inliers_np], axis=1))

    E_old = E.copy()
    E_best = E.astype(np.float64)
    mask_uint64 = mask.astype(np.uint64)

    E, mask = optimizeEssentialMatrix(correspondences_inliers, K1, K2, mask_uint64, E_best)

    if np.linalg.norm(E_old - E) < 0.01:
        breakpoint()

    return E


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


def estimate_pose_zaragoza(src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor, focal_x: torch.Tensor,
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

    src_pts_bearings_xy = compute_bearing_vectors(src_pts_xy, focal_x, focal_y, c_x, c_y)
    dst_pts_bearings_xy = compute_bearing_vectors(dst_pts_xy, focal_x, focal_y, c_x, c_y)

    src_pts_xy_bearings_np = src_pts_bearings_xy.numpy(force=True).T
    dst_pts_xy_bearings_np = dst_pts_bearings_xy.numpy(force=True).T

    solution = solver(src_pts_xy_bearings_np, dst_pts_xy_bearings_np)

    R_cam_10 = torch.from_numpy(solution["R01"]).to(torch.float32).cuda().unsqueeze(0)
    t_cam_10 = torch.from_numpy(solution["t01"]).to(torch.float32).cuda().unsqueeze(0)

    T_10 = Rt_to_matrix4x4(R_cam_10, t_cam_10)
    T_01 = inverse_transformation(T_10)

    R_cam_01, t_cam_01 = matrix4x4_to_Rt(T_01)
    r_cam_01 = rotation_matrix_to_axis_angle(R_cam_01)

    return r_cam_01, t_cam_01

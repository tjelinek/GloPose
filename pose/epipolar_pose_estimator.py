from typing import Tuple

import numpy as np
import torch
from kaolin.render.camera import PinholeIntrinsics
from kornia.geometry import axis_angle_to_rotation_matrix, Se3, Quaternion

from auxiliary_scripts.cameras import Cameras
from data_structures.data_graph import DataGraph
from auxiliary_scripts.depth import DepthAnythingProvider
from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam, Rt_epipolar_cam_from_Rt_obj
from flow import flow_unit_coords_to_image_coords, source_coords_to_target_coords, get_correct_correspondences_mask
from data_structures.keyframe_buffer import FrameObservation, FlowObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin, RenderedFlowResult, RenderingResult
from pose.essential_matrix_pose_estimation import filter_inliers_using_ransac, triangulate_points_from_Rt, \
    estimate_pose_zaragoza, estimate_pose_using_8pt_algorithm
from pose.pnp_pose_estimation import estimate_pose_using_PnP_solver
from tracker_config import TrackerConfig
from utils import erode_segment_mask2, dilate_mask, get_not_occluded_foreground_points, pinhole_intrinsics_from_tensor, \
    tensor_index_to_coordinates_xy, pinhole_intrinsics_to_tensor


class EpipolarPoseEstimator:

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, gt_rotations, gt_translations,
                 rendering: RenderingKaolin, rendering_backview: RenderingKaolin, gt_encoder,
                 camera_intrinsics: PinholeIntrinsics = None):

        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.gt_rotations = gt_rotations
        self.gt_translations = gt_translations
        self.rendering: RenderingKaolin = rendering
        self.rendering_backview: RenderingKaolin = rendering_backview
        self.gt_encoder: Encoder = gt_encoder
        self.depth_anything: DepthAnythingProvider = DepthAnythingProvider()

        self.occlusion_threshold = self.config.occlusion_coef_threshold
        self.segmentation_threshold = self.config.segmentation_mask_threshold

        if camera_intrinsics is None:
            self.camera_intrinsics = pinhole_intrinsics_from_tensor(self.rendering.camera_intrinsics,
                                                                    self.rendering.width,
                                                                    self.rendering.height)
        else:
            self.camera_intrinsics = camera_intrinsics

    def estimate_pose_using_optical_flow(self, flow_observations: FlowObservation, flow_arc_idx, flow_arc,
                                         camera_observation: FrameObservation, backview=False) -> Tuple[float, Se3]:

        K1 = K2 = pinhole_intrinsics_to_tensor(self.camera_intrinsics).cuda()

        W_4x4 = self.rendering.camera_transformation_matrix_4x4().permute(0, 2, 1)

        flow_observation_current_frame: FlowObservation = flow_observations.filter_frames([flow_arc_idx])

        gt_flow_observation, occlusions, segmentation, rendered_obj_cam0_coords = (
            self.get_occlusion_and_segmentation(backview, flow_arc, flow_observation_current_frame))

        flow = flow_observation_current_frame.cast_unit_coords_to_image_coords().observed_flow

        observed_segmentation_binary_mask = segmentation > float(self.segmentation_threshold)

        src_pts_yx, observed_visible_fg_points_mask = (
            get_not_occluded_foreground_points(occlusions, segmentation, self.occlusion_threshold,
                                               self.segmentation_threshold))

        _, gt_visible_fg_points_mask = (get_not_occluded_foreground_points(
            gt_flow_observation.rendered_flow_occlusion, gt_flow_observation.rendered_flow_segmentation,
            self.occlusion_threshold, self.segmentation_threshold))

        dst_pts_yx = source_coords_to_target_coords_world_frame(src_pts_yx.permute(1, 0), flow).permute(1, 0)

        gt_flow = flow_unit_coords_to_image_coords(gt_flow_observation.theoretical_flow)
        dst_pts_yx_gt_flow = source_coords_to_target_coords_world_frame(src_pts_yx.permute(1, 0), gt_flow).permute(1, 0)

        confidences = None
        if self.config.ransac_confidences_from_occlusion:
            confidences = 1 - occlusions[0, 0, 0, src_pts_yx[:, 0].to(torch.long), src_pts_yx[:, 1].to(torch.long)]

        confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx = self.augment_correspondences(src_pts_yx, dst_pts_yx,
                                                                                               dst_pts_yx_gt_flow,
                                                                                               confidences,
                                                                                               gt_flow_image_coord)

        src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
        dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)
        dst_pts_xy_gt_flow = tensor_index_to_coordinates_xy(dst_pts_yx_gt_flow)

        if self.config.ransac_inlier_filter == 'pnp_ransac':
            point_map_xy = rendered_obj_cam0_coords[0, 0, [1, 0, 2]].permute(1, 0, 2)

            # observed_image_target = camera_observation.observed_image[:, 0]
            # depth_image = self.depth_anything.infer_depth_anything(observed_image_target)

            result = estimate_pose_using_PnP_solver(src_pts_xy, dst_pts_xy, K1, K2, point_map_xy)

            rot_cam_ransac, t_cam_ransac, inlier_mask, triangulated_points_ransac = result
        elif self.config.ransac_inlier_filter in ['magsac++', 'ransac', '8point', 'pygcransac']:
            result = filter_inliers_using_ransac(src_pts_xy, dst_pts_xy, K1, K2, self.camera_intrinsics.width,
                                                 self.camera_intrinsics.height, self.config.ransac_inlier_filter,
                                                 self.config.ransac_confidence, confidences,
                                                 ransac_refine_E_numerically=self.config.ransac_refine_E_numerically)

            rot_cam_ransac, t_cam_ransac, inlier_mask, triangulated_points_ransac = result
        elif self.config.ransac_inlier_filter is None:
            inlier_mask = torch.ones(src_pts_yx.shape[0], dtype=torch.bool)
            rot_cam_ransac = None
            t_cam_ransac = None
            triangulated_points_ransac = None
        else:
            raise ValueError("Unknown RANSAC method")

        src_pts_yx_inliers = src_pts_yx[inlier_mask]
        dst_pts_yx_inliers = dst_pts_yx[inlier_mask]
        src_pts_xy_inliers = src_pts_xy[inlier_mask]
        dst_pts_xy_inliers = dst_pts_xy[inlier_mask]
        dst_pts_yx_gt_flow_inliers = dst_pts_yx_gt_flow[inlier_mask]
        dst_pts_xy_gt_flow_inliers = dst_pts_xy_gt_flow[inlier_mask]

        if self.config.ransac_inlier_pose_method == '8point':
            result = estimate_pose_using_8pt_algorithm(src_pts_xy_inliers, dst_pts_xy_inliers, K1, K2)
            r_cam, t_cam = result

        elif self.config.ransac_inlier_pose_method == 'zaragoza':
            # raise NotImplementedError('The intrinsics are wrong and return row of matrix rather than a correct value')
            r_cam, t_cam = estimate_pose_zaragoza(src_pts_xy_inliers, dst_pts_xy_inliers,
                                                  self.camera_intrinsics.focal_x.cuda(),
                                                  self.camera_intrinsics.focal_y.cuda(),
                                                  self.camera_intrinsics.x0.cuda(),
                                                  self.camera_intrinsics.y0.cuda())
        elif self.config.ransac_inlier_pose_method is None:
            if self.config.ransac_inlier_filter is None:
                raise ValueError("At least one of 'ransac_inlier_filter' or 'ransac_inlier_filter' must not be None.")
            r_cam = rot_cam_ransac
            t_cam = t_cam_ransac
        else:
            raise ValueError("Unknown inlier pose method")

        R_cam = axis_angle_to_rotation_matrix(r_cam[None])

        quat_cam = Quaternion.from_matrix(R_cam)
        Se3_cam = Se3(quat_cam, t_cam.squeeze()[None])

        R_obj, t_obj = Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam[None], W_4x4)
        t_obj = t_obj[..., 0]  # Shape (1, 3, 1) -> (1, 3)

        quat_obj = Quaternion.from_matrix(R_obj)
        Se3_obj = Se3(quat_obj, t_obj)

        common_inlier_indices = torch.nonzero(inlier_mask, as_tuple=True)
        outlier_indices = torch.nonzero(~inlier_mask, as_tuple=True)
        inlier_src_pts = src_pts_yx[common_inlier_indices]
        outlier_src_pts = src_pts_yx[outlier_indices]

        # max(1, x) avoids division by zero
        inlier_ratio = len(inlier_src_pts) / max(1, len(inlier_src_pts) + len(outlier_src_pts))

        camera1 = Cameras.BACKVIEW if backview else Cameras.FRONTVIEW
        data = self.data_graph.get_edge_observations(*flow_arc, camera=camera1)
        data.src_pts_yx = src_pts_yx.cpu()
        data.dst_pts_yx = dst_pts_yx.cpu()
        data.dst_pts_yx_gt = dst_pts_yx_gt_flow.cpu()
        data.observed_flow_segmentation = observed_segmentation_binary_mask.cpu()
        data.observed_visible_fg_points_mask = observed_visible_fg_points_mask.cpu()
        data.gt_visible_fg_points_mask = gt_visible_fg_points_mask.cpu()
        data.ransac_inliers = inlier_src_pts.cpu()
        data.ransac_outliers = outlier_src_pts.cpu()
        data.ransac_triangulated_points = triangulated_points_ransac.cpu()

        gt_rot_axis_angle_obj = self.gt_rotations[:, flow_arc[1]]
        gt_R_obj = axis_angle_to_rotation_matrix(gt_rot_axis_angle_obj)
        gt_trans_obj = self.gt_translations[0, :, flow_arc[1]].unsqueeze(-1)

        gt_R_cam, gt_trans_cam = Rt_epipolar_cam_from_Rt_obj(gt_R_obj, gt_trans_obj, W_4x4)

        ransac_triangulated_points_gt_Rt = triangulate_points_from_Rt(gt_R_cam, gt_trans_cam, src_pts_yx[None],
                                                                      dst_pts_yx[None], K1[None], K2[None])
        ransac_triangulated_points_gt_Rt_gt_flow = triangulate_points_from_Rt(gt_R_cam, gt_trans_cam, src_pts_yx[None],
                                                                              dst_pts_yx_gt_flow[None], K1[None],
                                                                              K2[None])

        data.ransac_triangulated_points_gt_Rt = ransac_triangulated_points_gt_Rt.cpu()
        data.ransac_triangulated_points_gt_Rt_gt_flow = ransac_triangulated_points_gt_Rt_gt_flow.cpu()
        data.ransac_inliers_mask = inlier_mask.cpu()
        data.ransac_inlier_ratio = inlier_ratio

        return inlier_ratio, Se3_obj

    def get_occlusion_and_segmentation(self, backview, flow_arc, flow_observation_current_frame):
        renderer: RenderingKaolin = self.rendering_backview if backview else self.rendering
        gt_flow_observation: RenderedFlowResult = renderer.render_flow_for_frame(self.gt_encoder, *flow_arc)

        gt_rendering_result: RenderingResult = renderer.rendering_result_for_frame(self.gt_encoder, 0)
        rendered_obj_cam0_coords = gt_rendering_result.rendered_face_camera_coords

        if self.config.ransac_erode_segmentation:
            eroded_gt_seg = erode_segment_mask2(5, gt_flow_observation.rendered_flow_segmentation[0])
            eroded_observed_seg = erode_segment_mask2(5, flow_observation_current_frame.observed_flow_segmentation[0])

            flow_observation_current_frame.observed_flow_segmentation = eroded_observed_seg[None]
            gt_flow_observation = gt_flow_observation._replace(rendered_flow_segmentation=eroded_gt_seg[None])

        if self.config.ransac_dilate_occlusion:
            dilated_gt_occ = dilate_mask(1, gt_flow_observation.rendered_flow_occlusion)
            dilated_observed_occ = dilate_mask(1, flow_observation_current_frame.observed_flow_occlusion)

            gt_flow_observation = gt_flow_observation._replace(rendered_flow_occlusion=dilated_gt_occ)
            flow_observation_current_frame.observed_flow_occlusion = dilated_observed_occ
        if self.config.ransac_use_gt_occlusions_and_segmentation:
            occlusions = gt_flow_observation.rendered_flow_occlusion
            segmentation = gt_flow_observation.rendered_flow_segmentation
        else:
            occlusions = flow_observation_current_frame.observed_flow_occlusion
            segmentation = flow_observation_current_frame.observed_flow_segmentation
        return gt_flow_observation, occlusions, segmentation, rendered_obj_cam0_coords

    def augment_correspondences(self, src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow, confidences, gt_flow_image_coord):
        if self.config.ransac_feed_only_inlier_flow:
            confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx = (
                self.filter_outlier_flow(src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow, confidences, gt_flow_image_coord))
        if self.config.ransac_replace_mft_flow_with_gt_flow:
            dst_pts_yx = self.replace_mft_flow_with_gt_flow(src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow)
        if self.config.ransac_distant_pixels_sampling:
            confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx = (
                self.distant_pixels_sampling(src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow, confidences))
        return confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx

    def distant_pixels_sampling(self, src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow, confidences):
        random_src_pts_permutation = np.random.default_rng(seed=42).permutation(src_pts_yx.shape[0])
        random_permutation_indices = random_src_pts_permutation[:min(self.config.ransac_distant_pixels_sample_size,
                                                                     src_pts_yx.shape[0])]
        dst_pts_yx = dst_pts_yx[random_permutation_indices]
        src_pts_yx = src_pts_yx[random_permutation_indices]
        dst_pts_yx_gt_flow = dst_pts_yx_gt_flow[random_permutation_indices]
        if self.config.ransac_confidences_from_occlusion:
            confidences = confidences[random_permutation_indices]
        return confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx

    def filter_outlier_flow(self, src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow, confidences, gt_flow_image_coord):
        ok_pts_indices = get_correct_correspondences_mask(gt_flow_image_coord, src_pts_yx, dst_pts_yx,
                                                          self.config.ransac_feed_only_inlier_flow_epe_threshold)
        dst_pts_yx = dst_pts_yx[ok_pts_indices]
        src_pts_yx = src_pts_yx[ok_pts_indices]
        dst_pts_yx_gt_flow = dst_pts_yx_gt_flow[ok_pts_indices]
        if self.config.ransac_confidences_from_occlusion:
            confidences = confidences[ok_pts_indices]
        return confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx

    def replace_mft_flow_with_gt_flow(self, src_pts_yx, dst_pts_yx, dst_pts_yx_gt_flow):
        n_points = src_pts_yx.shape[0]
        n_points_injected = int(n_points * self.config.ransac_feed_gt_flow_percentage)
        indices_permutation = torch.randperm(n_points)
        indices_to_be_replaced = indices_permutation[:n_points_injected]
        dst_pts_yx_gt_for_replacing = dst_pts_yx_gt_flow[indices_to_be_replaced].clone()
        if self.config.ransac_feed_gt_flow_add_gaussian_noise:
            if self.config.ransac_feed_gt_flow_add_gaussian_noise_use_mft_errors:
                errors: torch.Tensor = dst_pts_yx_gt_for_replacing - dst_pts_yx[indices_to_be_replaced]
                mean = errors.mean(dim=0)
                sigma = errors.var(dim=0).sqrt()
            else:
                sigma = self.config.ransac_feed_gt_flow_add_gaussian_noise_sigma
                mean = self.config.ransac_feed_gt_flow_add_gaussian_noise_mean
            dst_pts_yx_gt_for_replacing += sigma * torch.randn(n_points_injected, 2,
                                                               device=dst_pts_yx_gt_for_replacing.device) + mean
        dst_pts_yx[indices_to_be_replaced] = dst_pts_yx_gt_for_replacing

        return dst_pts_yx

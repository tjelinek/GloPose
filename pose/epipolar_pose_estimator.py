import kaolin
import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle, axis_angle_to_quaternion

from auxiliary_scripts.cameras import Cameras
from auxiliary_scripts.data_structures import DataGraph
from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam, Rt_epipolar_cam_from_Rt_obj
from flow import flow_unit_coords_to_image_coords, source_coords_to_target_coords, get_correct_correspondences_mask
from keyframe_buffer import FrameObservation, FlowObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin, RenderedFlowResult
from pose.dust3r import get_matches_using_dust3r
from pose.essential_matrix_pose_estimation import estimate_pose_using_dense_correspondences, triangulate_points_from_Rt
from tracker_config import TrackerConfig
from utils import homogenize_3x4_transformation_matrix, erode_segment_mask2, dilate_mask, \
    get_not_occluded_foreground_points


class EpipolarPoseEstimator:

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, gt_rotations, gt_translations,
                 rendering: RenderingKaolin, rendering_backview: RenderingKaolin, gt_encoder):

        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.gt_rotations = gt_rotations
        self.gt_translations = gt_translations
        self.rendering: RenderingKaolin = rendering
        self.rendering_backview: RenderingKaolin = rendering_backview
        self.gt_encoder: Encoder = gt_encoder

    def estimate_pose_using_optical_flow(self, flow_observations, flow_arc_idx, flow_arc,
                                         camera_observation: FrameObservation, backview=False):

        K1 = K2 = self.rendering.camera_intrinsics

        camera_translation = self.rendering.camera_trans
        if backview:
            camera_translation = -camera_translation

        W_4x3 = kaolin.render.camera.generate_transformation_matrix(camera_position=camera_translation,
                                                                    camera_up_direction=self.rendering.camera_up,
                                                                    look_at=self.rendering.obj_center)

        W_4x4 = homogenize_3x4_transformation_matrix(W_4x3.permute(0, 2, 1))

        camera = Cameras.BACKVIEW if backview else Cameras.FRONTVIEW
        arc_data = self.data_graph.get_edge_observations(*flow_arc, camera=camera)

        flow_observation_current_frame: FlowObservation = flow_observations.filter_frames([flow_arc_idx])

        gt_flow_observation, occlusions, segmentation = (
            self.get_occlusion_and_segmentation(backview, flow_arc, flow_observation_current_frame))

        arc_data.observed_flow = flow_observation_current_frame.send_to_device('cpu')

        optical_flow = flow_unit_coords_to_image_coords(flow_observation_current_frame.observed_flow)

        observed_segmentation_binary_mask = segmentation > float(self.config.segmentation_mask_threshold)
        gt_segmentation_binary_mask = (gt_flow_observation.rendered_flow_segmentation >
                                       float(self.config.segmentation_mask_threshold))

        src_pts_yx, observed_visible_fg_points_mask = (
            get_not_occluded_foreground_points(occlusions, segmentation,
                                               self.config.occlusion_coef_threshold,
                                               self.config.segmentation_mask_threshold))

        _, gt_visible_fg_points_mask = (get_not_occluded_foreground_points(
            gt_flow_observation.rendered_flow_occlusion, gt_flow_observation.rendered_flow_segmentation,
            self.config.occlusion_coef_threshold, self.config.segmentation_mask_threshold))

        dst_pts_yx = source_coords_to_target_coords(src_pts_yx.permute(1, 0), optical_flow).permute(1, 0)
        gt_flow_image_coord = flow_unit_coords_to_image_coords(gt_flow_observation.theoretical_flow)
        dst_pts_yx_gt_flow = source_coords_to_target_coords(src_pts_yx.permute(1, 0), gt_flow_image_coord).permute(1, 0)
        src_pts_yx_gt_flow = src_pts_yx.clone()

        pts3d_dust3r = None
        if self.config.ransac_use_dust3r:
            observed_images = camera_observation.observed_image[0][[0, -1]]
            images_sequence = list(torch.unbind(observed_images, 0))
            src_pts_yx_dust3r, dst_pts_yx_dust3r, pts3d_dust3r = get_matches_using_dust3r(images_sequence)

            dust3r_fg_points_mask = gt_segmentation_binary_mask[0, 0, 0, src_pts_yx_dust3r[:, 0], src_pts_yx_dust3r[:, 1]]
            src_pts_yx = src_pts_yx_dust3r[dust3r_fg_points_mask][:, [1, 0]].to(torch.float32)
            dst_pts_yx = dst_pts_yx_dust3r[dust3r_fg_points_mask][:, [1, 0]].to(torch.float32)


        if self.config.ransac_confidences_from_occlusion:
            confidences = 1 - occlusions[0, 0, 0, src_pts_yx[:, 0].to(torch.long), src_pts_yx[:, 1].to(torch.long)]
        else:
            confidences = None

        confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx = self.augment_correspondences(src_pts_yx, dst_pts_yx,
                                                                                               dst_pts_yx_gt_flow,
                                                                                               confidences,
                                                                                               gt_flow_image_coord)

        result = estimate_pose_using_dense_correspondences(src_pts_yx, dst_pts_yx, K1, K2, self.rendering.width,
                                                           self.rendering.height, self.config, confidences)

        rot_cam, t_cam, inlier_mask, triangulated_points = result

        # if flow_arc[1] > 1:
        #
        #     data_suffix = 'back' if backview else 'front'
        #
        #     relative_scale_recovery(essential_matrix_data, flow_arc, K1)

        R_cam = axis_angle_to_rotation_matrix(rot_cam[None])

        R_obj, t_obj = Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam[None], W_4x4)

        rot_obj = rotation_matrix_to_axis_angle(R_obj)
        quat_obj = axis_angle_to_quaternion(rot_obj).squeeze()

        t_obj = t_obj.squeeze()

        common_inlier_indices = torch.nonzero(inlier_mask, as_tuple=True)
        outlier_indices = torch.nonzero(~inlier_mask, as_tuple=True)
        inlier_src_pts = src_pts_yx[common_inlier_indices]
        outlier_src_pts = src_pts_yx[outlier_indices]

        # max(1, x) avoids division by zero
        inlier_ratio = len(inlier_src_pts) / max(1, len(inlier_src_pts) + len(outlier_src_pts))

        # LOG RANSAC RESULT
        gt_flow_cpu = RenderedFlowResult(theoretical_flow=gt_flow_observation.theoretical_flow.detach().cpu(),
                                         rendered_flow_segmentation=gt_flow_observation.rendered_flow_segmentation.detach().cpu(),
                                         rendered_flow_occlusion=gt_flow_observation.rendered_flow_occlusion.detach().cpu())
        camera1 = Cameras.BACKVIEW if backview else Cameras.FRONTVIEW
        data = self.data_graph.get_edge_observations(*flow_arc, camera=camera1)
        data.src_pts_yx = src_pts_yx.cpu()
        data.dst_pts_yx = dst_pts_yx.cpu()
        data.dst_pts_yx_gt = dst_pts_yx_gt_flow.cpu()
        data.gt_flow_result = gt_flow_cpu
        data.observed_flow_segmentation = observed_segmentation_binary_mask.cpu()
        data.gt_flow_segmentation = gt_segmentation_binary_mask.cpu()
        data.observed_visible_fg_points_mask = observed_visible_fg_points_mask.cpu()
        data.gt_visible_fg_points_mask = gt_visible_fg_points_mask.cpu()
        data.ransac_inliers = inlier_src_pts.cpu()
        data.ransac_outliers = outlier_src_pts.cpu()
        data.ransac_triangulated_points = triangulated_points.cpu()
        data.dust3r_point_cloud = pts3d_dust3r.flatten(0, 1).cpu()

        # quat_obj = self.gt_rotations
        gt_rot_axis_angle_obj = self.gt_rotations[:, flow_arc[1]]
        gt_R_obj = axis_angle_to_rotation_matrix(gt_rot_axis_angle_obj)
        gt_trans_obj = self.gt_translations[0, :, flow_arc[1]].unsqueeze(-1)

        gt_R_cam, gt_trans_cam = Rt_epipolar_cam_from_Rt_obj(gt_R_obj, gt_trans_obj, W_4x4)

        # gt_trans_cam = torch.zeros(1, 3, 1).cuda()
        # gt_trans_cam[:, 0] = 1.

        ransac_triangulated_points_gt_Rt = triangulate_points_from_Rt(gt_R_cam, gt_trans_cam, src_pts_yx[None],
                                                                      dst_pts_yx[None], K1[None], K2[None])
        ransac_triangulated_points_gt_Rt_gt_flow = triangulate_points_from_Rt(gt_R_cam, gt_trans_cam, src_pts_yx[None],
                                                                              dst_pts_yx_gt_flow[None], K1[None],
                                                                              K2[None])

        data.ransac_triangulated_points_gt_Rt = ransac_triangulated_points_gt_Rt
        data.ransac_triangulated_points_gt_Rt_gt_flow = ransac_triangulated_points_gt_Rt_gt_flow
        data.ransac_inliers_mask = inlier_mask.cpu()
        data.ransac_inlier_ratio = inlier_ratio

        return inlier_ratio, quat_obj, t_obj

    def get_occlusion_and_segmentation(self, backview, flow_arc, flow_observation_current_frame):
        renderer: RenderingKaolin = self.rendering_backview if backview else self.rendering
        gt_flow_observation: RenderedFlowResult = renderer.render_flow_for_frame(self.gt_encoder, *flow_arc)
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
        return gt_flow_observation, occlusions, segmentation

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

from typing import Tuple, List
from time import time

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, quaternion_to_axis_angle, PinholeCamera

from auxiliary_scripts.cameras import Cameras
from auxiliary_scripts.math_utils import Se3_obj_from_epipolar_Se3_cam, quaternion_minimal_angular_difference, \
    Se3_epipolar_cam_from_Se3_obj
from data_structures.data_graph import DataGraph
from auxiliary_scripts.depth import DepthAnythingProvider
from data_structures.pose_icosphere import PoseIcosphere
from flow import get_correct_correspondence_mask_world_system, source_to_target_coords_world_coord_system
from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, BaseFlowObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin, RenderingResult
from pose.essential_matrix_pose_estimation import (filter_inliers_using_ransac, estimate_pose_zaragoza,
                                                   estimate_pose_using_8pt_algorithm)
from pose.pnp_pose_estimation import estimate_pose_using_PnP_solver
from tracker_config import TrackerConfig
from utils import erode_segment_mask2, dilate_mask, get_not_occluded_foreground_points, tensor_index_to_coordinates_xy


class EpipolarPoseEstimator:

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, rendering: RenderingKaolin, gt_encoder, encoder,
                 pose_icosphere, camera: PinholeCamera):

        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.rendering: RenderingKaolin = rendering
        self.encoder: Encoder = encoder
        self.gt_encoder: Encoder = gt_encoder
        self.pose_icosphere: PoseIcosphere = pose_icosphere
        self.depth_anything: DepthAnythingProvider = DepthAnythingProvider()

        self.image_width: int = int(camera.width.item())
        self.image_height: int = int(camera.height.item())

        self.occlusion_threshold = self.config.occlusion_coef_threshold
        self.segmentation_threshold = self.config.segmentation_mask_threshold

        self.camera = camera

    @torch.no_grad()
    def essential_matrix_preinitialization(self, keyframes, flow_tracks_inits):

        start_time = time()
        frame_i = max(keyframes)

        flow_arc_long_jump = (flow_tracks_inits[-1], frame_i)

        flow_arc_short_jump = (frame_i - 1, frame_i)

        flow_long_jump_source, flow_long_jump_target = flow_arc_long_jump
        flow_short_jump_source, flow_short_jump_target = flow_arc_short_jump

        flow_long_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_long_jump).
                                                        observed_flow).cuda()
        flow_short_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_short_jump).
                                                         observed_flow).cuda()

        short_jumps_data = [self.data_graph.get_edge_observations(i, i + 1)
                            for i in range(flow_long_jump_source, frame_i - 1)]

        pred_short_deltas_se3: List[Se3] = [data.predicted_obj_delta_se3 for data in short_jumps_data]

        if self.config.long_short_flow_chaining_pixel_level_verification:
            short_jumps_flows_ref_to_current: List[FlowObservation] = \
                [data.observed_flow for data in short_jumps_data] + [flow_short_jump_observations]

            chained_short_jumps_flows_ref_to_current = FlowObservation.chain(*short_jumps_flows_ref_to_current)
        else:
            chained_short_jumps_flows_ref_to_current = None

        Se3_cam_short_jump, Se3_cam_short_jump_RANSAC = (
            self.estimate_pose_using_optical_flow(flow_short_jump_observations, flow_arc_short_jump))

        Se3_cam_long_jump, Se3_cam_long_jump_RANSAC = (
            self.estimate_pose_using_optical_flow(flow_long_jump_observations, flow_arc_long_jump,
                                                  chained_short_jumps_flows_ref_to_current))

        Se3_world_to_cam = Se3.from_matrix(self.camera.extrinsics)
        Se3_obj_reference_frame = self.encoder.get_se3_at_frame_vectorized()[[flow_long_jump_source]]
        Se3_obj_short_jump_ref_frame = self.encoder.get_se3_at_frame_vectorized()[[flow_short_jump_source]]

        Se3_cam_reference_frame = Se3_epipolar_cam_from_Se3_obj(Se3_obj_reference_frame, Se3_world_to_cam)
        Se3_cam_short_jump_ref_frame = Se3_epipolar_cam_from_Se3_obj(Se3_obj_short_jump_ref_frame, Se3_world_to_cam)

        if flow_arc_long_jump != flow_arc_short_jump:
            Se3_cam_long_jump, Se3_cam_short_jump = self.relative_scale_recovery(Se3_cam_reference_frame,
                                                                                 Se3_cam_short_jump_ref_frame,
                                                                                 Se3_cam_long_jump, Se3_cam_short_jump)

        Se3_obj_long_jump = Se3_obj_from_epipolar_Se3_cam(Se3_cam_long_jump, Se3_world_to_cam)
        Se3_obj_short_jump = Se3_obj_from_epipolar_Se3_cam(Se3_cam_short_jump, Se3_world_to_cam)

        Se3_obj_chained_long_jump = Se3_obj_long_jump * Se3_obj_reference_frame

        products = reversed([Se3_obj_reference_frame] +
                            pred_short_deltas_se3 +
                            [Se3_obj_short_jump])

        if self.config.icosphere_use_gt_long_jumps:
            Se3_obj_long_jump_gt = self.get_relative_gt_rotation(flow_long_jump_source, flow_long_jump_target)
            Se3_obj_chained_long_jump = Se3_obj_long_jump_gt * Se3_obj_reference_frame

        Se3_obj_chained_short_jumps = np.prod(list(products))

        short_long_chain_ang_diff = quaternion_minimal_angular_difference(Se3_obj_chained_long_jump.quaternion,
                                                                          Se3_obj_chained_short_jumps.quaternion).item()

        print(f'-----------------------------------Long, short chain diff: {short_long_chain_ang_diff}')
        if short_long_chain_ang_diff > self.config.long_short_flow_chaining_pose_level_threshold \
                and self.config.long_short_flow_chaining_pose_level_verification:
            print(f'-----------------------------------Last long jump axis-angle '
                  f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_reference_frame.quaternion.q))}')
            print(f'-----------------------------------Chained long jump axis-angle '
                  f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_chained_long_jump.quaternion.q))}')
            print(f'-----------------------------------Chained short jumps axis-angle '
                  f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_chained_short_jumps.quaternion.q))}')

            prev_node_idx = frame_i - 1
            prev_node_observation = self.data_graph.get_camera_specific_frame_data(prev_node_idx).frame_observation
            prev_node_pose = self.data_graph.get_frame_data(prev_node_idx).predicted_object_se3_long_jump.quaternion
            self.pose_icosphere.insert_new_reference(prev_node_observation, prev_node_pose, prev_node_idx)

        self.encoder.quaternion_offsets[flow_long_jump_target] = Se3_obj_chained_long_jump.quaternion.q
        self.encoder.translation_offsets[flow_long_jump_target] = Se3_obj_chained_long_jump.translation

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(frame_i)
        datagraph_node.pose_estimation_time = duration
        datagraph_node.predicted_object_se3_long_jump = Se3_obj_chained_long_jump
        datagraph_node.predicted_object_se3_short_jump = Se3_obj_chained_short_jumps
        datagraph_node.predicted_object_se3_total = self.encoder.get_se3_at_frame_vectorized()[[frame_i]]
        datagraph_node.predicted_obj_long_short_chain_diff = short_long_chain_ang_diff

        datagraph_short_edge = self.data_graph.get_edge_observations(*flow_arc_short_jump)
        datagraph_long_edge = self.data_graph.get_edge_observations(*flow_arc_long_jump)

        datagraph_short_edge.predicted_obj_delta_se3 = Se3_obj_short_jump
        datagraph_short_edge.predicted_cam_delta_se3 = Se3_cam_short_jump
        datagraph_short_edge.predicted_cam_delta_se3_ransac = Se3_cam_short_jump_RANSAC

        datagraph_long_edge.predicted_obj_delta_se3 = Se3_obj_long_jump
        datagraph_long_edge.predicted_cam_delta_se3 = Se3_cam_long_jump
        datagraph_long_edge.predicted_cam_delta_se3_ransac = Se3_cam_long_jump_RANSAC

        datagraph_camera_node = self.data_graph.get_camera_specific_frame_data(frame_i)
        datagraph_camera_node.long_jump_source = flow_long_jump_source
        datagraph_camera_node.short_jump_source = flow_short_jump_source

    def get_relative_gt_rotation(self, flow_long_jump_source, flow_long_jump_target):
        ref_data = self.data_graph.get_frame_data(flow_long_jump_source)
        target_data = self.data_graph.get_frame_data(flow_long_jump_target)

        ref_rot = Quaternion.from_axis_angle(ref_data.gt_rot_axis_angle[None])
        ref_trans = ref_data.gt_translation[None]
        target_rot = Quaternion.from_axis_angle(target_data.gt_rot_axis_angle[None])
        target_trans = target_data.gt_translation[None]

        Se3_obj_ref_frame_gt = Se3(ref_rot, ref_trans)
        Se3_obj_target_frame_gt = Se3(target_rot, target_trans)

        Se3_obj_chained_long_jump = Se3_obj_target_frame_gt * Se3_obj_ref_frame_gt.inverse()
        return Se3_obj_chained_long_jump

    def estimate_pose_using_optical_flow(self, flow_observation_long_jump: FlowObservation, flow_arc,
                                         chained_flow_verification=None) -> Tuple[Se3, Se3]:

        K1 = K2 = self.camera.intrinsics[0, :3, :3].to(dtype=torch.float32)

        datagraph_edge_data = self.data_graph.get_edge_observations(flow_arc[0], flow_arc[1], Cameras.FRONTVIEW)
        gt_flow_observation: SyntheticFlowObservation = datagraph_edge_data.synthetic_flow_result.cuda()

        occlusion, segmentation = self.get_adjusted_occlusion_and_segmentation(flow_observation_long_jump)
        gt_occlusion, gt_segmentation = self.get_adjusted_occlusion_and_segmentation(gt_flow_observation)
        uncertainty = flow_observation_long_jump.observed_flow_uncertainty

        if self.config.ransac_use_gt_occlusions_and_segmentation:
            occlusion, segmentation = gt_occlusion, gt_segmentation

        flow = flow_observation_long_jump.cast_unit_coords_to_image_coords().observed_flow
        gt_flow = gt_flow_observation.cast_unit_coords_to_image_coords().observed_flow

        observed_segmentation_binary_mask = segmentation > float(self.segmentation_threshold)

        src_pts_yx, observed_visible_fg_points_mask = (
            get_not_occluded_foreground_points(occlusion, segmentation, self.occlusion_threshold,
                                               self.segmentation_threshold))
        _, gt_visible_fg_points_mask = (
            get_not_occluded_foreground_points(gt_occlusion, gt_segmentation, self.occlusion_threshold,
                                               self.segmentation_threshold))

        original_points_selected = src_pts_yx.shape[0]
        dst_pts_yx_chained_flow = None
        if chained_flow_verification is not None:
            chained_flow_image_coords = chained_flow_verification.cast_unit_coords_to_image_coords().observed_flow
            dst_pts_yx_chained_flow = source_to_target_coords_world_coord_system(src_pts_yx, chained_flow_image_coords)
            dst_pts_yx = source_to_target_coords_world_coord_system(src_pts_yx, flow)
            ok_pts_indices = (
                get_correct_correspondence_mask_world_system(chained_flow_image_coords, src_pts_yx, dst_pts_yx,
                                                             self.config.long_short_flow_chaining_pixel_level_threshold)
            )

            src_pts_yx = src_pts_yx[ok_pts_indices]

        remained_after_filtering = src_pts_yx.shape[0]
        remaining_ratio = remained_after_filtering / original_points_selected

        if self.config.ransac_sample_points:
            perm = torch.randperm(src_pts_yx.shape[0])
            src_pts_yx = src_pts_yx[perm[:self.config.ransac_sampled_points_number]]

        dst_pts_yx = source_to_target_coords_world_coord_system(src_pts_yx, flow)
        dst_pts_yx_gt_flow = source_to_target_coords_world_coord_system(src_pts_yx, gt_flow)

        confidences = None
        if self.config.ransac_confidences_from_occlusion:
            confidences = 1 - uncertainty[0, 0, 0, src_pts_yx[:, 0].to(torch.long), src_pts_yx[:, 1].to(torch.long)]

        confidences, dst_pts_yx, dst_pts_yx_gt_flow, src_pts_yx = self.augment_correspondences(src_pts_yx, dst_pts_yx,
                                                                                               dst_pts_yx_gt_flow,
                                                                                               confidences,
                                                                                               gt_flow)

        src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
        dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

        if self.config.ransac_inlier_filter == 'pnp_ransac':
            renderer: RenderingKaolin = self.rendering

            gt_rendering_result: RenderingResult = renderer.rendering_result_for_frame(self.gt_encoder, 0)
            point_map_xy = gt_rendering_result.rendered_face_camera_coords
            # observed_image_target = camera_observation.observed_image[:, 0]
            # depth_image = self.depth_anything.infer_depth_anything(observed_image_target)

            result = estimate_pose_using_PnP_solver(src_pts_xy, dst_pts_xy, K1, K2, point_map_xy)
            rot_cam_ransac, t_cam_ransac, inlier_mask, triangulated_points_ransac = result
        elif self.config.ransac_inlier_filter in ['magsac++', 'ransac', '8point', 'pygcransac']:
            result = filter_inliers_using_ransac(src_pts_xy, dst_pts_xy, K1, K2, self.image_width,
                                                 self.image_height, self.config.ransac_inlier_filter,
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

        Se3_cam_RANSAC = None
        if rot_cam_ransac is not None:
            Se3_cam_RANSAC = Se3(Quaternion.from_axis_angle(rot_cam_ransac[None]), t_cam_ransac)

        src_pts_xy_inliers = src_pts_xy[inlier_mask]
        dst_pts_xy_inliers = dst_pts_xy[inlier_mask]

        if self.config.ransac_inlier_pose_method == '8point':
            result = estimate_pose_using_8pt_algorithm(src_pts_xy_inliers, dst_pts_xy_inliers, K1, K2)
            r_cam, t_cam = result

        elif self.config.ransac_inlier_pose_method == 'zaragoza':
            # raise NotImplementedError('The intrinsics are wrong and return row of matrix rather than a correct value')
            r_cam, t_cam = estimate_pose_zaragoza(src_pts_xy_inliers, dst_pts_xy_inliers, self.camera.fx[0],
                                                  self.camera.fy[0], self.camera.cx[0], self.camera.cy[0])
        elif self.config.ransac_inlier_pose_method is None:
            if self.config.ransac_inlier_filter is None:
                raise ValueError("At least one of 'ransac_inlier_filter' or 'ransac_inlier_filter' must not be None.")
            r_cam = rot_cam_ransac
            t_cam = t_cam_ransac
        else:
            raise ValueError("Unknown inlier pose method")

        quat_cam = Quaternion.from_axis_angle(r_cam[None])
        Se3_cam = Se3(quat_cam, t_cam.squeeze()[None])

        common_inlier_indices = torch.nonzero(inlier_mask, as_tuple=True)
        outlier_indices = torch.nonzero(~inlier_mask, as_tuple=True)
        inlier_src_pts = src_pts_yx[common_inlier_indices]
        outlier_src_pts = src_pts_yx[outlier_indices]

        # max(1, x) avoids division by zero
        inlier_ratio = len(inlier_src_pts) / max(1, len(inlier_src_pts) + len(outlier_src_pts))

        data = self.data_graph.get_edge_observations(*flow_arc, camera=Cameras.FRONTVIEW)
        data.src_pts_yx = src_pts_yx.cpu()
        data.dst_pts_yx = dst_pts_yx.cpu()
        data.dst_pts_yx_gt = dst_pts_yx_gt_flow.cpu()
        data.observed_flow_segmentation = observed_segmentation_binary_mask.cpu()
        data.observed_visible_fg_points_mask = observed_visible_fg_points_mask.cpu()
        data.gt_visible_fg_points_mask = gt_visible_fg_points_mask.cpu()
        data.ransac_inliers = inlier_src_pts.cpu()
        data.ransac_outliers = outlier_src_pts.cpu()
        data.ransac_triangulated_points = triangulated_points_ransac.cpu()
        data.ransac_inliers_mask = inlier_mask.cpu()
        data.ransac_inlier_ratio = inlier_ratio
        data.remaining_pts_after_filtering = remaining_ratio
        data.dst_pts_yx_chained = dst_pts_yx_chained_flow.cpu() if dst_pts_yx_chained_flow is not None else None

        return Se3_cam, Se3_cam_RANSAC

    def get_adjusted_occlusion_and_segmentation(self, flow_observation_current_frame: BaseFlowObservation):

        segmentation = flow_observation_current_frame.observed_flow_segmentation
        occlusions = flow_observation_current_frame.observed_flow_occlusion

        # Apply erosion if the config flag is set
        if self.config.ransac_erode_segmentation:
            segmentation = erode_segment_mask2(5, flow_observation_current_frame.observed_flow_segmentation[0])[None]

        # Apply dilation if the config flag is set
        if self.config.ransac_dilate_occlusion:
            occlusions = dilate_mask(1, flow_observation_current_frame.observed_flow_occlusion)

        return occlusions, segmentation

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
        ok_pts_indices = (
            get_correct_correspondence_mask_world_system(gt_flow_image_coord, src_pts_yx, dst_pts_yx,
                                                         self.config.ransac_feed_only_inlier_flow_epe_threshold))
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

    @staticmethod
    def relative_scale_recovery(Se3_world_to_i: Se3, Se3_world_to_j: Se3, Se3_i_to_q: Se3, Se3_j_to_q: Se3) \
            -> Tuple[Se3, Se3]:

        # Extract rotation and translation components from the Se3 objects
        R_rho_i = Se3_world_to_i.r.matrix().squeeze()
        R_rho_j = Se3_world_to_j.r.matrix().squeeze()
        t_rho_i = Se3_world_to_i.translation.squeeze()
        t_rho_j = Se3_world_to_j.translation.squeeze()
        R_i = Se3_i_to_q.r.matrix().squeeze()
        R_j = Se3_j_to_q.r.matrix().squeeze()
        t_i = Se3_i_to_q.translation.squeeze()
        t_j = Se3_j_to_q.translation.squeeze()

        print("t_rho_i:", t_rho_i)
        print("t_rho_j:", t_rho_j)
        print("t_i:", t_i)
        print("t_j:", t_j)

        # A=[R_rho_i^T * t_i, −R_rho_j^T * t_j]
        A = torch.stack([R_rho_i @ t_i, -R_rho_j @ t_j], dim=-1)
        # A = torch.stack([R_rho_i.transpose(1, 0) @ R_i @ t_i, -R_rho_j.transpose(0, 1) @ R_j @ t_j], dim=-1)

        # delta_c = c_i - c_j
        delta_c = t_rho_i - t_rho_j

        # Solve A * λ = delta_c, where λ = [λ_i, λ_j]
        # Since torch.linalg.lstsq returns a result object, we extract the solution.
        result = torch.linalg.lstsq(A, delta_c)
        Lambda = result.solution

        Se3_i_to_q = Se3(Se3_i_to_q.quaternion, Se3_i_to_q.t * Lambda[0])
        Se3_j_to_q = Se3(Se3_j_to_q.quaternion, Se3_j_to_q.t * Lambda[1])
        print("Lambda:", Lambda)
        print("t_i rescaled:", Se3_i_to_q.translation.squeeze())
        print("t_j rescaled:", Se3_j_to_q.translation.squeeze())

        return Se3_i_to_q, Se3_j_to_q


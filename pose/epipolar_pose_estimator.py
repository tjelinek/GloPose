from typing import Tuple, List
from time import time

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera

from data_providers.flow_wrappers import RoMaFlowProvider
from auxiliary_scripts.math_utils import Se3_obj_from_epipolar_Se3_cam, quaternion_minimal_angular_difference, \
    Se3_epipolar_cam_from_Se3_obj
from data_structures.data_graph import DataGraph
from data_structures.datagraph_utils import get_relative_gt_obj_rotation
from data_structures.pose_icosphere import PoseIcosphere
from flow import get_correct_correspondence_mask_world_system, source_to_target_coords_world_coord_system
from data_structures.keyframe_buffer import FlowObservation, SyntheticFlowObservation, BaseFlowObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin, RenderingResult
from pose.essential_matrix_pose_estimation import (filter_inliers_using_ransac, estimate_pose_zaragoza,
                                                   estimate_pose_using_8pt_algorithm)
from pose.pnp_pose_estimation import estimate_pose_using_PnP_solver
from tracker_config import TrackerConfig
from utils import erode_segment_mask2, dilate_mask, get_not_occluded_foreground_points, tensor_index_to_coordinates_xy, \
    get_foreground_and_segment_mask


class EpipolarPoseEstimator:

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, rendering: RenderingKaolin, gt_encoder, encoder,
                 pose_icosphere, camera: PinholeCamera, roma_flow_provider):

        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.rendering: RenderingKaolin = rendering
        self.encoder: Encoder = encoder
        self.gt_encoder: Encoder = gt_encoder
        self.pose_icosphere: PoseIcosphere = pose_icosphere

        self.image_width: int = int(camera.width.item())
        self.image_height: int = int(camera.height.item())

        self.occlusion_threshold = self.config.occlusion_coef_threshold
        self.segmentation_threshold = self.config.segmentation_mask_threshold
        self.roma_flow_provider: RoMaFlowProvider = roma_flow_provider

        self.camera = camera

        self.i_can_recover_the_scale_myself = False

    def add_roma_match(self, source_image_frame, target_image_frame):

        source_image = self.data_graph.get_frame_data(source_image_frame).frame_observation.observed_image.cuda()
        target_image = self.data_graph.get_frame_data(target_image_frame).frame_observation.observed_image.cuda()

        flow_edge_data = self.data_graph.get_edge_observations(source_image_frame, target_image_frame)

        src_pts_xy, dst_pts_xy = self.roma_flow_provider.next_flow_roma_src_pts_xy(source_image, target_image,
                                                                                   self.image_height, self.image_width,
                                                                                   sample=10000)

        flow_edge_data.src_pts_xy_roma = src_pts_xy.cpu()
        flow_edge_data.dst_pts_xy_roma = dst_pts_xy.cpu()

    @torch.no_grad()
    def essential_matrix_preinitialization(self, keyframes):

        start_time = time()
        frame_i = max(keyframes)

        preceding_frame_node = self.data_graph.get_frame_data(frame_i - 1)

        reliable_flows = set()
        if preceding_frame_node.is_source_reliable and frame_i > 1:
            source = preceding_frame_node.long_jump_source
        elif frame_i > 1:

            best_source: int = 0
            best_source_reliability: float = 0.

            for node in self.pose_icosphere.reference_poses:
                source_node_idx = node.keyframe_idx_observed

                self.add_new_flow(source_node_idx, frame_i)
                self.add_roma_match(source_node_idx, frame_i)
                flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
                flow_reliability = self.flow_reliability(flow_edge_data.observed_flow)
                flow_edge_data.reliability_score = flow_reliability.item()

                if flow_reliability > best_source_reliability:
                    best_source = source_node_idx
                    best_source_reliability = flow_reliability
                    reliable_flows |= {source_node_idx}

            source = best_source

        else:
            source = 0

        flow_arc_long_jump = (source, frame_i)

        self.add_new_flow(source, frame_i)
        if self.data_graph.get_edge_observations(source, frame_i).src_pts_xy_roma is None:
            self.add_roma_match(source, frame_i)

        flow_arc_short_jump = (frame_i - 1, frame_i)

        long_jump_source, long_jump_target = flow_arc_long_jump
        short_jump_source, short_jump_target = flow_arc_short_jump

        flow_long_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_long_jump).
                                                        observed_flow).cuda()
        flow_short_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_short_jump).
                                                         observed_flow).cuda()

        short_jumps_data = [self.data_graph.get_edge_observations(i, i + 1)
                            for i in range(long_jump_source, frame_i - 1)]

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

        per_axis_scale_long_jump, scale_long_jump = self.recover_scale_with_gt(Se3_cam_long_jump, Se3_world_to_cam,
                                                                               long_jump_source, long_jump_target)
        per_axis_scale_short_jump, scale_short_jump = self.recover_scale_with_gt(Se3_cam_short_jump, Se3_world_to_cam,
                                                                                 short_jump_source, short_jump_target)

        if not self.i_can_recover_the_scale_myself:
            Se3_cam_long_jump = Se3(Se3_cam_long_jump.quaternion, Se3_cam_long_jump.t * scale_long_jump)
            Se3_cam_short_jump = Se3(Se3_cam_short_jump.quaternion, Se3_cam_short_jump.t * scale_short_jump)
        else:
            datagraph_short_node = self.data_graph.get_frame_data(flow_arc_short_jump[0])
            datagraph_long_node = self.data_graph.get_frame_data(flow_arc_long_jump[0])

            Se3_obj_short_jump = datagraph_short_node.predicted_object_se3_long_jump
            Se3_obj_long_jump = datagraph_long_node.predicted_object_se3_long_jump

            Se3_cam_long_jump_pose = Se3_epipolar_cam_from_Se3_obj(Se3_obj_long_jump, Se3_world_to_cam)
            Se3_cam_short_jump_pose = Se3_epipolar_cam_from_Se3_obj(Se3_obj_short_jump, Se3_world_to_cam)

            Se3_cam_long_jump, Se3_cam_short_jump = self.recover_scale_with_rays(Se3_cam_long_jump, Se3_cam_short_jump,
                                                                                 Se3_cam_long_jump_pose,
                                                                                 Se3_cam_short_jump_pose)

        Se3_obj_reference_frame = self.encoder.get_se3_at_frame_vectorized()[[long_jump_source]]
        Se3_obj_short_jump_ref_frame = self.encoder.get_se3_at_frame_vectorized()[[short_jump_source]]

        Se3_obj_long_jump = Se3_obj_from_epipolar_Se3_cam(Se3_cam_long_jump, Se3_world_to_cam)
        Se3_obj_short_jump = Se3_obj_from_epipolar_Se3_cam(Se3_cam_short_jump, Se3_world_to_cam)

        Se3_obj_chained_long_jump = Se3_obj_long_jump * Se3_obj_reference_frame

        products = reversed([Se3_obj_reference_frame] +
                            pred_short_deltas_se3 +
                            [Se3_obj_short_jump])

        if self.config.icosphere_use_gt_long_jumps:
            Se3_obj_long_jump_gt = get_relative_gt_obj_rotation(long_jump_source, long_jump_target, self.data_graph)
            Se3_obj_chained_long_jump = Se3_obj_long_jump_gt * Se3_obj_reference_frame

        Se3_obj_chained_short_jumps = np.prod(list(products))

        short_long_chain_ang_diff = quaternion_minimal_angular_difference(Se3_obj_chained_long_jump.quaternion,
                                                                          Se3_obj_chained_short_jumps.quaternion).item()

        self.encoder.quaternion_offsets[long_jump_target] = Se3_obj_chained_long_jump.quaternion.q
        self.encoder.translation_offsets[long_jump_target] = Se3_obj_chained_long_jump.translation

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
        datagraph_long_edge.camera_scale_estimated = scale_long_jump.item()
        datagraph_long_edge.camera_scale_per_axis_gt = per_axis_scale_long_jump

        datagraph_node.long_jump_source = long_jump_source
        datagraph_node.short_jump_source = short_jump_source

        flow_arc_observation: FlowObservation = datagraph_long_edge.observed_flow

        flow_reliability = self.flow_reliability(flow_arc_observation)
        datagraph_long_edge.reliability_score = flow_reliability.item()
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{flow_reliability}')
        if flow_reliability < self.config.flow_reliability_threshold:
            datagraph_node.is_source_reliable = False

            new_node_frame_idx = frame_i
            frame_data = self.data_graph.get_frame_data(new_node_frame_idx)
            pose: Se3 = Se3(Quaternion.from_axis_angle(frame_data.gt_rot_axis_angle[None]),
                            frame_data.gt_translation[None])
            cam_frame_data = self.data_graph.get_frame_data(new_node_frame_idx)
            self.pose_icosphere.insert_new_reference(cam_frame_data.frame_observation, pose, new_node_frame_idx)

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_flows)

    def flow_reliability(self, flow_arc_observation):
        segment = flow_arc_observation.observed_flow_segmentation
        occlusion = flow_arc_observation.observed_flow_occlusion
        not_occluded_binary_mask, segmentation_binary_mask, not_occluded_foreground_mask = (
            get_foreground_and_segment_mask(occlusion, segment, self.config.occlusion_coef_threshold,
                                            self.config.segmentation_mask_threshold))
        return not_occluded_foreground_mask.sum() / segmentation_binary_mask.sum()

    def add_new_flow(self, source_frame, target_frame):
        if (source_frame, target_frame) not in self.data_graph.G.edges:
            self.data_graph.add_new_arc(source_frame, target_frame)
            frame_node_observation = self.data_graph.get_frame_data(source_frame).frame_observation.cuda()
            current_frame_observation = self.data_graph.get_frame_data(target_frame).frame_observation.cuda()

            flow_obs = self.roma_flow_provider.next_flow_observation(frame_node_observation, current_frame_observation)
            self.data_graph.get_edge_observations(source_frame, target_frame).observed_flow = flow_obs.cpu()
            self.data_graph.get_edge_observations(source_frame, target_frame).synthetic_flow_result = flow_obs.cpu()

    def recover_scale_with_gt(self, Se3_cam1_to_cam2_est, Se3_world_to_cam, flow_source, flow_target):

        Se3_obj1_to_obj2_gt = get_relative_gt_obj_rotation(flow_source, flow_target, self.data_graph)
        Se3_cam1_to_cam2_gt = Se3_epipolar_cam_from_Se3_obj(Se3_obj1_to_obj2_gt, Se3_world_to_cam)

        per_axis_scale_factor = (Se3_cam1_to_cam2_gt.t / Se3_cam1_to_cam2_est.t).squeeze()

        d_cam_gt = torch.linalg.vector_norm(Se3_cam1_to_cam2_gt.t)
        d_cam_est = torch.linalg.vector_norm(Se3_cam1_to_cam2_est.t)

        aggregated_scale = d_cam_gt / d_cam_est

        return per_axis_scale_factor, aggregated_scale

    def estimate_pose_using_optical_flow(self, flow_observation_long_jump: FlowObservation, flow_arc,
                                         chained_flow_verification=None) -> Tuple[Se3, Se3]:

        K1 = K2 = self.camera.intrinsics[0, :3, :3].to(dtype=torch.float32)

        datagraph_edge_data = self.data_graph.get_edge_observations(flow_arc[0], flow_arc[1])
        gt_flow_observation: SyntheticFlowObservation = datagraph_edge_data.synthetic_flow_result.cuda()

        occlusion, segmentation = self.get_adjusted_occlusion_and_segmentation(flow_observation_long_jump)
        gt_occlusion, gt_segmentation = self.get_adjusted_occlusion_and_segmentation(gt_flow_observation)
        uncertainty = flow_observation_long_jump.observed_flow_uncertainty

        if self.config.ransac_use_gt_occlusions_and_segmentation:
            occlusion, segmentation = gt_occlusion, gt_segmentation

        assert flow_observation_long_jump.coordinate_system == 'unit'
        assert gt_flow_observation.coordinate_system == 'unit'

        flow = flow_observation_long_jump.cast_unit_coords_to_image_coords().observed_flow
        gt_flow = gt_flow_observation.cast_unit_coords_to_image_coords().observed_flow

        observed_segmentation_binary_mask = segmentation > float(self.segmentation_threshold)

        src_pts_yx, observed_visible_fg_points_mask = (
            get_not_occluded_foreground_points(occlusion, segmentation, self.occlusion_threshold,
                                               self.segmentation_threshold))

        src_pts_yx_all, dst_pts_yx_all = self.get_all_src_and_dst_pts(flow_observation_long_jump)

        _, gt_visible_fg_points_mask = (
            get_not_occluded_foreground_points(gt_occlusion, gt_segmentation, self.occlusion_threshold,
                                               self.segmentation_threshold))

        original_points_selected = src_pts_yx.shape[0]
        dst_pts_yx_chained_flow = None
        if chained_flow_verification is not None and False:
            chained_flow_image_coords = chained_flow_verification.cast_unit_coords_to_image_coords().observed_flow
            dst_pts_yx_chained_flow = source_to_target_coords_world_coord_system(src_pts_yx, chained_flow_image_coords)
            dst_pts_yx = source_to_target_coords_world_coord_system(src_pts_yx, flow)
            ok_pts_indices = (
                get_correct_correspondence_mask_world_system(chained_flow_image_coords, src_pts_yx, dst_pts_yx,
                                                             self.config.long_short_flow_chaining_pixel_level_threshold)
            )

            src_pts_yx = src_pts_yx[ok_pts_indices]

        remained_after_filtering = src_pts_yx.shape[0]
        remaining_ratio = remained_after_filtering / (original_points_selected + 1e-5)

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
            Se3_cam_RANSAC = Se3(Quaternion.from_axis_angle(rot_cam_ransac[None]), t_cam_ransac.T)

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
            r_cam = rot_cam_ransac[None]
            t_cam = t_cam_ransac[None]
        else:
            raise ValueError("Unknown inlier pose method")
        quat_cam = Quaternion.from_axis_angle(r_cam)
        Se3_cam = Se3(quat_cam, t_cam.squeeze(-1))

        common_inlier_indices = torch.nonzero(inlier_mask, as_tuple=True)
        outlier_indices = torch.nonzero(~inlier_mask, as_tuple=True)
        inlier_src_pts = src_pts_yx[common_inlier_indices]
        outlier_src_pts = src_pts_yx[outlier_indices]

        # max(1, x) avoids division by zero
        inlier_ratio = len(inlier_src_pts) / max(1, len(inlier_src_pts) + len(outlier_src_pts))

        data = self.data_graph.get_edge_observations(*flow_arc)
        data.src_pts_yx = src_pts_yx.cpu()
        data.dst_pts_yx = dst_pts_yx.cpu()
        data.src_pts_xy_roma = src_pts_yx_all.cpu()
        data.dst_pts_xy_roma = dst_pts_yx_all.cpu()
        data.dst_pts_yx_gt = dst_pts_yx_gt_flow.cpu()
        data.adjusted_segmentation = observed_segmentation_binary_mask.cpu()
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

    def get_all_src_and_dst_pts(self, flow_observation: FlowObservation):

        assert flow_observation.coordinate_system == 'unit'

        flow = flow_observation.cast_unit_coords_to_image_coords().observed_flow
        src_pts_yx_all, observed_visible_fg_points_mask = (
            get_not_occluded_foreground_points(flow_observation.observed_flow_occlusion,
                                               flow_observation.observed_flow_segmentation,
                                               self.occlusion_threshold,
                                               self.segmentation_threshold))

        dst_pts_yx_all = source_to_target_coords_world_coord_system(src_pts_yx_all, flow)

        return src_pts_yx_all, dst_pts_yx_all

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
    def recover_scale_with_rays(Se3_cam_i_to_cam2_unscaled: Se3, Se3_cam_j_to_cam2_unscaled: Se3,
                                Se3_cam_i: Se3, Se3_cam_j: Se3, i, j, q, datagraph: DataGraph, Se3_world_to_cam: Se3):

        # Se3_cam_i_to_cam2_gt = get_relative_gt_cam_rotation(i, q, Se3_world_to_cam, datagraph)
        # Se3_cam_i_to_cam2_unscaled = Se3(Se3_cam_i_to_cam2_gt.quaternion, torch.nn.functional.normalize(Se3_cam_i_to_cam2_gt.t))
        #
        # Se3_cam_j_to_cam2_gt = get_relative_gt_cam_rotation(j, q, Se3_world_to_cam, datagraph)
        # Se3_cam_j_to_cam2_unscaled = Se3(Se3_cam_j_to_cam2_gt.quaternion, torch.nn.functional.normalize(Se3_cam_j_to_cam2_gt.t))
        #
        # Se3_cam_i = get_gt_cam_pose(i, Se3_world_to_cam, datagraph)
        # Se3_cam_j = get_gt_cam_pose(j, Se3_world_to_cam, datagraph)
        # Se3_cam_q = get_gt_cam_pose(q, Se3_world_to_cam, datagraph)

        # c_q = Se3_cam_q.inverse().t.T

        c_i = Se3_cam_i.inverse().t.T
        c_j = Se3_cam_j.inverse().t.T

        R_i_inv = Se3_cam_i.quaternion.matrix().mT
        R_j_inv = Se3_cam_j.quaternion.matrix().mT

        R_iq = Se3_cam_i_to_cam2_unscaled.quaternion.matrix().mT
        R_jq = Se3_cam_j_to_cam2_unscaled.quaternion.matrix().mT

        t_iq = Se3_cam_i_to_cam2_unscaled.translation.T
        t_jq = Se3_cam_j_to_cam2_unscaled.translation.T

        # t_iq_scaled = Se3_cam_i_to_cam2_gt.translation.T
        # t_jq_scaled = Se3_cam_j_to_cam2_gt.translation.T

        ray_iq = (R_i_inv @ R_iq @ t_iq).squeeze(0)
        ray_jq = (R_j_inv @ R_jq @ t_jq).squeeze(0)

        A = torch.cat([ray_iq, -ray_jq], dim=1)
        b = c_i - c_j

        solution = torch.linalg.lstsq(A, b)
        lambdas = solution.solution
        lambda_i = lambdas[0]
        lambda_j = lambdas[1]

        # lambda_i_gt = (t_iq_scaled / t_iq).mean().squeeze()
        # lambda_j_gt = (t_jq_scaled / t_jq).mean().squeeze()

        # print(f'lambda i {lambda_i.mean().squeeze()}, gt {lambda_i_gt}')
        # print(f'lambda j {lambda_j.mean().squeeze()}, gt {lambda_j_gt}')

        # ray_iq_gt = (R_i_inv @ R_iq @ t_iq_scaled).squeeze(0)
        # ray_jq_gt = (R_j_inv @ R_jq @ t_jq_scaled).squeeze(0)

        ray_iq_scaled = ray_iq * lambda_i
        ray_jq_scaled = ray_jq * lambda_j

        # print(f'ray iq {ray_iq_scaled.squeeze().numpy(force=True)}, gt {ray_iq_gt.squeeze().numpy(force=True)}')
        # print(f'ray jq {ray_jq_scaled.squeeze().numpy(force=True)}, gt {ray_jq_gt.squeeze().numpy(force=True)}')

        camera_pos_q_iq = c_i - ray_iq_scaled
        camera_pos_q_jq = c_j - ray_jq_scaled
        # print(camera_pos_q_iq.squeeze().numpy(force=True))
        # print(camera_pos_q_jq.squeeze().numpy(force=True))
        # print(c_q.squeeze().numpy(force=True))

        Se3_cam_i_to_cam2_scaled = Se3(Se3_cam_i_to_cam2_unscaled.quaternion, Se3_cam_i_to_cam2_unscaled.t * lambda_i)
        Se3_cam_j_to_cam2_scaled = Se3(Se3_cam_j_to_cam2_unscaled.quaternion, Se3_cam_j_to_cam2_unscaled.t * lambda_j)

        # c_i_ray = (Se3_cam_i_to_cam2_scaled * Se3_cam_i).inverse().t
        # c_j_ray = (Se3_cam_j_to_cam2_scaled * Se3_cam_j).inverse().t

        # c_i_ray_gt = (Se3_cam_i_to_cam2_gt * Se3_cam_i).inverse().t
        # c_j_ray_gt = (Se3_cam_j_to_cam2_gt * Se3_cam_j).inverse().t
        # print('c_q', c_q.squeeze().numpy(force=True))
        # print('c_i', c_i_ray.squeeze().numpy(force=True), Se3_cam_i_to_cam2_scaled.quaternion.q.squeeze().numpy(force=True), Se3_cam_i_to_cam2_scaled.t.squeeze().numpy(force=True))
        # print('c_i gt', c_i_ray_gt.squeeze().numpy(force=True), Se3_cam_i_to_cam2_gt.quaternion.q.squeeze().numpy(force=True), Se3_cam_i_to_cam2_gt.t.squeeze().numpy(force=True))
        # print('c_j', c_j_ray.squeeze().numpy(force=True), Se3_cam_j_to_cam2_scaled.quaternion.q.squeeze().numpy(force=True), Se3_cam_j_to_cam2_scaled.t.squeeze().numpy(force=True))
        # print('c_j gt', c_j_ray_gt.squeeze().numpy(force=True), Se3_cam_j_to_cam2_gt.quaternion.q.squeeze().numpy(force=True), Se3_cam_j_to_cam2_gt.t.squeeze().numpy(force=True))

        return Se3_cam_i_to_cam2_scaled, Se3_cam_j_to_cam2_scaled

    @staticmethod
    def recover_scale(Se3_cam1_to_cam2_unscaled: Se3, Se3_world_to_cam: Se3):
        """
        ***************************************************************
        *                                      X                      *
        *                                     / \                     *
        *                             T_obj  / α \                    *
        *                            ,-'`-. /     \                   *
        *                         ,-'      `-.     \  T_w2c           *
        *                    Obj (      X     )     \                 *
        *                         `-.      ,-'       \                *
        *                            `-.,-'           \               *
        *                          /                   \              *
        *                  T_w2   /                     \             *
        *                        /                   ________         *
        *                       /                    \      /         *
        *                      /                      \    /          *
        *                _______                       \  /  C_2      *
        *                \     /                    γ   \/            *
        *                 \   /                 ________X             *
        *              C_1 \ / β      _________/                      *
        *                   X________/       T_C                      *
        *                                                             *
        ***************************************************************
        !!!!!! Works only if the underlying translation of T_o is zero
        """
        d_cam1_to_cam2_unscaled = torch.linalg.vector_norm(Se3_cam1_to_cam2_unscaled.translation)
        d_world_to_cam1_scaled = torch.linalg.vector_norm(Se3_world_to_cam.inverse().translation)

        cam1_to_obj1_ray = Se3_world_to_cam.translation
        cam1_to_cam2_ray = Se3_cam1_to_cam2_unscaled.inverse().translation

        beta = torch.nn.functional.cosine_similarity(cam1_to_obj1_ray, cam1_to_cam2_ray).acos()
        alpha = torch.pi - 2 * beta

        d_world_to_cam1_unscaled = d_cam1_to_cam2_unscaled / torch.sin(alpha) * torch.sin(beta)
        scale_factor = d_world_to_cam1_scaled / d_world_to_cam1_unscaled

        cam_t_scaled = Se3_cam1_to_cam2_unscaled.translation * scale_factor
        Se3_cam1_to_cam2_scaled = Se3(Se3_cam1_to_cam2_unscaled.quaternion, cam_t_scaled)

        return Se3_cam1_to_cam2_scaled, float(scale_factor)


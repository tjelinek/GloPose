import rerun as rr
import torch
from matplotlib import pyplot as plt

from data_structures.rerun_annotations import RerunAnnotations
from utils.image_utils import overlay_mask


def visualize_flow_with_matching_rerun(frame_i):

    datagraph_camera_data = self.data_graph.get_frame_data(frame_i)
    new_flow_arc = (datagraph_camera_data.matching_source_keyframe, frame_i)
    flow_arc_source, flow_arc_target = new_flow_arc

    if self.config.frame_filter == 'passthrough':
        return

    arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

    template_data = self.data_graph.get_frame_data(flow_arc_source)
    target_data = self.data_graph.get_frame_data(flow_arc_target)

    if self.config.frame_filter == 'dense_matching':
        reliability = arc_observation.reliability_score
        rr.log(RerunAnnotations.matching_reliability, rr.Scalar(reliability))
        rr.log(RerunAnnotations.matching_reliability_threshold_roma,
               rr.Scalar(target_data.current_flow_reliability_threshold))
        if self.config.matchability_based_reliability:
            matchability_share = template_data.relative_area_matchable
            min_roma_certainty = template_data.roma_certainty_threshold
            rr.log(RerunAnnotations.matching_matchability_plot_share_matchable, rr.Scalar(matchability_share))
            rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty, rr.Scalar(min_roma_certainty))
    elif self.config.frame_filter == 'SIFT':
        rr.log(RerunAnnotations.matches_sift, rr.Scalar(arc_observation.num_matches))
        rr.log(RerunAnnotations.min_matches_sift, rr.Scalar(self.config.sift_filter_min_matches))
        rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
               rr.Scalar(self.config.sift_filter_good_to_add_matches))

    if frame_i == 0 or (frame_i % self.config.large_images_results_write_frequency != 0):
        return

    template_image = template_data.frame_observation.observed_image.squeeze()
    target_image = target_data.frame_observation.observed_image.squeeze()

    template_target_image = torch.cat([template_image, target_image], dim=-2)
    template_target_image_np = template_target_image.permute(1, 2, 0).numpy(force=True)
    rerun_image = rr.Image(template_target_image_np)
    rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
    rr.log(RerunAnnotations.matches_low_certainty, rerun_image)

    if self.config.frame_filter == 'dense_matching':
        certainties = arc_observation.src_dst_certainty_roma.numpy(force=True)
        threshold = template_data.roma_certainty_threshold
        if threshold is None:
            threshold = self.config.min_roma_certainty_threshold

        above_threshold_mask = certainties >= threshold
        src_pts_xy_roma = arc_observation.src_pts_xy_roma[:, [1, 0]].numpy(force=True)
        dst_pts_xy_roma = arc_observation.dst_pts_xy_roma[:, [1, 0]].numpy(force=True)

        inliers_source_yx = src_pts_xy_roma[above_threshold_mask]
        inliers_target_yx = dst_pts_xy_roma[above_threshold_mask]
        outliers_source_yx = src_pts_xy_roma[~above_threshold_mask]
        outliers_target_yx = dst_pts_xy_roma[~above_threshold_mask]

        if self.config.matchability_based_reliability:
            certainties_matchable = arc_observation.src_dst_certainty_roma_matchable.numpy(force=True)
            above_threshold_mask_matchable = certainties_matchable >= threshold
            src_pts_xy_roma_matchable = arc_observation.src_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            dst_pts_xy_roma_matchable = arc_observation.dst_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            inliers_source_yx_matchable = src_pts_xy_roma_matchable[above_threshold_mask_matchable]
            inliers_target_yx_matchable = dst_pts_xy_roma_matchable[above_threshold_mask_matchable]
            outliers_source_yx_matchable = src_pts_xy_roma_matchable[~above_threshold_mask_matchable]
            outliers_target_yx_matchable = dst_pts_xy_roma_matchable[~above_threshold_mask_matchable]

        roma_certainty_map = arc_observation.roma_flow_warp_certainty
        roma_h, roma_w = roma_certainty_map.shape[0], roma_certainty_map.shape[1] // 2
        certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(roma_certainty_map.device)
        certainty_map_column[:roma_h, :roma_w] = roma_certainty_map[:roma_h, :roma_w]
        certainty_map_column[roma_h:, :roma_w] = roma_certainty_map[:roma_h, roma_w:]
        certainty_map_column = certainty_map_column[None]
        roma_certainty_map_image_size = (
            torchvision.transforms.functional.resize(certainty_map_column, size=template_target_image.shape[1:]))

        roma_certainty_map_im_size_np = roma_certainty_map_image_size.numpy(force=True)
        template_target_blacks = np.ones_like(template_target_image_np)
        template_target_image_certainty_np = overlay_mask(template_target_blacks, roma_certainty_map_im_size_np)

        rerun_certainty_img = rr.Image(template_target_image_certainty_np)
        rr.log(RerunAnnotations.matching_certainty, rerun_certainty_img)
    elif self.config.frame_filter == 'SIFT':
        keypoints_matching_indices = arc_observation.sift_keypoint_indices

        template_src_pts = template_data.sift_keypoints
        target_src_pts = target_data.sift_keypoints

        inliers_source_xy = template_src_pts[keypoints_matching_indices[:, 0]]
        inliers_target_xy = target_src_pts[keypoints_matching_indices[:, 1]]

        inliers_source_yx = inliers_source_xy[:, [1, 0]].numpy(force=True)
        inliers_target_yx = inliers_target_xy[:, [1, 0]].numpy(force=True)
        outliers_source_yx = np.zeros((0, 2))
        outliers_target_yx = np.zeros((0, 2))
    else:
        return

    def log_correspondences_rerun(cmap, src_yx, target_yx, rerun_annotation, source_image_height, sample_size=None):
        if sample_size is not None:
            random_indices = torch.randperm(min(sample_size, src_yx.shape[0]))
            src_yx = src_yx[random_indices]
            target_yx = target_yx[random_indices]

        if len(src_yx.shape) == 1 or src_yx.shape[1] == 0:
            return  # No matches to draw
        target_yx_2nd_image = target_yx
        target_yx_2nd_image[:, 0] = source_image_height + target_yx_2nd_image[:, 0]

        line_strips_xy = np.stack([src_yx[:, [1, 0]], target_yx_2nd_image[:, [1, 0]]], axis=1)

        num_points = line_strips_xy.shape[0]
        colors = [cmap(i / num_points)[:3] for i in range(num_points)]
        colors = (np.array(colors) * 255).astype(int).tolist()

        rr.log(
            rerun_annotation,
            rr.LineStrips2D(
                strips=line_strips_xy,
                colors=colors,
            ),
        )

    template_image_size = template_data.image_shape
    cmap_inliers = plt.get_cmap('Greens')
    log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                              RerunAnnotations.matches_high_certainty, template_image_size.height, 20)
    cmap_outliers = plt.get_cmap('Reds')
    log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                              RerunAnnotations.matches_low_certainty, template_image_size.height, 20)

    if self.config.matchability_based_reliability and self.config.frame_filter == 'dense_matching':
        log_correspondences_rerun(cmap_inliers, inliers_source_yx_matchable, inliers_target_yx_matchable,
                                  RerunAnnotations.matches_high_certainty_matchable, template_image_size.height, 20)
        log_correspondences_rerun(cmap_outliers, outliers_source_yx_matchable, outliers_target_yx_matchable,
                                  RerunAnnotations.matches_low_certainty_matchable, template_image_size.height, 20)

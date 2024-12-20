from typing import Final

axes = ['x', 'y', 'z']


class RerunAnnotations:
    # Observations
    space_visualization: Final[str] = '/3d_space'
    colmap_visualization: Final[str] = '/3d_colmap'
    space_gt_mesh: Final[str] = '/3d_space/gt_mesh'
    space_gt_camera_pose: Final[str] = '/3d_space/gt_camera_pose'
    space_predicted_camera_pose: Final[str] = '/3d_space/predicted_camera_pose'
    space_gt_camera_track: Final[str] = '/3d_space/gt_camera_track'
    space_predicted_camera_track: Final[str] = '/3d_space/predicted_camera_track'
    space_predicted_camera_keypoints: Final[str] = '/3d_space/predicted_camera_keypoints'
    space_predicted_closest_keypoint: Final[str] = '/3d_space/predicted_closest_keypoint'
    space_predicted_reliable_templates: Final[str] = '/3d_space/predicted_reliable_templates'

    # COLMAP
    colmap_pointcloud: Final[str] = '/3d_colmap/pointcloud'
    colmap_gt_camera_pose: Final[str] = '/3d_colmap/gt_camera_pose'
    colmap_gt_camera_track: Final[str] = '/3d_colmap/gt_camera_track'
    colmap_predicted_camera_poses: Final[str] = '/3d_colmap/predicted_camera_keypoints'
    colmap_predicted_reliable_templates: Final[str] = '/3d_colmap/predicted_reliable_templates'
    colmap_predicted_line_strips_reliable: Final[str] = '/3d_colmap/line_strips_reliable'

    template_image_frontview: Final[str] = '/observations/template_image_frontview'

    template_image_segmentation_frontview: Final[str] = '/observations/template_image_frontview/segment'

    observed_image_frontview: Final[str] = '/observations/observed_image_frontview'

    observed_image_segmentation_frontview: Final[str] = '/observations/observed_image_frontview/segment'

    observed_flow_frontview: Final[str] = '/observed_flow/observed_flow_frontview'
    observed_flow_occlusion_frontview: Final[str] = '/observed_flow/occlusion_frontview'
    observed_flow_uncertainty_frontview: Final[str] = '/observed_flow/uncertainty_frontview'
    observed_flow_with_uncertainty_frontview: Final[str] = '/observed_flow/observed_flow_front_uncertainty'
    observed_flow_errors_frontview: Final[str] = '/observed_flow/observed_flow_gt_disparity'

    # Optimized model visualizations
    optimized_model_occlusion: Final[str] = '/optimized_values/occlusion'
    optimized_model_render: Final[str] = '/optimized_values/rendering'

    # Triangulated points RANSAC
    triangulated_points_gt_Rt_gt_flow: Final[str] = '/point_clouds/triangulated_points_gt_Rt_gt_flow'
    triangulated_points_gt_Rt_mft_flow: Final[str] = '/point_clouds/triangulated_points_gt_Rt_mft_flow'
    point_cloud_dust3r_im1: Final[str] = '/point_clouds/point_cloud_dust3r_im1'
    point_cloud_dust3r_im2: Final[str] = '/point_clouds/point_cloud_dust3r_im2'

    # Ransac
    matching_correspondences_inliers: Final[str] = '/epipolar/matching/correspondences_inliers'
    matching_correspondences_outliers: Final[str] = '/epipolar/matching/correspondences_outliers'

    ransac_stats: Final[str] = '/epipolar/ransac_stats'
    ransac_stats_visible: Final[str] = '/epipolar/ransac_stats/visible'
    ransac_stats_predicted_as_visible: Final[str] = '/epipolar/ransac_stats/predicted_as_visible'
    ransac_stats_correctly_predicted_flows: Final[str] = '/epipolar/ransac_stats/correctly_predicted_flows'
    ransac_stats_ransac_predicted_inliers: Final[str] = '/epipolar/ransac_stats/ransac_predicted_inliers'
    ransac_stats_correctly_predicted_inliers: Final[str] = '/epipolar/ransac_stats/correctly_predicted_inliers'
    ransac_stats_ransac_inlier_ratio: Final[str] = '/epipolar/ransac_stats/ransac_inlier_ratio'

    # Pose
    pose_estimation_timing: Final[str] = '/pose/timing/'
    pose_estimation_time: Final[str] = '/pose/timing/pose_estimation_time'
    long_short_chain_remaining_pts: Final[str] = '/pose/chaining/remaining_pts/remaining_percent'
    long_short_chain_diff_template: Final[str] = '/pose/chaining/template'
    chained_pose_polar: Final[str] = '/pose/chaining/polar_angle'
    chained_pose_polar_template: Final[str] = '/pose/chaining/polar_angle/template'
    chained_pose_long_flow_polar: Final[str] = '/pose/chaining/polar_angle/long_flow'
    chained_pose_short_flow_polar: Final[str] = '/pose/chaining/polar_angle/short_flow'
    long_short_chain_diff: Final[str] = '/pose/chaining/polar_angle/difference'

    chained_pose_long_flow: Final[str] = '/pose/chaining/long_flow'
    chained_pose_long_flow_template: Final[str] = '/pose/chaining/long_flow/template'
    chained_pose_long_flow_axes: Final[dict] = {
        axis: f'/pose/chaining/long_flow/{axis}_axis' for axis in axes
    }

    chained_pose_short_flow: Final[str] = '/pose/chaining/short_flow'
    chained_pose_short_flow_template: Final[str] = '/pose/chaining/short_flow/template'
    chained_pose_short_flow_axes: Final[dict] = {
        axis: f'/pose/chaining/short_flow/{axis}_axis' for axis in axes
    }

    cam_delta_r_short_flow: Final[str] = '/pose/cam/rot/short_flow/'
    cam_delta_r_short_flow_template = '/pose/cam/rot/short_flow/template'
    cam_delta_r_short_flow_zaragoza: Final[str] = '/pose/cam/rot/short_flow/cam_pose_delta_zaragoza'
    cam_delta_r_short_flow_RANSAC: Final[str] = '/pose/cam/rot/short_flow/cam_pose_delta_RANSAC'

    cam_delta_t_short_flow: Final[str] = '/pose/cam/tran/short_flow/'
    cam_delta_t_short_flow_template = '/pose/cam/tran/short_flow/template'
    cam_delta_t_short_flow_zaragoza: Final[str] = '/pose/cam/tran/short_flow/cam_pose_delta_zaragoza'
    cam_delta_t_short_flow_RANSAC: Final[str] = '/pose/cam/tran/short_flow/cam_pose_delta_RANSAC'

    cam_delta_r_long_flow: Final[str] = '/pose/cam/rot/long_flow/'
    cam_delta_r_long_flow_template = '/pose/cam/rot/long_flow/template'
    cam_delta_r_long_flow_zaragoza: Final[str] = '/pose/cam/rot/long_flow/cam_pose_delta_zaragoza'
    cam_delta_r_long_flow_RANSAC: Final[str] = '/pose/cam/rot/long_flow/cam_pose_delta_RANSAC'

    cam_delta_t_long_flow: Final[str] = '/pose/cam/tran/long_flow/'
    cam_delta_t_long_flow_template = '/pose/cam/tran/long_flow/template'
    cam_delta_t_long_flow_zaragoza: Final[str] = '/pose/cam/tran/long_flow/cam_pose_delta_zaragoza'
    cam_delta_t_long_flow_RANSAC: Final[str] = '/pose/cam/tran/long_flow/cam_pose_delta_RANSAC'

    obj_rot_1st_to_last: Final[str] = '/pose/rotation'
    obj_rot_1st_to_last_axes: Final[dict] = {
        axis: f'/pose/rotation/{axis}_axis' for axis in axes
    }
    obj_rot_1st_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/rotation/{axis}_axis_gt' for axis in axes
    }

    obj_tran_1st_to_last: Final[str] = '/pose/translation'
    obj_tran_1st_to_last_axes: Final[dict] = {
        axis: f'/pose/translation/{axis}_axis' for axis in axes
    }
    obj_tran_1st_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/translation/{axis}_axis_gt' for axis in axes
    }

    cam_rot_ref_to_last: Final[str] = '/pose/cam_rot_ref_to_last'
    cam_rot_ref_to_last_template: Final[str] = '/pose/cam_rot_ref_to_last/template'
    cam_rot_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/cam_rot_ref_to_last/{axis}_axis' for axis in axes
    }
    cam_rot_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/cam_rot_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    cam_tran_ref_to_last: Final[str] = '/pose/cam_tran_ref_to_last'
    cam_tran_ref_to_last_template: Final[str] = '/pose/cam_tran_ref_to_last/template'
    cam_tran_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/cam_tran_ref_to_last/{axis}_axis' for axis in axes
    }
    cam_tran_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/cam_tran_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    obj_rot_ref_to_last: Final[str] = '/pose/obj_rot_ref_to_last'
    obj_rot_ref_to_last_template: Final[str] = '/pose/obj_rot_ref_to_last/template'
    obj_rot_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/obj_rot_ref_to_last/{axis}_axis' for axis in axes
    }
    obj_rot_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/obj_rot_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    obj_tran_ref_to_last: Final[str] = '/pose/obj_tran_ref_to_last'
    obj_tran_ref_to_last_template: Final[str] = '/pose/obj_tran_ref_to_last/template'
    obj_tran_ref_to_last_axes: Final[dict] = {
        axis: f'/pose/obj_tran_ref_to_last/{axis}_axis' for axis in axes
    }
    obj_tran_ref_to_last_gt_axes: Final[dict] = {
        axis: f'/pose/obj_tran_ref_to_last/{axis}_axis_gt' for axis in axes
    }

    translation_scale: Final[str] = '/pose/translation_scale'
    translation_scale_gt_axes: Final[dict] = {
        axis: f'/pose/translation_scale/{axis}_gt' for axis in axes
    }
    translation_scale_estimated: Final[str] = '/pose/translation_scale/estimated'

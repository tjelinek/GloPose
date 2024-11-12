from dataclasses import dataclass


@dataclass
class RerunAnnotations:
    # Observations
    space_visualization: str = '/3d_space'
    space_gt_mesh: str = '/3d_space/gt_mesh'
    space_gt_camera_pose: str = '/3d_space/gt_camera_pose'
    space_predicted_camera_pose: str = '/3d_space/predicted_camera_pose'
    space_gt_camera_track: str = '/3d_space/gt_camera_track'
    space_predicted_camera_track: str = '/3d_space/predicted_camera_track'
    space_predicted_camera_keypoints: str = '/3d_space/predicted_camera_keypoints'
    space_predicted_closest_keypoint: str = '/3d_space/predicted_closest_keypoint'
    space_predicted_reliable_templates: str = '/3d_space/predicted_reliable_templates'

    template_image_frontview: str = '/observations/template_image_frontview'

    template_image_segmentation_frontview: str = '/observations/template_image_frontview/segment'

    observed_image_frontview: str = '/observations/observed_image_frontview'

    observed_image_segmentation_frontview: str = '/observations/observed_image_frontview/segment'

    observed_flow_frontview: str = '/observed_flow/observed_flow_frontview'
    observed_flow_occlusion_frontview: str = '/observed_flow/occlusion_frontview'
    observed_flow_uncertainty_frontview: str = '/observed_flow/uncertainty_frontview'
    observed_flow_with_uncertainty_frontview: str = '/observed_flow/observed_flow_front_uncertainty'
    observed_flow_errors_frontview: str = '/observed_flow/observed_flow_gt_disparity'

    # Optimized model visualizations
    optimized_model_occlusion: str = '/optimized_values/occlusion'
    optimized_model_render: str = '/optimized_values/rendering'

    # Triangulated points RANSAC
    triangulated_points_gt_Rt_gt_flow: str = '/point_clouds/triangulated_points_gt_Rt_gt_flow'
    triangulated_points_gt_Rt_mft_flow: str = '/point_clouds/triangulated_points_gt_Rt_mft_flow'
    point_cloud_dust3r_im1: str = '/point_clouds/point_cloud_dust3r_im1'
    point_cloud_dust3r_im2: str = '/point_clouds/point_cloud_dust3r_im2'

    # Ransac
    matching_correspondences_inliers: str = '/epipolar/matching/correspondences_inliers'
    matching_correspondences_outliers: str = '/epipolar/matching/correspondences_outliers'

    ransac_stats: str = '/epipolar/ransac_stats'
    ransac_stats_visible: str = '/epipolar/ransac_stats/visible'
    ransac_stats_predicted_as_visible: str = '/epipolar/ransac_stats/predicted_as_visible'
    ransac_stats_correctly_predicted_flows: str = '/epipolar/ransac_stats/correctly_predicted_flows'
    ransac_stats_ransac_predicted_inliers: str = '/epipolar/ransac_stats/ransac_predicted_inliers'
    ransac_stats_correctly_predicted_inliers: str = '/epipolar/ransac_stats/correctly_predicted_inliers'
    ransac_stats_ransac_inlier_ratio: str = '/epipolar/ransac_stats/ransac_inlier_ratio'

    # Pose
    pose_estimation_timing: str = '/pose/timing/'
    pose_estimation_time: str = '/pose/timing/pose_estimation_time'
    long_short_chain_remaining_pts: str = '/pose/chaining/remaining_pts/remaining_percent'
    long_short_chain_diff_template: str = '/pose/chaining/template'
    chained_pose_polar: str = '/pose/chaining/polar_angle'
    chained_pose_polar_template: str = '/pose/chaining/polar_angle/template'
    chained_pose_long_flow_polar: str = '/pose/chaining/polar_angle/long_flow'
    chained_pose_short_flow_polar: str = '/pose/chaining/polar_angle/short_flow'
    long_short_chain_diff: str = '/pose/chaining/polar_angle/difference'

    chained_pose_long_flow: str = '/pose/chaining/long_flow'
    chained_pose_long_flow_template: str = '/pose/chaining/long_flow/template'
    chained_pose_long_flow_x: str = '/pose/chaining/long_flow/x_axis'
    chained_pose_long_flow_y: str = '/pose/chaining/long_flow/y_axis'
    chained_pose_long_flow_z: str = '/pose/chaining/long_flow/z_axis'

    chained_pose_short_flow: str = '/pose/chaining/short_flow'
    chained_pose_short_flow_template: str = '/pose/chaining/short_flow/template'
    chained_pose_short_flow_x: str = '/pose/chaining/short_flow/x_axis'
    chained_pose_short_flow_y: str = '/pose/chaining/short_flow/y_axis'
    chained_pose_short_flow_z: str = '/pose/chaining/short_flow/z_axis'

    cam_delta_r_short_flow: str = '/pose/cam/rot/short_flow/'
    cam_delta_r_short_flow_template = '/pose/cam/rot/short_flow/template'
    cam_delta_r_short_flow_zaragoza: str = '/pose/cam/rot/short_flow/cam_pose_delta_zaragoza'
    cam_delta_r_short_flow_RANSAC: str = '/pose/cam/rot/short_flow/cam_pose_delta_RANSAC'

    cam_delta_t_short_flow: str = '/pose/cam/tran/short_flow/'
    cam_delta_t_short_flow_template = '/pose/cam/tran/short_flow/template'
    cam_delta_t_short_flow_zaragoza: str = '/pose/cam/tran/short_flow/cam_pose_delta_zaragoza'
    cam_delta_t_short_flow_RANSAC: str = '/pose/cam/tran/short_flow/cam_pose_delta_RANSAC'

    cam_delta_r_long_flow: str = '/pose/cam/rot/long_flow/'
    cam_delta_r_long_flow_template = '/pose/cam/rot/long_flow/template'
    cam_delta_r_long_flow_zaragoza: str = '/pose/cam/rot/long_flow/cam_pose_delta_zaragoza'
    cam_delta_r_long_flow_RANSAC: str = '/pose/cam/rot/long_flow/cam_pose_delta_RANSAC'

    cam_delta_t_long_flow: str = '/pose/cam/tran/long_flow/'
    cam_delta_t_long_flow_template = '/pose/cam/tran/long_flow/template'
    cam_delta_t_long_flow_zaragoza: str = '/pose/cam/tran/long_flow/cam_pose_delta_zaragoza'
    cam_delta_t_long_flow_RANSAC: str = '/pose/cam/tran/long_flow/cam_pose_delta_RANSAC'

    obj_rot_1st_to_last: str = '/pose/rotation'
    obj_rot_1st_to_last_x: str = '/pose/rotation/x_axis'
    obj_rot_1st_to_last_x_gt: str = '/pose/rotation/x_axis_gt'
    obj_rot_1st_to_last_y: str = '/pose/rotation/y_axis'
    obj_rot_1st_to_last_y_gt: str = '/pose/rotation/y_axis_gt'
    obj_rot_1st_to_last_z: str = '/pose/rotation/z_axis'
    obj_rot_1st_to_last_z_gt: str = '/pose/rotation/z_axis_gt'

    obj_tran_1st_to_last: str = '/pose/translation'
    obj_tran_1st_to_last_x: str = '/pose/translation/x_axis'
    obj_tran_1st_to_last_x_gt: str = '/pose/translation/x_axis_gt'
    obj_tran_1st_to_last_y: str = '/pose/translation/y_axis'
    obj_tran_1st_to_last_y_gt: str = '/pose/translation/y_axis_gt'
    obj_tran_1st_to_last_z: str = '/pose/translation/z_axis'
    obj_tran_1st_to_last_z_gt: str = '/pose/translation/z_axis_gt'

    cam_rot_ref_to_last: str = '/pose/cam_rot_ref_to_last'
    cam_rot_ref_to_last_template: str = '/pose/cam_rot_ref_to_last/template'
    cam_rot_ref_to_last_x: str = '/pose/cam_rot_ref_to_last/x_axis'
    cam_rot_ref_to_last_x_gt: str = '/pose/cam_rot_ref_to_last/x_axis_gt'
    cam_rot_ref_to_last_y: str = '/pose/cam_rot_ref_to_last/y_axis'
    cam_rot_ref_to_last_y_gt: str = '/pose/cam_rot_ref_to_last/y_axis_gt'
    cam_rot_ref_to_last_z: str = '/pose/cam_rot_ref_to_last/z_axis'
    cam_rot_ref_to_last_z_gt: str = '/pose/cam_rot_ref_to_last/z_axis_gt'

    cam_tran_ref_to_last: str = '/pose/cam_tran_ref_to_last'
    cam_tran_ref_to_last_template: str = '/pose/cam_tran_ref_to_last/template'
    cam_tran_ref_to_last_x: str = '/pose/cam_tran_ref_to_last/x_axis'
    cam_tran_ref_to_last_x_gt: str = '/pose/cam_tran_ref_to_last/x_axis_gt'
    cam_tran_ref_to_last_y: str = '/pose/cam_tran_ref_to_last/y_axis'
    cam_tran_ref_to_last_y_gt: str = '/pose/cam_tran_ref_to_last/y_axis_gt'
    cam_tran_ref_to_last_z: str = '/pose/cam_tran_ref_to_last/z_axis'
    cam_tran_ref_to_last_z_gt: str = '/pose/cam_tran_ref_to_last/z_axis_gt'

    obj_rot_ref_to_last: str = '/pose/obj_rot_ref_to_last'
    obj_rot_ref_to_last_template: str = '/pose/obj_rot_ref_to_last/template'
    obj_rot_ref_to_last_x: str = '/pose/obj_rot_ref_to_last/x_axis'
    obj_rot_ref_to_last_x_gt: str = '/pose/obj_rot_ref_to_last/x_axis_gt'
    obj_rot_ref_to_last_y: str = '/pose/obj_rot_ref_to_last/y_axis'
    obj_rot_ref_to_last_y_gt: str = '/pose/obj_rot_ref_to_last/y_axis_gt'
    obj_rot_ref_to_last_z: str = '/pose/obj_rot_ref_to_last/z_axis'
    obj_rot_ref_to_last_z_gt: str = '/pose/obj_rot_ref_to_last/z_axis_gt'

    obj_tran_ref_to_last: str = '/pose/obj_tran_ref_to_last'
    obj_tran_ref_to_last_template: str = '/pose/obj_tran_ref_to_last/template'
    obj_tran_ref_to_last_x: str = '/pose/obj_tran_ref_to_last/x_axis'
    obj_tran_ref_to_last_x_gt: str = '/pose/obj_tran_ref_to_last/x_axis_gt'
    obj_tran_ref_to_last_y: str = '/pose/obj_tran_ref_to_last/y_axis'
    obj_tran_ref_to_last_y_gt: str = '/pose/obj_tran_ref_to_last/y_axis_gt'
    obj_tran_ref_to_last_z: str = '/pose/obj_tran_ref_to_last/z_axis'
    obj_tran_ref_to_last_z_gt: str = '/pose/obj_tran_ref_to_last/z_axis_gt'

    translation_scale: str = '/pose/translation_scale'
    translation_scale_x_gt: str = '/pose/translation_scale/x_gt'
    translation_scale_y_gt: str = '/pose/translation_scale/y_gt'
    translation_scale_z_gt: str = '/pose/translation_scale/z_gt'
    translation_scale_estimated: str = '/pose/translation_scale/estimated'

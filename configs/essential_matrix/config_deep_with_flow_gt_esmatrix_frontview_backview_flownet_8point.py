from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.run_main_optimization_loop = False
    cfg.optimize_shape = False

    cfg.ransac_essential_matrix_algorithm = None

    cfg.ransac_use_gt_occlusions_and_segmentation = False
    cfg.ransac_erode_segmentation_dilate_occlusion = False

    cfg.ransac_feed_only_inlier_flow = False
    cfg.ransac_replace_mft_flow_with_gt_flow = False
    cfg.ransac_feed_gt_flow_add_gaussian_noise = False
    cfg.ransac_feed_gt_flow_add_gaussian_noise_use_mft_errors = False
    cfg.ransac_distant_pixels_sampling = False
    cfg.ransac_confidences_from_occlusion = False
    cfg.ransac_inlier_pose_method = '8point'

    cfg.refine_pose_using_numerical_optimization = False

    return cfg

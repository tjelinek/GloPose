from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.run_main_optimization_loop = False
    cfg.optimize_shape = False

    cfg.ransac_inlier_filter = None

    cfg.ransac_use_gt_occlusions_and_segmentation = False
    cfg.ransac_dilate_occlusion = False
    cfg.ransac_erode_segmentation = True

    cfg.ransac_feed_only_inlier_flow = False
    cfg.ransac_replace_mft_flow_with_gt_flow = False
    cfg.ransac_feed_gt_flow_add_gaussian_noise = False
    cfg.ransac_feed_gt_flow_add_gaussian_noise_use_mft_errors = False
    cfg.ransac_distant_pixels_sampling = False
    cfg.ransac_confidences_from_occlusion = False
    cfg.ransac_inlier_pose_method = 'zaragoza'

    cfg.ransac_refine_E_numerically = False

    cfg.long_short_flow_chaining_pose_level_verification = True
    cfg.long_short_flow_chaining_pixel_level_verification = True

    cfg.ransac_inlier_filter = 'pygcransac'
    cfg.long_flow_model = 'MFT'
    cfg.MFT_backbone_cfg = 'MFT_RoMa_direct_cfg'

    return cfg

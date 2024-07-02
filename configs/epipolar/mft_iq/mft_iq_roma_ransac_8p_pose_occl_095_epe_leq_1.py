from configs.epipolar.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.relative_camera_pose_algorithm = 'RANSAC_2D_to_2D_E_solver'
    cfg.ransac_essential_matrix_algorithm = 'pygcransac'
    cfg.long_flow_model = 'MFT_IQ'
    cfg.MFT_backbone_cfg = 'MFTIQ_ROMA_bs3_bce_200k_kubric_binary_cfg'

    cfg.occlusion_coef_threshold = 0.95

    cfg.ransac_feed_only_inlier_flow = True
    cfg.ransac_feed_only_inlier_flow_epe_threshold = 1.0

    return cfg

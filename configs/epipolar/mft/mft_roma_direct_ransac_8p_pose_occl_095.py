from configs.epipolar.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.ransac_essential_matrix_algorithm = 'pygcransac'
    cfg.long_flow_model = 'MFT'
    cfg.MFT_backbone_cfg = 'RoMa_thresholds/MFT_RoMa_direct_cfg_occl_095'

    cfg.occlusion_coef_threshold = 0.95
    return cfg

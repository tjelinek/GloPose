from configs.epipolar.ransacs.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.ransac_essential_matrix_algorithm = 'pygcransac'
    cfg.long_flow_model = 'MFT'
    cfg.MFT_backbone_cfg = 'MFT_RoMa_cfg'

    return cfg

from configs.epipolar.ransacs.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.long_flow_model = 'MFT_IQ'
    cfg.MFT_backbone_cfg = 'MFTIQ_ROMA_bs3_bce_200k_kubric_binary_cfg'

    return cfg

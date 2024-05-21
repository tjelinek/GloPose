from configs.epipolar.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.long_flow_model = 'MFT'
    cfg.MFT_backbone_cfg = 'MFT_cfg'

    return cfg

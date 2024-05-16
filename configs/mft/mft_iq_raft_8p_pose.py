from configs.essential_matrix.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.long_flow_model = 'MFT_IQ'
    cfg.MFT_IQ_backbone_cfg = 'MFT_cfg'

    return cfg

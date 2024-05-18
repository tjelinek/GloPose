from configs.essential_matrix.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.long_flow_model = 'MFT_Synth'
    cfg.MFT_IQ_backbone_cfg = 'MFTIQ_SYNTHETIC_bs3_bce_200k_kubric_binary_cfg'
    cfg.MFT_synth_add_noise = True

    return cfg

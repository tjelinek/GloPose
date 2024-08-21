from configs.epipolar.ransacs.config_deep_with_flow_gt_esmatrix_frontview_backview_flownet_8point import TrackerConfig8P


def get_config() -> TrackerConfig8P:
    cfg = TrackerConfig8P()

    cfg.long_flow_model = 'MFT_Synth'
    cfg.MFT_backbone_cfg = 'MFTIQ_SYNTHETIC_bs3_bce_200k_kubric_binary_cfg'
    cfg.MFT_synth_add_noise = True
    cfg.MFT_synth_noise_sigma = 0.05
    cfg.MFT_synth_noise_mu = 0.

    return cfg

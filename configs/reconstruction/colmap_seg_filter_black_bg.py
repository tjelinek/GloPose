from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.min_certainty_threshold = 0.975
    cfg.onboarding.flow_reliability_threshold = 0.5
    cfg.onboarding.frame_filter = 'dense_matching'
    cfg.onboarding.filter_matcher = 'UFM'
    cfg.onboarding.reconstruction_method = 'colmap'
    cfg.onboarding.filter_points_by_segmentation = True
    cfg.input.frame_provider_config.background_color = 'black'

    return cfg

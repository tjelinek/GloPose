from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.min_certainty_threshold = 0.975
    cfg.onboarding.flow_reliability_threshold = 0.5
    cfg.onboarding.frame_filter = 'RANSAC'
    cfg.onboarding.filter_matcher = 'UFM'

    cfg.onboarding.ransac.method = 'pycolmap'
    cfg.onboarding.ransac.max_error = 0.5
    cfg.onboarding.ransac.confidence = 0.9999
    cfg.onboarding.ransac.min_inlier_ratio = 0.1

    return cfg

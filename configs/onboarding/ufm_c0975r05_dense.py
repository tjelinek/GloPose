from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.min_certainty_threshold = 0.975
    cfg.onboarding.flow_reliability_threshold = 0.5
    cfg.onboarding.frame_filter = 'dense_matching'
    cfg.onboarding.filter_matcher = 'UFM'
    cfg.onboarding.view_graph_strategy = 'dense'

    return cfg

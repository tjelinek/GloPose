from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'passthrough'
    cfg.onboarding.allow_disk_cache = False
    cfg.onboarding.passthrough_skip = 2
    cfg.onboarding.view_graph_strategy = 'linear'  # sequential chain: (k0,k1),(k1,k2),(k2,k3),...

    return cfg

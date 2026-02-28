from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'passthrough'
    cfg.onboarding.allow_disk_cache = False
    cfg.onboarding.passthrough_skip = 1

    return cfg

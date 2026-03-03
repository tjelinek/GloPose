from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'SIFT'
    cfg.onboarding.reconstruction_matcher = 'SIFT'

    return cfg

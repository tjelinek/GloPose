from configs.glopose_config import GloPoseConfig


def get_config() -> GloPoseConfig:
    cfg = GloPoseConfig()

    cfg.onboarding.frame_filter = 'max_visible'
    cfg.onboarding.reconstruction_method = 'sam3d'
    cfg.input.frame_provider_config.background_color = 'original'

    return cfg

glotracker_roma_c05_fg_025_m50_fgmr_01.pyfrom tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.optimize_shape = False

    cfg.matcher = 'RoMa'

    return cfg

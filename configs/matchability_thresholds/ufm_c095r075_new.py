from tracker_config import TrackerConfig


def get_config() -> TrackerConfig:
    cfg = TrackerConfig()

    cfg.min_roma_certainty_threshold = 0.95
    cfg.flow_reliability_threshold = 0.75
    cfg.frame_filter = 'dense_matching'
    cfg.frame_filter_matcher = 'UFM'
    cfg.frame_filter_view_graph = 'from_matching'
    cfg.add_track_merging_matches = True

    return cfg

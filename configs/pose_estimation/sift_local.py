"""Pose estimation config for local testing with SIFT+LightGlue matching.

Lightweight config that runs on CPU without requiring large dense matching models (RoMa, UFM).
Uses cached ViewGraphs from RCI and default CNOS detections.
"""

from configs.glopose_config import GloPoseConfig, PoseEstimationConfig, OnboardingConfig, PathsConfig, RunConfig

config = GloPoseConfig(
    run=RunConfig(
        device='cpu',
    ),
    paths=PathsConfig(
        cache_folder='/home/tom/rci_data/cache/',
        results_folder='/home/tom/Projects/GloPose/results/',
        bop_data_folder='/mnt/data/vrg/public_datasets/bop/',
    ),
    onboarding=OnboardingConfig(
        filter_matcher='SIFT',
    ),
    pose_estimation=PoseEstimationConfig(
        matcher='SIFT',
        sample_size=10000,
        min_certainty_threshold=0.5,
        flow_reliability_threshold=0.15,
        black_background=True,
        max_templates_to_match=10,
    ),
)

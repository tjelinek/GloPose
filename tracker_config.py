from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from kornia.image import ImageSize

from configs.components_config.base_BOP_config import BaseBOPConfig
from configs.components_config.base_pose_estimation_config import BasePoseEstimationConfig
from configs.components_config.frame_provider_config import BaseFrameProviderConfig
from configs.matching_configs.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching_configs.sift_configs.base_sift_config import BaseSiftConfig
from configs.matching_configs.ufm_configs.base_ufm_config import BaseUFMConfig


@dataclass
class TrackerConfig:
    # General settings
    device: str = 'cuda'
    default_data_folder: Path = Path('/mnt/data/vrg/public_datasets/').expanduser()
    default_results_folder: Path = Path('/mnt/personal/jelint19/results/FlowTracker/').expanduser()
    default_cache_folder: Path = Path('/mnt/personal/jelint19/cache/').expanduser()
    purge_cache: bool = False

    dataset: str = None
    sequence: str = None
    experiment_name: str = None
    special_hash: str = ''
    object_id: int | str = None

    # Visualization settings
    write_to_rerun_rather_than_disk: bool = True
    rerun_jpeg_quality: int = 75
    large_images_results_write_frequency: int = 1

    # Input data settings
    input_frames: int = None
    skip_indices: int = 1
    frame_provider: str = 'precomputed'  # 'precomputed' or 'synthetic'
    frame_provider_config: BaseFrameProviderConfig = field(default_factory=BaseFrameProviderConfig)
    segmentation_provider: str = 'SAM2'  # 'precomputed', 'SAM2', 'whites', or 'synthetic'
    gt_flow_source: str = 'FlowNetwork'  # One of 'FlowNetwork', 'GenerateSynthetic'
    image_downsample: float = 1.0
    depth_scale_to_meter: float = 1.0  # depth * depth_scale_to_meter = <depth_in_meters>
    run_only_on_frames_with_known_pose: bool = True

    # Renderer settings
    camera_position: Tuple[float, float, float] = (0, 0, 5.0)
    camera_up: Tuple[float, float, float] = (0, 1, 0)
    obj_center: Tuple[float, float, float] = (0, 0, 0)
    rendered_image_shape: ImageSize = ImageSize(500, 500)
    sigmainv: float = 7000
    features: str = 'deep'

    # Mesh settings
    mesh_normalize: bool = False
    texture_size: int = 1000
    gt_mesh_path: Path = None
    optimize_shape: bool = False
    gt_texture_path: Path = None

    # Tracking initialization
    tran_init: Tuple[float] = (0., 0., 0.)  # (0., 0., 0.)
    rot_init: Tuple[float] = (0., 0., 0.)   # (0., 0., 0.)

    # Frame Filter configurations
    roma_sample_size: int = 10000
    min_roma_certainty_threshold: float = 0.975
    flow_reliability_threshold: float = 0.5
    min_number_of_reliable_matches: int = 0
    matchability_based_reliability: bool = False
    frame_filter: str = 'dense_matching'  # Either 'dense_matching', 'SIFT', or 'passthrough'
    frame_filter_view_graph: str = 'from_matching'  # Either 'dense' or 'from_matching'
    passthrough_frame_filter_skip: int = 1

    # Dense matching config
    frame_filter_matcher: str = 'UFM'  # 'UFM' or 'RoMa'
    dense_matching_allow_disk_cache: bool = True
    roma_config: BaseRomaConfig = field(default_factory=BaseRomaConfig)
    ufm_config: BaseUFMConfig = field(default_factory=BaseUFMConfig)

    # Reconstruction settings
    mapper: str = 'pycolmap'  # Either 'colmap', 'pycolmap', or 'glomap'
    init_with_first_two_images: bool = False
    add_track_merging_matches: bool = True
    use_default_colmap_K: bool = True
    reconstruction_matcher: str = 'UFM'  # Either SIFT, UFM, or RoMa
    similarity_transformation: str = 'kabsch'  # Either 'depths' or 'kabsch'

    # SIFT options
    sift_matcher_config: BaseSiftConfig = field(default_factory=BaseSiftConfig)
    sift_filter_min_matches: int = 100
    sift_filter_good_to_add_matches: int = 450
    sift_cache: Path = None
    sift_mapping_num_feats: int = 8192
    sift_mapping_min_matches: int = 15

    # BOP Config
    bop_config: BaseBOPConfig = field(default_factory=BaseBOPConfig)
    export_view_graph: bool = False

    # Pose Estimation Config
    pose_estimation_config: BasePoseEstimationConfig = field(default_factory=BasePoseEstimationConfig)

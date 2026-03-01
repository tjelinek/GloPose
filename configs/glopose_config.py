
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

from kornia.image import ImageSize

from configs.components.bop_config import BaseBOPConfig
from configs.components.frame_provider_config import BaseFrameProviderConfig
from configs.matching.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching.sift_configs.base_sift_config import BaseSiftConfig
from configs.matching.ufm_configs.base_ufm_config import BaseUFMConfig


@dataclass
class PathsConfig:
    results_folder: Path = Path('/mnt/personal/jelint19/results/FlowTracker/').expanduser()
    cache_folder: Path = Path('/mnt/personal/jelint19/cache/').expanduser()
    purge_cache: bool = False

    bop_data_folder: Path = Path('/mnt/data/vrg/public_datasets/bop/')
    ho3d_data_folder: Path = Path('/mnt/personal/jelint19/data/HO3D/')
    navi_data_folder: Path = Path('/mnt/personal/jelint19/data/NAVI/navi_v1.5/')
    behave_data_folder: Path = Path('/mnt/personal/jelint19/data/BEHAVE/')
    tum_rgbd_data_folder: Path = Path('/mnt/personal/jelint19/data/SLAM/tum_rgbd/')
    google_scanned_objects_data_folder: Path = Path('/mnt/personal/jelint19/data/GoogleScannedObjects/')
    handal_data_folder: Path = Path('/mnt/personal/jelint19/data/HANDAL/')


@dataclass
class RunConfig:
    device: str = 'cuda'
    dataset: str = None
    sequence: str = None
    experiment_name: str = None
    special_hash: str = ''
    object_id: int | str = None


@dataclass
class InputConfig:
    input_frames: int = None
    skip_indices: int = 1
    frame_provider: str = 'precomputed'
    frame_provider_config: BaseFrameProviderConfig = field(default_factory=BaseFrameProviderConfig)
    segmentation_provider: str = 'SAM2'
    gt_flow_source: str = 'FlowNetwork'
    image_downsample: float = 1.0
    depth_scale_to_meter: float = 1.0
    run_only_on_frames_with_known_pose: bool = True


@dataclass
class OnboardingConfig:
    # Frame filter settings
    frame_filter: str = 'dense_matching'
    view_graph_strategy: str = 'from_matching'
    passthrough_skip: int = 1
    min_certainty_threshold: float = 0.975
    flow_reliability_threshold: float = 0.5
    min_number_of_reliable_matches: int = 0
    matchability_based_reliability: bool = False
    sample_size: int = 10000

    # Matching settings
    filter_matcher: str = 'UFM'
    reconstruction_matcher: str = 'UFM'
    allow_disk_cache: bool = True
    roma: BaseRomaConfig = field(default_factory=BaseRomaConfig)
    ufm: BaseUFMConfig = field(default_factory=BaseUFMConfig)
    sift: BaseSiftConfig = field(default_factory=BaseSiftConfig)
    sift_filter_min_matches: int = 100
    sift_filter_good_to_add_matches: int = 450

    # Reconstruction settings
    mapper: str = 'pycolmap'
    init_with_first_two_images: bool = False
    add_track_merging_matches: bool = True
    use_default_colmap_K: bool = True
    similarity_transformation: str = 'kabsch'
    sift_mapping_num_feats: int = 8192
    sift_mapping_min_matches: int = 15
    export_view_graph: bool = False


@dataclass
class CondensationConfig:
    method: str = 'hart'
    descriptor_model: str = 'dinov2'
    descriptor_mask_detections: bool = True
    min_cls_cosine_similarity: float = 0.15
    min_avg_patch_cosine_similarity: float = 0.15
    patch_descriptors_filtering: bool = True
    whiten_dim: int = 0
    csls_k: int = 10
    augment_with_split_detections: bool = True
    augment_with_train_pbr_detections: bool = True
    augmentations_detector: str = 'sam2'
    split: str = 'onboarding_static'


@dataclass
class DetectionConfig:
    templates_source: str = 'cnns'
    condensation_source: str = '1nn-hart'
    descriptor_model: str = 'dinov2'
    descriptor_mask_detections: bool = True
    detector_name: str = 'sam'
    aggregation_function: str = None
    similarity_metric: str = 'cosine'
    confidence_thresh: float = None
    ood_detection_method: str = None
    cosine_similarity_quantile: float = None
    mahalanobis_quantile: float = None
    lowe_ratio_threshold: float = None
    patch_descriptors_filtering: bool = True
    min_avg_patch_cosine_similarity: float = 0.25
    nms_thresh: float = None


@dataclass
class VisualizationConfig:
    write_to_rerun: bool = True
    jpeg_quality: int = 75
    large_images_write_frequency: int = 1


@dataclass
class PoseEstimationConfig:
    matcher: str = 'UFM'
    sample_size: int = 10000
    min_certainty_threshold: float = 0.975
    flow_reliability_threshold: float = 0.5
    black_background: bool = True
    max_templates_to_match: int = 10


@dataclass
class RendererConfig:
    camera_position: Tuple[float, float, float] = (0, 0, 5.0)
    camera_up: Tuple[float, float, float] = (0, 1, 0)
    obj_center: Tuple[float, float, float] = (0, 0, 0)
    rendered_image_shape: ImageSize = ImageSize(500, 500)
    sigmainv: float = 7000
    features: str = 'deep'
    mesh_normalize: bool = False
    texture_size: int = 1000
    gt_mesh_path: Path = None
    optimize_shape: bool = False
    gt_texture_path: Path = None
    tran_init: Tuple[float] = (0., 0., 0.)
    rot_init: Tuple[float] = (0., 0., 0.)


@dataclass
class GloPoseConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    run: RunConfig = field(default_factory=RunConfig)
    input: InputConfig = field(default_factory=InputConfig)
    onboarding: OnboardingConfig = field(default_factory=OnboardingConfig)
    condensation: CondensationConfig = field(default_factory=CondensationConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    bop: BaseBOPConfig = field(default_factory=BaseBOPConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)

import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

from data_providers.frame_provider import PrecomputedSegmentationProvider
from onboarding_pipeline import OnboardingPipeline
from tracker_config import TrackerConfig
from utils.general import load_config
from utils.runtime_utils import parse_args


def run_on_custom_data(images_paths: List[Path], segmentations_paths: Optional[List[Path]]):
    config, write_folder = prepare_config(images_paths)

    first_segment_tensor = PrecomputedSegmentationProvider.get_initial_segmentation(images_paths, segmentations_paths)

    tracker = OnboardingPipeline(config, write_folder, input_images=images_paths,
                                 input_segmentations=segmentations_paths,
                                 initial_segmentation=first_segment_tensor)
    view_graph = tracker.run_pipeline()


def prepare_config(images_paths) -> Tuple[TrackerConfig, Path]:
    dataset = 'custom_input'
    combined_paths = '\n'.join(str(p) for p in images_paths)
    hash_object = hashlib.sha256(combined_paths.encode('utf-8'))
    sequence_hash = hash_object.hexdigest()[:16]
    sequence = sequence_hash
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    experiment_name = args.experiment
    config.experiment_name = experiment_name
    config.sequence = sequence
    config.dataset = dataset
    config.frame_provider_config.erode_segmentation = True
    config.input_frames = len(images_paths)

    write_folder = config.default_results_folder / experiment_name / dataset / sequence

    return config, write_folder


if __name__ == "__main__":
    base_seq_path = Path('/mnt/personal/jelint19/data/bop/handal/onboarding_static/obj_000005_down/')
    img_paths = sorted(Path(base_seq_path / 'rgb').iterdir())[::10]
    seg_paths = sorted(Path(base_seq_path / 'mask_visib').iterdir())[::10]

    run_on_custom_data(img_paths, seg_paths)

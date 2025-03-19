from pathlib import Path

from tracker_config import TrackerConfig
from utils.data_utils import get_initial_image_and_segment
from utils.bop_challenge import (read_gt_Se3_cam2obj_transformations, get_bop_images_and_segmentations,
                                 read_pinhole_params)
from utils.general import load_config
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D


def main():
    dataset = 'handal'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            '000001',
            '000002',
            '000003',
            '000004',
            '000005',
        ]

    for sequence in sequences:
        config = load_config(args.config)

        if config.gt_flow_source == 'GenerateSynthetic':
            exit()

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5
        config.large_images_results_write_frequency = 4

        skip_indices = 1

        sequence_type = 'val'
        onboarding_type = None

        run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, skip_indices,
                             onboarding_type)


def run_on_bop_sequences(dataset: str, experiment_name: str, sequence: str, sequence_type: str, args,
                         config: TrackerConfig, skip_indices: int, onboarding_type: str = None):
    if args.output_folder is not None:
        write_folder = Path(args.output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / sequence
    bop_folder = config.default_data_folder / 'bop'
    gt_images, gt_segs, sequence_starts = get_bop_images_and_segmentations(bop_folder, dataset, sequence,
                                                                           sequence_type, onboarding_type)
    dict_gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence, sequence_type,
                                                              onboarding_type, sequence_starts, config.device)
    gt_Se3_obj2cam_frame0 = dict_gt_Se3_cam2obj[min(dict_gt_Se3_cam2obj.keys())]
    valid_indices = sorted(list(dict_gt_Se3_cam2obj.keys()))[::skip_indices]
    gt_images = [gt_images[i] for i in valid_indices]
    gt_segs = [gt_segs[i] for i in valid_indices]
    dict_gt_Se3_cam2obj = {i: dict_gt_Se3_cam2obj[frame] for i, frame in enumerate(valid_indices)}
    first_image, first_segmentation = get_initial_image_and_segment(gt_images, gt_segs, segmentation_channel=0)
    pinhole_params = read_pinhole_params(bop_folder, dataset, sequence, sequence_type,
                                         None, sequence_starts)
    config.camera_intrinsics = pinhole_params[min(valid_indices)].intrinsics.squeeze().numpy(force=True)
    config.camera_extrinsics = pinhole_params[min(valid_indices)].extrinsics.squeeze().numpy(force=True)
    config.input_frames = len(gt_images)
    config.frame_provider = 'precomputed'
    config.segmentation_provider = 'SAM2'
    sfb = Tracker6D(config, write_folder, initial_gt_Se3_cam2obj=gt_Se3_obj2cam_frame0,
                    gt_Se3_cam2obj=dict_gt_Se3_cam2obj, images_paths=gt_images, segmentation_paths=gt_segs,
                    initial_segmentation=first_segmentation,
                    initial_image=first_image)
    sfb.run_filtering_with_reconstruction()


if __name__ == "__main__":
    main()

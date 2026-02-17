from pathlib import Path

from data_providers.frame_provider import PrecomputedSegmentationProvider
from tracker6d import Tracker6D
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
                                read_object_id, read_static_onboarding_world2cam, add_extrinsics_to_pinhole_params, \
                                read_pinhole_params
from utils.dataset_sequences import get_bop_classic_sequences
from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    args = parse_args()
    config = load_config(args.config)

    bop_path = config.default_data_folder / 'bop'
    tless_seqs = get_bop_classic_sequences(bop_path, 'tless', 'train_primesense')
    lmo_seqs = get_bop_classic_sequences(bop_path, 'lmo', 'train')
    icbin_seqs = get_bop_classic_sequences(bop_path, 'icbin', 'train')

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (tless_seqs + lmo_seqs + icbin_seqs)[0:1]

    for sequence_code in sequences:
        sequence_code_split = sequence_code.split('@')
        dataset, onboarding_folder, sequence_name = sequence_code_split

        with exception_logger():
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.dataset = dataset
            config.sequence = sequence_name
            config.image_downsample = 1.
            config.large_images_results_write_frequency = 4
            config.depth_scale_to_meter = 0.001
            config.skip_indices *= 4

            # Path to BOP dataset
            bop_folder = config.default_data_folder / 'bop'
            # Determine output folder
            if args.output_folder is not None:
                folder = Path(args.output_folder) / dataset / f'{sequence_name}'
            else:
                folder = config.default_results_folder / experiment_name / dataset / f'{sequence_name}'

            # Load images and segmentations
            gt_images, gt_segs, gt_depths, sequence_starts = \
                get_bop_images_and_segmentations(bop_folder, dataset, sequence_name, onboarding_folder)
            # Get camera-to-object transformations
            dict_gt_Se3_cam2obj = \
                read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence_name, onboarding_folder, 1.0,
                                                    device=config.device)

            object_id = read_object_id(bop_folder, dataset, sequence_name, onboarding_folder)
            config.object_id = object_id
            # Apply frame skipping
            if config.run_only_on_frames_with_known_pose:
                valid_frames = sorted(dict_gt_Se3_cam2obj.keys())
            else:
                valid_frames = list(range(min(gt_images.keys()), max(gt_images.keys()) + 1))
            gt_images = [gt_images[i] for i in valid_frames]
            gt_segs = [gt_segs[i] for i in valid_frames]
            if gt_depths is not None:
                gt_depths = [gt_depths[i] for i in valid_frames]
            dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)
            # Get initial image and segmentation
            first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                          segmentation_channel=0)
            # Get camera parameters
            pinhole_params = read_pinhole_params(bop_folder, dataset, sequence_name, onboarding_folder,
                                                 config.image_downsample, device=config.device)

            pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)

            gt_Se3_obj2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}

            # Update config with frame information
            config.input_frames = len(gt_images)
            config.frame_provider = 'precomputed'
            config.segmentation_provider = 'SAM2'
            # Initialize and run the tracker
            tracker = Tracker6D(config, folder, input_images=gt_images, gt_Se3_world2cam=gt_Se3_obj2cam,
                                gt_pinhole_params=pinhole_params, input_segmentations=gt_segs, depth_paths=gt_depths,
                                initial_segmentation=first_segmentation)
            tracker.run_pipeline()


if __name__ == "__main__":
    main()

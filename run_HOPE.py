import time
from pathlib import Path

from utils.data_utils import get_initial_image_and_segment
from utils.bop_challenge import (get_bop_images_and_segmentations,
                                 read_gt_Se3_cam2obj_transformations, read_pinhole_params)
from utils.general import load_config
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D


def main():
    dataset = 'hope'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'obj_000001', 'obj_000006', 'obj_000011', 'obj_000016', 'obj_000021', 'obj_000026',
            'obj_000002', 'obj_000007', 'obj_000012', 'obj_000017', 'obj_000022', 'obj_000027',
            'obj_000003', 'obj_000008', 'obj_000013', 'obj_000018', 'obj_000023', 'obj_000028',
            'obj_000004', 'obj_000009', 'obj_000014', 'obj_000019', 'obj_000024',
            'obj_000005', 'obj_000010', 'obj_000015', 'obj_000020', 'obj_000025',
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

        skip_indices = 1

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = config.default_results_folder / experiment_name / dataset / sequence

        t0 = time.time()

        onboarding_type = config.bop_config.onboarding_type
        obj_id = 0

        sequence_type = 'onboarding'

        bop_folder = config.default_data_folder / 'bop'

        gt_images, gt_segs, sequence_starts = get_bop_images_and_segmentations(bop_folder, dataset, sequence,
                                                                               sequence_type, onboarding_type)

        gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence, sequence_type,
                                                             onboarding_type, sequence_starts, config.device)

        pinhole_params = read_pinhole_params(bop_folder, dataset, sequence, sequence_type,
                                             onboarding_type, sequence_starts)

        gt_Se3_cam2obj_first_frame = gt_Se3_cam2obj[0]

        first_image, first_segmentation = get_initial_image_and_segment(gt_images, gt_segs, segmentation_channel=0)

        config.camera_intrinsics = pinhole_params[0].intrinsics.squeeze().numpy(force=True)
        config.camera_extrinsics = pinhole_params[0].extrinsics.squeeze().numpy(force=True)
        config.input_frames = len(gt_images)
        config.frame_provider = 'precomputed'
        config.segmentation_provider = 'SAM2'

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        tracker = Tracker6D(config, write_folder, initial_gt_Se3_cam2obj=gt_Se3_cam2obj_first_frame,
                            images_paths=gt_images, segmentation_paths=gt_segs,
                            initial_segmentation=first_segmentation, initial_image=first_image,
                            sequence_starts=sequence_starts)

        tracker.run_filtering_with_reconstruction()


if __name__ == "__main__":
    main()

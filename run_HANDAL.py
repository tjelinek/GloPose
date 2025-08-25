from pathlib import Path
from typing import Tuple, List

from data_providers.frame_provider import PrecomputedSegmentationProvider
from tracker6d import Tracker6D
from utils.bop_challenge import add_extrinsics_to_pinhole_params, load_gt_images, load_gt_segmentations, \
                                 extract_gt_Se3_cam2obj, extract_object_id, get_pinhole_params

from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def get_handal_sequences() -> Tuple[List[Path], List[Path]]:
    dataset_path = Path("/mnt/personal/jelint19/data/HANDAL")

    train_sequences = []
    test_sequences = []

    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        train_dir = category_dir / "train"
        if train_dir.exists():
            for sequence_dir in train_dir.iterdir():
                if sequence_dir.is_dir():
                    train_sequences.append(f"{category_name}@{sequence_dir.name}")

        test_dir = category_dir / "test"
        if test_dir.exists():
            for sequence_dir in test_dir.iterdir():
                if sequence_dir.is_dir():
                    test_sequences.append(f"{category_name}@{sequence_dir.name}")

    return train_sequences, test_sequences


HANDAL_TRAIN_SEQUENCES, HANDAL_TEST_SEQUENCES = get_handal_sequences()


def main():
    dataset = 'handal_native'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = HANDAL_TEST_SEQUENCES[4:5]

    for obj_type_sequence in sequences:
        with (exception_logger()):

            if obj_type_sequence in HANDAL_TRAIN_SEQUENCES:
                sequence_type = 'train'
            elif obj_type_sequence in HANDAL_TEST_SEQUENCES:
                sequence_type = 'test'
            else:
                raise ValueError(f"Unknown sequence {obj_type_sequence}")

            obj_name, sequence = obj_type_sequence.split('@')
            config = load_config(args.config)

            experiment_name = args.experiment
            output_folder = args.output_folder

            config.experiment_name = experiment_name
            config.sequence = sequence
            config.dataset = dataset
            config.image_downsample = .5
            config.large_images_results_write_frequency = 8

            config.skip_indices *= 1

            config.special_hash = obj_name.replace('handal_dataset_', '')

            # Determine output folder
            if output_folder is not None:
                write_folder = Path(output_folder) / dataset / f'{config.special_hash}_{sequence}'
            else:
                write_folder = config.default_results_folder / experiment_name / dataset / \
                               f'{config.special_hash}_{sequence}'

            base_folder = config.default_data_folder / 'HANDAL' / obj_name / sequence_type / sequence
            image_folder = base_folder / 'rgb'
            segmentation_folder = base_folder / 'mask_visib'
            scene_gt_path = base_folder / 'scene_gt.json'
            scene_cam_path = base_folder / 'scene_camera.json'

            gt_images = load_gt_images(image_folder)
            gt_segs = load_gt_segmentations(segmentation_folder)

            cam_scale = 1.0
            dict_gt_Se3_cam2obj = extract_gt_Se3_cam2obj(scene_gt_path, cam_scale, device=config.device)
            object_id = extract_object_id(scene_gt_path)[1]
            config.object_id = object_id

            valid_frames = sorted(set(gt_images.keys()) & set(gt_segs.keys()) & set(dict_gt_Se3_cam2obj.keys()))

            gt_images = [gt_images[i] for i in valid_frames]
            gt_segs = [gt_segs[i] for i in valid_frames]

            dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)
            gt_Se3_world2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}

            pinhole_params = get_pinhole_params(scene_cam_path, config.image_downsample, device=config.device)
            pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)
            pinhole_params = add_extrinsics_to_pinhole_params(pinhole_params, gt_Se3_world2cam)

            first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                          segmentation_channel=0)

            config.input_frames = len(gt_images)
            config.frame_provider = 'precomputed'
            config.segmentation_provider = 'SAM2'

            tracker = Tracker6D(config, write_folder, input_images=gt_images, gt_Se3_cam2obj=dict_gt_Se3_cam2obj,
                                gt_Se3_world2cam=gt_Se3_world2cam, gt_pinhole_params=pinhole_params,
                                input_segmentations=gt_segs, initial_segmentation=first_segmentation)
            tracker.run_pipeline()


if __name__ == "__main__":
    main()

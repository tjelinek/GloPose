import time
from pathlib import Path

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion

from utils.data_utils import get_initial_image_and_segment
from utils.dataset_utils.bop_challenge import get_pinhole_params, read_obj_to_cam_transformations_from_gt, \
    load_gt_images_and_segmentations
from utils.general import load_config
from utils.runtime_utils import parse_args
from tracker6d import run_tracking_on_sequence


def main():
    dataset = 'HANDAL'
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

        skip_indices = 4

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = config.default_results_folder / experiment_name / dataset / sequence

        t0 = time.time()

        sequence_folder = config.default_data_folder / 'bop' / 'handal' / 'val' / sequence
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'

        gt_images, gt_segs = load_gt_images_and_segmentations(image_folder, segmentation_folder)

        pose_json_path = sequence_folder / 'scene_gt.json'

        gt_rotations_objs_to_cam, gt_translations_objs_to_cam = read_obj_to_cam_transformations_from_gt(pose_json_path)
        obj_ids = sorted(gt_rotations_objs_to_cam.keys())
        gt_rotations_obj_to_cam_dict = gt_rotations_objs_to_cam[obj_ids[0]]
        gt_translations_obj_to_cam_dict = gt_translations_objs_to_cam[obj_ids[0]]

        pinhole_params = get_pinhole_params(sequence_folder / 'scene_camera.json')
        valid_indices = list(sorted(gt_segs.keys() & gt_images.keys() & gt_translations_obj_to_cam_dict.keys() &
                                    gt_rotations_obj_to_cam_dict.keys()))
        valid_indices = valid_indices[::skip_indices]

        # Filter the lists to include only valid elements
        filtered_gt_rotations_obj_i_to_obj_i = [gt_rotations_obj_to_cam_dict[i] for i in valid_indices]
        filtered_gt_translations_obj_1_to_obj_i = [gt_translations_obj_to_cam_dict[i] for i in valid_indices]
        gt_images = [gt_images[i] for i in valid_indices]
        gt_segs = [gt_segs[i] for i in valid_indices]

        config.input_frames = len(gt_images)

        config.camera_intrinsics = pinhole_params[0].intrinsics.squeeze().numpy(force=True)
        config.camera_extrinsics = pinhole_params[0].extrinsics.squeeze().numpy(force=True)
        config.frame_provider = 'precomputed'
        config.segmentation_provider = 'SAM2'

        gt_rotations_obj_to_cam = torch.from_numpy(np.array(filtered_gt_rotations_obj_i_to_obj_i))
        gt_translations_obj_to_cam = torch.from_numpy(np.array(filtered_gt_translations_obj_1_to_obj_i))

        gt_rotations_obj_to_cam = gt_rotations_obj_to_cam.to(torch.float32).to(config.device)
        gt_translations_obj_to_cam = gt_translations_obj_to_cam.to(torch.float32).to(config.device)

        gt_Se3_obj_to_cam = Se3(Quaternion.from_axis_angle(gt_rotations_obj_to_cam), gt_translations_obj_to_cam)
        gt_Se3_cam_to_obj = gt_Se3_obj_to_cam.inverse()

        first_image, first_segmentation = get_initial_image_and_segment(gt_images, gt_segs, segmentation_channel=0)

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        run_tracking_on_sequence(config, write_folder, gt_Se3_cam2obj=gt_Se3_cam_to_obj,
                                 images_paths=gt_images, segmentation_paths=gt_segs,
                                 initial_segmentation=first_segmentation, initial_image=first_image)


if __name__ == "__main__":
    main()

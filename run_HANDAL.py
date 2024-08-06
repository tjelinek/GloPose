import numpy as np
import torch
import sys
import time
import json
from pathlib import Path

from kornia.geometry import rotation_matrix_to_axis_angle

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config

sys.path.append('repositories/OSTrack/S2DNet')


def main():
    dataset = 'HANDAL'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            '004006',
            '004002'
        ]

    for sequence in sequences:
        config = load_config(args.config)

        if config.augment_gt_track:
            exit()

        gt_texture_path = None
        gt_mesh_path = None

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = gt_mesh_path
        config.gt_track_path = None
        config.sequence = sequence
        config.image_downsample = 0.25

        # config.camera_position = (-5.0, -5.0, -5.0)
        # config.camera_up = (0, 0, 1)

        skip_indices = 1

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        t0 = time.time()

        sequence_folder = Path(dataset_folder) / 'HANDAL' / 'handal_dataset_hammers' / 'train' / sequence
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask'

        gt_images_list = [file for file in sorted(image_folder.iterdir()) if file.is_file()]
        gt_segmentations_list = [file for file in sorted(segmentation_folder.iterdir()) if file.is_file()]

        gt_translations = []
        gt_rotations = []
        cam_intrinsics_list = []

        pose_json_path = sequence_folder / 'scene_gt.json'
        with open(pose_json_path, 'r') as file:
            pose_json = json.load(file)
            for frame, data in pose_json.items():
                for entry in data:
                    cam_R_m2c = entry['cam_R_m2c']
                    R = torch.tensor(np.array(cam_R_m2c).reshape(3, 3))
                    r = rotation_matrix_to_axis_angle(R).numpy()

                    cam_t_m2c = entry['cam_t_m2c']
                    t = np.array(cam_t_m2c)

                    gt_rotations.append(r)
                    gt_translations.append(t)

        camera_calibrations_json_path = sequence_folder / 'scene_camera.json'
        with open(camera_calibrations_json_path, 'r') as file:
            pose_json = json.load(file)
            for frame, data in pose_json.items():
                cam_K = data['cam_K']
                K = np.array(cam_K).reshape(3, 3)

                cam_intrinsics_list.append(K)

        config.generate_synthetic_observations_if_possible = False

        valid_indices = [i for i in range(len(gt_rotations))]
        valid_indices = valid_indices[::skip_indices]

        # Filter the lists to include only valid elements
        filtered_gt_rotations = [gt_rotations[i] for i in valid_indices]
        filtered_gt_translations = [gt_translations[i] for i in valid_indices]
        gt_images_list = [gt_images_list[i] for i in valid_indices]
        gt_segmentations_list = [gt_segmentations_list[i] for i in valid_indices]

        config.input_frames = len(gt_images_list)

        rotations_array = torch.from_numpy(np.array(filtered_gt_rotations)[None]).cuda().to(torch.float32)
        translations_array = torch.from_numpy(np.array(filtered_gt_translations)[None, None]).cuda().to(torch.float32)

        config.rot_init = tuple(rotations_array[0, 0].numpy(force=True).tolist())
        config.tran_init = tuple(translations_array[0, 0, 0].numpy(force=True).tolist())

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        cam_intrinsics = torch.from_numpy(cam_intrinsics_list[0]).cuda()

        run_tracking_on_sequence(config, write_folder, gt_texture=None, gt_mesh=None, gt_rotations=rotations_array,
                                 gt_translations=translations_array, images_paths=gt_images_list,
                                 segmentation_paths=gt_segmentations_list, camera_intrinsics=cam_intrinsics)


if __name__ == "__main__":
    main()

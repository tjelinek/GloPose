import numpy as np
import torch
import time
import json
from pathlib import Path

from kornia.geometry import rotation_matrix_to_axis_angle

from utils.dataset_utils.bop_challenge import get_pinhole_params
from utils.runtime_utils import run_tracking_on_sequence, parse_args
from utils.general import load_config


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

        gt_texture_path = None
        gt_mesh_path = None

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5

        # config.camera_position = (-5.0, -5.0, -5.0)
        # config.camera_up = (0, 0, 1)

        skip_indices = 4

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = config.default_results_folder / experiment_name / dataset / sequence

        t0 = time.time()

        sequence_folder = config.default_data_folder / 'bop' / 'handal' / 'val' / sequence
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'

        gt_segs = {int(file.stem.split('_')[0]): file for file in sorted(segmentation_folder.iterdir()) if
                       file.stem.endswith('000000')}
        gt_images = {int(file.stem): file for file in sorted(image_folder.iterdir()) if file.is_file()}

        gt_translations = {}
        gt_rotations = {}
        cam_intrinsics = {}

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

                    gt_rotations[int(frame)] = r
                    gt_translations[int(frame)] = t

        camera_calibrations_json_path = sequence_folder / 'scene_camera.json'
        with open(camera_calibrations_json_path, 'r') as file:
            pose_json = json.load(file)
            for frame, data in pose_json.items():
                cam_K = data['cam_K']
                K = np.array(cam_K).reshape(3, 3)

                cam_intrinsics[int(frame)] = K

        # config.camera_intrinsics = cam_intrinsics[0]
        pinhole_params = get_pinhole_params(sequence_folder / 'scene_camera.json')
        config.camera_intrinsics = pinhole_params[0].intrinsics.squeeze().numpy(force=True)
        config.camera_extrinsics = pinhole_params[0].extrinsics.squeeze().numpy(force=True)

        config.generate_synthetic_observations_if_possible = False

        valid_indices = list(sorted(gt_segs.keys() & gt_images.keys() & gt_translations.keys() &
                                    gt_rotations.keys()))
        valid_indices = valid_indices[::skip_indices]

        # Filter the lists to include only valid elements
        filtered_gt_rotations = [gt_rotations[i] for i in valid_indices]
        filtered_gt_translations = [gt_translations[i] for i in valid_indices]
        gt_images = [gt_images[i] for i in valid_indices]
        gt_segs = [gt_segs[i] for i in valid_indices]
        cam_intrinsics = [cam_intrinsics[i] for i in sorted(list(cam_intrinsics.keys()))]

        config.input_frames = len(gt_images)

        rotations_array = torch.from_numpy(np.array(filtered_gt_rotations)).cuda().to(torch.float32)
        translations_array = torch.from_numpy(np.array(filtered_gt_translations)).cuda().to(torch.float32)

        # config.rot_init = tuple(rotations_array[0, 0].numpy(force=True).tolist())
        # config.tran_init = tuple(translations_array[0, 0, 0].numpy(force=True).tolist())

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        run_tracking_on_sequence(config, write_folder, gt_texture=None, gt_mesh=None, gt_rotations=rotations_array,
                                 gt_translations=translations_array, images_paths=gt_images,
                                 segmentation_paths=gt_segs)


if __name__ == "__main__":
    main()

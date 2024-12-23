import glob
import numpy as np
import os
import time
from pathlib import Path

import torch

from dataset_generators import scenarios
from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils.data_utils import load_gt_data
from utils.general import load_config


def main():
    dataset = 'SyntheticObjects'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'Textured_Sphere_5_y',
            'Textured_Cube_5_y',
            'Textured_Sphere_5_z',
            'Translating_Textured_Sphere',
            'Textured_Sphere_10_xy',
            'Rotating_Translating_Textured_Sphere_5_y',
            'Rotating_Translating_Textured_Sphere_5_xy',
            'Rotating_Contra_Translating_Textured_Sphere_5_y',
            'Rotating_Contra_Translating_Textured_Sphere_5_xy',
            '8_Colored_Sphere_5_x',
            '6_Colored_Cube_5_z']

    for sequence in sequences:
        config = load_config(args.config)

        if '8_Colored_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/8-colored-sphere/8-colored-sphere.obj')
            gt_texture_path = Path('prototypes/8-colored-sphere/8-colored-sphere-tex.png')
        elif 'Textured_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = Path('prototypes/tex.png')
        elif 'Textured_Cube' in sequence:
            gt_mesh_path = Path('prototypes/textured-cube/textured-cube.obj')
            gt_texture_path = Path('prototypes/textured-cube/tex.png')
        else:
            gt_texture_path = None
            gt_mesh_path = None

        gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path(sequence) / Path('gt_tracking_log') / \
                           Path('gt_tracking_log.csv')

        experiment_name = args.experiment

        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = gt_mesh_path
        config.gt_track_path = gt_tracking_path

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        renderings_folder = 'renderings'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, sequence, renderings_folder, '*.*')))
        files.sort()

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        skip_frames = 1
        gt_texture, gt_mesh, gt_rotations, gt_translations = load_gt_data(config)
        gt_rotations = torch.deg2rad(scenarios.random_walk_on_a_sphere().rotations).cuda().to(torch.float32)
        images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]

        gt_rotations = gt_rotations[::skip_frames]
        images_paths = images_paths[::skip_frames]
        gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations.cuda()

        config.input_frames = gt_rotations.shape[0]
        run_tracking_on_sequence(config, write_folder, gt_texture, gt_mesh, gt_rotations, gt_translations)


if __name__ == "__main__":
    main()

import glob
import os

import sys
import time
import shutil

import numpy as np

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config
from pathlib import Path

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def main():
    dataset = 'SyntheticObjects'
    sequences = ['Textured_Sphere', '8_Colored_Sphere', '6_Colored_Cube',
                 # '6_Colored_Cube_2_directions', 'Textured_Sphere_2_directions', '8_Colored_Sphere_2_directions'
                 ]

    for sequence in sequences:
        gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)

        if sequence in ['8_Colored_Sphere', '8_Colored_Sphere_2_directions']:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = None
        elif sequence in ['Textured_Sphere', 'Textured_Sphere_2_directions']:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = Path('prototypes/tex.png')
        else:
            gt_texture_path = None
            gt_mesh_path = None

        args = parse_args(sequence, dataset)

        experiment_name = args.experiment
        config = load_config(args.config)
        config["image_downsample"] = args.perc
        config["tran_init"] = 2.5
        config["gt_texture"] = gt_texture_path
        config["gt_mesh_prototype"] = gt_mesh_path

        write_folder = os.path.join(tmp_folder, experiment_name, args.dataset, args.sequence)
        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.makedirs(write_folder)
        os.makedirs(os.path.join(write_folder, 'imgs'))
        shutil.copyfile(os.path.join('prototypes', 'model.mtl'), os.path.join(write_folder, 'model.mtl'))
        config["sequence"] = args.sequence

        renderings_folder = 'renderings'
        segmentations_folder = 'segmentations'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, args.sequence, renderings_folder, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, args.sequence, segmentations_folder, '*.*')))

        segms.sort()
        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(args, config, files, segms, write_folder)


if __name__ == "__main__":
    main()

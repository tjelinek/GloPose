import argparse
import glob
import os

import sys
import time
import shutil

import numpy as np
import torch

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config
from types import SimpleNamespace
from pathlib import Path

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def main():
    dataset = 'GoogleScannedObjects'
    sequences = ['Squirrel', 'STACKING_BEAR', 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
                 'Schleich_Allosaurus', 'Threshold_Ramekin_White_Porcelain', 'Tag_Dishtowel_Green']

    for sequence in sequences:
        gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)
        gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
        gt_mesh_path = gt_model_path / Path('meshes/model.obj')

        args = parse_args(sequence, dataset)

        experiment_name = args.experiment
        config = load_config(args.config)
        config["image_downsample"] = args.perc
        config["tran_init"] = 2.5
        config["loss_dist_weight"] = 0
        config["gt_texture"] = gt_texture_path
        config["gt_mesh_prototype"] = gt_mesh_path
        config["use_gt"] = False

        write_folder = os.path.join(tmp_folder, args.dataset, experiment_name, args.sequence)

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
            glob.glob(os.path.join(dataset_folder, args.dataset, renderings_folder, args.sequence, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, segmentations_folder, args.sequence, '*.*')))

        segms.sort()
        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(args, config, files, segms, write_folder)


if __name__ == "__main__":
    main()

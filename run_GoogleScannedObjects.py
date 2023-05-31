import argparse
import glob
import os

import sys
import time
import shutil

import numpy as np
import torch

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence
from utils import load_config
from types import SimpleNamespace
from pathlib import Path

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def parse_args(sequence):
    dataset = 'GoogleScannedObjects'

    gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)
    gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
    gt_materials_path = gt_model_path / Path('meshes/model.mtl')
    gt_mesh_path = gt_model_path / Path('meshes/model.obj')

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_deep.yaml")
    parser.add_argument("--dataset", required=False, default=dataset)
    parser.add_argument("--sequence", required=False, default=sequence)
    parser.add_argument("--start", required=False, default=0)
    parser.add_argument("--length", required=False, default=72)
    parser.add_argument("--skip", required=False, default=1)
    parser.add_argument("--perc", required=False, default=0.15)
    parser.add_argument("--folder_name", required=False, default=dataset)
    parser.add_argument("--gt_texture", required=False, default=gt_texture_path)
    parser.add_argument("--gt_mesh_prototype", required=False, default=gt_mesh_path)
    parser.add_argument("--use_gt", required=False, default=False)
    return parser.parse_args()


def main():
    sequences = ['Squirrel', 'STACKING_BEAR', 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
                 'Schleich_Allosaurus', 'Threshold_Ramekin_White_Porcelain']

    for sequence in sequences:
        args = parse_args(sequence)
        config = load_config(args.config)
        config["image_downsample"] = args.perc
        config["tran_init"] = 2.5
        config["loss_dist_weight"] = 0
        config["gt_texture"] = args.gt_texture
        config["gt_mesh_prototype"] = args.gt_mesh_prototype
        config["use_gt"] = args.use_gt

        write_folder = os.path.join(tmp_folder, args.folder_name, args.sequence)
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

import glob
import os

import sys
import time
import shutil

import numpy as np

from pathlib import Path

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


# TODO: Copy data:
# scp rozumden@ptak.felk.cvut.cz:/datagrid/personal/rozumden/360photo/360photo.zip ~/scratch/dataset/360photo/
# unzip ~/scratch/dataset/360photo/360photo.zip -d ~/scratch/dataset/360photo/


def main():
    dataset = 'concept'
    folder_name = '360photo'
    sequences = ["09"]
    args = parse_args()

    for sequence in sequences:

        experiment_name = args.experiment
        config = load_config(args.config)
        config.sequence = sequence

        write_folder = os.path.join(tmp_folder, experiment_name, folder_name, sequence)
        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.makedirs(write_folder)
        os.makedirs(os.path.join(write_folder, 'imgs'))
        shutil.copyfile(os.path.join('prototypes', 'model.mtl'), os.path.join(write_folder, 'model.mtl'))

        renderings_folder = 'original'
        segmentations_folder = 'masks_U2Net'
        dataset_folder_ = Path(dataset_folder) / '360photo'

        t0 = time.time()
        files = np.array(
            glob.glob(os.path.join(dataset_folder_, renderings_folder, dataset, sequence, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder_, segmentations_folder, dataset, sequence, '*.*')))
        segms.sort()

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        run_tracking_on_sequence(config, files, segms, write_folder, None)


if __name__ == "__main__":
    main()

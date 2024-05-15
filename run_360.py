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

sys.path.append('repositories/OSTrack/S2DNet')

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

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        renderings_folder = 'original'
        dataset_folder_ = Path(dataset_folder) / '360photo'

        t0 = time.time()
        files = np.array(
            glob.glob(os.path.join(dataset_folder_, renderings_folder, dataset, sequence, '*.*')))
        files.sort()

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        config.input_frames = len(files)

        run_tracking_on_sequence(config, write_folder)


if __name__ == "__main__":
    main()

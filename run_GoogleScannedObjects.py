import math

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


def main():
    dataset = 'GoogleScannedObjects_default_pose'
    args = parse_args()

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
            'Twinlab_Nitric_Fuel',
            'Squirrel',
            'STACKING_BEAR',
            'Schleich_Allosaurus',
            'Threshold_Ramekin_White_Porcelain',
            'Tag_Dishtowel_Green'
        ]

    for sequence in sequences:
        config = load_config(args.config)

        gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)
        gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
        gt_mesh_path = gt_model_path / Path('meshes/model.obj')
        gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path('gt_tracking_log') / Path(sequence) / \
                           Path('gt_tracking_log.csv')

        experiment_name = args.experiment

        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = gt_mesh_path
        config.gt_track_path = gt_tracking_path
        config.tran_init = 0
        config.rot_init = [0, 0, 0]
        # config['rot_init'] = [-math.pi / 2.0, 0, 0]
        config.sequence = sequence

        write_folder = os.path.join(tmp_folder, experiment_name, dataset, sequence)

        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.makedirs(write_folder)
        os.makedirs(os.path.join(write_folder, 'imgs'))
        shutil.copyfile(os.path.join('prototypes', 'model.mtl'), os.path.join(write_folder, 'model.mtl'))

        renderings_folder = 'renderings'
        segmentations_folder = 'segmentations'
        optical_flows_folder = 'optical_flow'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, renderings_folder, sequence, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, segmentations_folder, sequence, '*.*')))
        segms.sort()
        optical_flows = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, optical_flows_folder, sequence, '*.*')))
        optical_flows.sort()
        # optical_flows = None

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(config, files, segms, write_folder, optical_flows=optical_flows)


if __name__ == "__main__":
    main()

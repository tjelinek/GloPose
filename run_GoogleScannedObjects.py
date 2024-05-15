import glob
import os
import sys
import time

import numpy as np

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args, load_gt_data
from utils import load_config
from pathlib import Path

sys.path.append('repositories/OSTrack/S2DNet')


def main():
    dataset = 'GoogleScannedObjects'
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

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        renderings_folder = 'renderings'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, renderings_folder, sequence, '*.*')))
        files.sort()

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        gt_texture, gt_mesh, gt_rotations, gt_translations = load_gt_data(config)
        run_tracking_on_sequence(config, files, write_folder, gt_texture, gt_mesh, gt_rotations, gt_translations)


if __name__ == "__main__":
    main()

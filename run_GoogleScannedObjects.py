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
    dataset = 'GoogleScannedObjects'
    sequences = ['INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count', 'Squirrel', 'STACKING_BEAR',
                 'Schleich_Allosaurus', 'Threshold_Ramekin_White_Porcelain', 'Tag_Dishtowel_Green']

    for sequence in sequences:
        gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)
        gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
        gt_mesh_path = gt_model_path / Path('meshes/model.obj')
        gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path('gt_tracking_log') / Path(sequence) / \
                           Path('gt_tracking_log.csv')

        args = parse_args(sequence, dataset)

        experiment_name = args.experiment
        config = load_config(args.config)
        config["image_downsample"] = args.perc
        config["tran_init"] = 0
        config["gt_texture"] = gt_texture_path
        config["gt_mesh_prototype"] = gt_mesh_path
        config["gt_tracking_log"] = gt_tracking_path
        config["rot_init"] = [0, 0, 0]

        write_folder = os.path.join(tmp_folder, experiment_name, args.dataset, args.sequence)

        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.makedirs(write_folder)
        os.makedirs(os.path.join(write_folder, 'imgs'))
        shutil.copyfile(os.path.join('prototypes', 'model.mtl'), os.path.join(write_folder, 'model.mtl'))
        config["sequence"] = args.sequence

        renderings_folder = 'renderings'
        segmentations_folder = 'segmentations'
        optical_flows_folder = 'optical_flow'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, renderings_folder, args.sequence, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, segmentations_folder, args.sequence, '*.*')))
        optical_flows = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, optical_flows_folder, args.sequence, '*.*')))
        # optical_flows = None

        segms.sort()
        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(args, config, files, segms, write_folder, optical_flows=optical_flows)


if __name__ == "__main__":
    main()

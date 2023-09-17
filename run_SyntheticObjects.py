import glob
import numpy as np
import os
import shutil
import sys
import time
from pathlib import Path

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config

sys.path.append('OSTrack/S2DNet')


def main():
    dataset = 'SyntheticObjects'
    sequences = [
        'Textured_Sphere_5_y',
        # 'Translating_Textured_Sphere',
        # 'Textured_Sphere_5_xy',
        'Rotating_Translating_Textured_Sphere_5_y',
        'Rotating_Translating_Textured_Sphere_5_xy'
        '8_Colored_Sphere_5_x',
        '6_Colored_Cube_5_z']

    for sequence in sequences:

        if '8_Colored_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = None
        elif 'Textured_Sphere' in sequence:
            gt_mesh_path = Path('prototypes/sphere.obj')
            gt_texture_path = Path('prototypes/tex.png')
        else:
            gt_texture_path = None
            gt_mesh_path = None

        gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path(sequence) / Path('gt_tracking_log') / \
                           Path('gt_tracking_log.csv')

        args = parse_args(sequence, dataset)

        experiment_name = args.experiment
        config = load_config(args.config)
        config["image_downsample"] = args.perc
        config["tran_init"] = 0
        config["gt_texture"] = gt_texture_path
        config["gt_mesh_prototype"] = gt_mesh_path
        config["gt_tracking_log"] = gt_tracking_path

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
            glob.glob(os.path.join(dataset_folder, args.dataset, args.sequence, renderings_folder, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, args.sequence, segmentations_folder, '*.*')))
        segms.sort()
        optical_flows = np.array(
            glob.glob(os.path.join(dataset_folder, args.dataset, args.sequence, optical_flows_folder, '*.*')))
        optical_flows.sort()
        # optical_flows = None

        segms.sort()
        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(args, config, files, segms, write_folder, optical_flows)


if __name__ == "__main__":
    main()

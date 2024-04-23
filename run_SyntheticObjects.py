import glob
import numpy as np
import os
import sys
import time
from pathlib import Path

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config

sys.path.append('OSTrack/S2DNet')


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
        config.sequence = sequence

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        renderings_folder = 'renderings'
        segmentations_folder = 'segmentations'
        optical_flows_folder = 'optical_flow'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, sequence, renderings_folder, '*.*')))
        files.sort()
        segms = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, sequence, segmentations_folder, '*.*')))
        segms.sort()
        optical_flows = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, sequence, optical_flows_folder, '*.*')))
        optical_flows.sort()
        # optical_flows = None

        segms.sort()
        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))
        run_tracking_on_sequence(config, files, segms, write_folder)


if __name__ == "__main__":
    main()

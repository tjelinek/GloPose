import shutil
import sys
import os
import torch
import time
import argparse
from pathlib import Path

from tracker_config import TrackerConfig

sys.path.append('repositories/OSTrack/S2DNet')

from tracking6d import Tracking6D


def run_tracking_on_sequence(config: TrackerConfig, write_folder: Path, gt_texture=None, gt_mesh=None, gt_rotations=None,
                             gt_translations=None):
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)

    write_folder.mkdir(exist_ok=True, parents=True)
    model_3d_dir = write_folder / '3d_model'
    model_3d_dir.mkdir(exist_ok=True, parents=True)

    shutil.copyfile(Path('prototypes') / 'model.mtl', model_3d_dir  / 'model.mtl')

    print('\n\n\n---------------------------------------------------')
    write_folder_path = Path(write_folder)
    print("Running tracking on dataset:", write_folder_path.parent.name)
    print("Sequence:", write_folder_path.name)
    print('---------------------------------------------------\n\n')

    torch.cuda.empty_cache()
    t0 = time.time()

    sfb = Tracking6D(config, write_folder, gt_texture=gt_texture, gt_mesh=gt_mesh,
                     gt_rotations=gt_rotations, gt_translations=gt_translations)
    best_model = sfb.run_tracking()
    print(f'{config.input_frames} epochs took {(time.time() - t0) / 1} seconds, best model loss {best_model["value"]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_deep.py")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()

import shutil
import sys
import os
import torch
import time
import argparse
from pathlib import Path

from auxiliary_scripts.data_utils import load_texture, load_mesh, load_gt_annotations_file
from tracker_config import TrackerConfig

sys.path.append('repositories/OSTrack/S2DNet')

from tracking6d import Tracking6D


def load_gt_data(config: TrackerConfig):
    gt_texture = None
    gt_mesh = None
    gt_rotations = None
    gt_translations = None
    if config.gt_texture_path is not None:
        gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
    if config.gt_mesh_path is not None:
        gt_mesh = load_mesh(Path(config.gt_mesh_path))
    if config.gt_track_path is not None:
        gt_rotations, gt_translations = load_gt_annotations_file(config.gt_track_path)

    return gt_texture, gt_mesh, gt_rotations, gt_translations


def run_tracking_on_sequence(config: TrackerConfig, files, segms, write_folder, gt_texture=None, gt_mesh=None,
                             gt_rotations=None, gt_translations=None):
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)

    os.makedirs(write_folder)
    os.makedirs(os.path.join(write_folder, 'imgs'))
    shutil.copyfile(os.path.join('prototypes', 'model.mtl'), os.path.join(write_folder, 'model.mtl'))

    print('\n\n\n---------------------------------------------------')
    write_folder_path = Path(write_folder)
    print("Running tracking on dataset:", write_folder_path.parent.name)
    print("Sequence:", write_folder_path.name)
    print('---------------------------------------------------\n\n')

    config.input_frames = len(files)
    if config.inc_step == 0:
        config.inc_step = len(files)
    inds = [os.path.splitext(os.path.basename(temp))[0] for temp in segms]
    baseline_dict = dict(zip(inds, segms))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    sfb = Tracking6D(config, device, write_folder, files[0], baseline_dict, gt_texture=gt_texture, gt_mesh=gt_mesh,
                     gt_rotations=gt_rotations, gt_translations=gt_translations)
    best_model = sfb.run_tracking(files)
    print(f'{config.input_frames} epochs took {(time.time() - t0) / 1} seconds, best model loss {best_model["value"]}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_deep.yaml")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()

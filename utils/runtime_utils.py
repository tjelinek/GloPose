import gc
import shutil
import os
import torch
import time
import argparse
from pathlib import Path

from tracker_config import TrackerConfig
from tracking6d import Tracking6D


def run_tracking_on_sequence(config: TrackerConfig, write_folder: Path, **kwargs):
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)

    write_folder.mkdir(exist_ok=True, parents=True)

    print('\n\n\n---------------------------------------------------')
    write_folder_path = Path(write_folder)
    print("Running tracking on dataset:", write_folder_path.parent.name)
    print("Sequence:", write_folder_path.name)
    print('---------------------------------------------------\n\n')

    torch.cuda.empty_cache()
    t0 = time.time()

    sfb = Tracking6D(config, write_folder, **kwargs)
    sfb.run_filtering_with_reconstruction()

    del sfb
    gc.collect()
    print(f'{config.input_frames} epochs took {(time.time() - t0) / 1} seconds.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/base_config.py")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()

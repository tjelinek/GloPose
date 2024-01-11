import sys
import os
import torch
import time
import argparse
from pathlib import Path

from tracker_config import TrackerConfig

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def run_tracking_on_sequence(config: TrackerConfig, files, segms, write_folder, optical_flows=None):
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

    sfb = Tracking6D(config, device, write_folder, files[0], baseline_dict)
    best_model = sfb.run_tracking(files, baseline_dict)
    print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"],
                                                                            (time.time() - t0) / 1,
                                                                            best_model["value"]))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_deep.yaml")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()

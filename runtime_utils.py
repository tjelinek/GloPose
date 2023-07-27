import sys
import os
import torch
import time
import argparse
from pathlib import Path

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def run_tracking_on_sequence(args, config, files, segms, write_folder, optical_flows=None):
    print('\n\n\n---------------------------------------------------')
    write_folder_path = Path(write_folder)
    print("Running tracking on dataset:", write_folder_path.parent.name)
    print("Sequence:", write_folder_path.name)
    print('---------------------------------------------------\n\n')

    if args.length is None:
        args.length = len(files)
    files = files[args.start:args.length:args.skip]
    segms = segms[args.start:args.length:args.skip]
    if optical_flows is not None:
        optical_flows = optical_flows[args.start:args.length:args.skip]
    config["input_frames"] = len(files)
    if config["inc_step"] == 0:
        config["inc_step"] = len(files)
    inds = [os.path.splitext(os.path.basename(temp))[0] for temp in segms]
    baseline_dict = dict(zip(inds, segms))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    sfb = Tracking6D(config, device, write_folder, files[0], baseline_dict)
    best_model = sfb.run_tracking(files, baseline_dict, optical_flows)
    print('{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"],
                                                                            (time.time() - t0) / 1,
                                                                            best_model["value"]))


def parse_args(sequence, dataset, folder_name=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/config_deep.yaml")
    parser.add_argument("--dataset", required=False, default=dataset)
    parser.add_argument("--sequence", required=False, default=sequence)
    parser.add_argument("--start", required=False, default=0)
    parser.add_argument("--length", required=False, default=360)
    parser.add_argument("--skip", required=False, default=1)
    parser.add_argument("--perc", required=False, default=0.25)
    parser.add_argument("--folder_name", required=False, default=folder_name)
    parser.add_argument("--experiment", required=False, default='')  # Experiment name
    return parser.parse_args()

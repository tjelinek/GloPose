import sys
import os
import torch
import time
from main_settings import dataset_folder

sys.path.append('OSTrack/S2DNet')

from tracking6d import Tracking6D


def run_tracking_on_sequence(args, config, files, segms, write_folder):
    if args.length is None:
        args.length = len(files)
    files = files[args.start:args.length:args.skip]
    segms = segms[args.start:args.length:args.skip]
    config["input_frames"] = len(files)
    if config["inc_step"] == 0:
        config["inc_step"] = len(files)
    inds = [os.path.splitext(os.path.basename(temp))[0] for temp in segms]
    baseline_dict = dict(zip(inds, segms))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    sfb = Tracking6D(config, device, write_folder, files[0], baseline_dict)
    best_model = sfb.run_tracking(files, baseline_dict)
    print(
        '{:4d} epochs took {:.2f} seconds, best model loss {:.4f}'.format(config["iterations"],
                                                                          (time.time() - t0) / 1,
                                                                          best_model["value"]))

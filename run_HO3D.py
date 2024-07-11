import glob
import numpy as np
import os
import sys
import time
from pathlib import Path

from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from auxiliary_scripts.data_utils import load_gt_data
from utils import load_config

sys.path.append('repositories/OSTrack/S2DNet')



def main():
    dataset = 'SyntheticObjects'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [

            ]

    for sequence in sequences:
        config = load_config(args.config)

        gt_texture_path = None
        gt_mesh_path = None

        gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path(sequence) / Path('gt_tracking_log') / \
                           Path('gt_tracking_log.csv')

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = gt_mesh_path
        config.gt_track_path = None
        config.sequence = sequence

        # config.camera_position = (-5.0, -5.0, -5.0)
        # config.camera_up = (0, 0, 1)

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        renderings_folder = 'renderings'

        t0 = time.time()

        files = np.array(
            glob.glob(os.path.join(dataset_folder, dataset, sequence, renderings_folder, '*.*')))
        files.sort()
        config.input_frames = len(files)

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        gt_texture, gt_mesh, gt_rotations, gt_translations = load_gt_data(config)

        # import torch
        # gt_rotations = torch.tensor(
        #     [[[0.0000, 0.0000, 0.0000],
        #       [0.1078, 0.0494, 0.0938],
        #       [0.2155, 0.0988, 0.1876],
        #       [0.3233, 0.1481, 0.2814],
        #       [0.4310, 0.1975, 0.3752],
        #       [0.5388, 0.2469, 0.4690],
        #       [0.6465, 0.2963, 0.5628],
        #       [0.7543, 0.3456, 0.6566],
        #       [0.8620, 0.3950, 0.7504],
        #       [0.9698, 0.4444, 0.8442],
        #       [1.0775, 0.4938, 0.9380],
        #       [1.1853, 0.5431, 1.0317],
        #       [1.2930, 0.5925, 1.1255],
        #       [1.4008, 0.6419, 1.2193],
        #       [1.5086, 0.6913, 1.3131],
        #       [1.6163, 0.7406, 1.4069],
        #       [1.7241, 0.7900, 1.5007],
        #       [1.8318, 0.8394, 1.5945],
        #       [1.9396, 0.8888, 1.6883],
        #       [2.0473, 0.9381, 1.7821],
        #       [2.1551, 0.9875, 1.8759],
        #       [-2.2164, -1.0156, -1.9293],
        #       [-2.1087, -0.9662, -1.8355],
        #       [-2.0009, -0.9169, -1.7417],
        #       [-1.8931, -0.8675, -1.6479],
        #       [-1.7854, -0.8181, -1.5541],
        #       [-1.6776, -0.7687, -1.4603],
        #       [-1.5699, -0.7194, -1.3665],
        #       [-1.4621, -0.6700, -1.2727],
        #       [-1.3544, -0.6206, -1.1789],
        #       [-1.2466, -0.5712, -1.0851],
        #       [-1.1389, -0.5219, -0.9913],
        #       [-1.0311, -0.4725, -0.8975],
        #       [-0.9234, -0.4231, -0.8038],
        #       [-0.8156, -0.3737, -0.7100],
        #       [-0.7079, -0.3244, -0.6162],
        #       [-0.6001, -0.2750, -0.5224],
        #       [-0.4924, -0.2256, -0.4286],
        #       [-0.3846, -0.1762, -0.3348],
        #       [-0.2768, -0.1269, -0.2410],
        #       [-0.1691, -0.0775, -0.1472],
        #       [-0.0613, -0.0281, -0.0534],
        #       [0.0464, 0.0213, 0.0404],
        #       [0.1542, 0.0706, 0.1342],
        #       [0.2619, 0.1200, 0.2280],
        #       [0.3697, 0.1694, 0.3218],
        #       [0.4774, 0.2188, 0.4156],
        #       [0.5852, 0.2681, 0.5094],
        #       [0.6929, 0.3175, 0.6032],
        #       [0.8007, 0.3669, 0.6970],
        #       [0.9084, 0.4163, 0.7908],
        #       [1.0162, 0.4656, 0.8846],
        #       [1.1240, 0.5150, 0.9784],
        #       [1.2317, 0.5644, 1.0722],
        #       [1.3395, 0.6138, 1.1659],
        #       [1.4472, 0.6632, 1.2597],
        #       [1.5550, 0.7125, 1.3535],
        #       [1.6627, 0.7619, 1.4473],
        #       [1.7705, 0.8113, 1.5411],
        #       [1.8782, 0.8607, 1.6349],
        #       [1.9860, 0.9100, 1.7287],
        #       [2.0937, 0.9594, 1.8225],
        #       [2.2015, 1.0088, 1.9163],
        #       [-2.1700, -0.9943, -1.8889],
        #       [-2.0622, -0.9450, -1.7951],
        #       [-1.9545, -0.8956, -1.7013],
        #       [-1.8467, -0.8462, -1.6075],
        #       [-1.7390, -0.7968, -1.5137],
        #       [-1.6312, -0.7475, -1.4199],
        #       [-1.5235, -0.6981, -1.3261],
        #       [-1.4157, -0.6487, -1.2323],
        #       [-1.3080, -0.5993, -1.1385],
        #       [-1.2002, -0.5500, -1.0447]]]).cuda()

        # import torch
        # from dataset_generators import scenarios
        # gt_rotations_np = np.deg2rad(np.stack(scenarios.generate_rotations_xyz(5).rotations, axis=0))
        # gt_rotations = torch.from_numpy(gt_rotations_np).unsqueeze(0).cuda().to(torch.float32)
        # gt_translations = torch.zeros_like(gt_rotations).unsqueeze(0)
        #
        # gt_rotations[:, :, 0] = gt_rotations[..., 1] * -0.6
        # gt_rotations[:, :, 2] = gt_rotations[..., 1] * 2.0

        # gt_rotations = modify_rotations(gt_rotations)

        # gt_rotations = (gt_rotations * 3) % (2 * np.pi)

        run_tracking_on_sequence(config, write_folder, gt_texture, gt_mesh, gt_rotations, gt_translations)


if __name__ == "__main__":
    main()

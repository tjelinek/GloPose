import csv
from pathlib import Path

import torch
from kornia.geometry import Quaternion, Se3

from onboarding_pipeline import OnboardingPipeline
from utils.dataset_sequences import get_tum_rgbd_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args


def main():
    dataset = 'TUM_RGBD'
    args = parse_args()
    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = get_tum_rgbd_sequences(config.default_data_folder / 'SLAM' / 'tum_rgbd')

    for sequence in sequences:
        config = load_config(args.config)

        if config.gt_flow_source == 'GenerateSynthetic':
            exit()

        experiment_name = args.experiment
        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 1.0

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = config.default_results_folder / experiment_name / dataset / sequence

        sequence_folder = config.default_data_folder / 'SLAM' / 'tum_rgbd' / sequence

        image_paths = []

        gt_images_list_txt_path = sequence_folder / 'rgb.txt'
        with open(gt_images_list_txt_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            for row in csv_reader:
                if not row or row[0].startswith('#'):
                    continue

                if len(row) > 1:
                    image_paths.append(sequence_folder / Path(row[1]))

        sequence_length = len(image_paths)

        gt_txt_path = sequence_folder / 'groundtruth.txt'

        gt_cam_ts = []
        gt_cam_quats = []
        with open(gt_txt_path, 'r') as file:
            csv_reader = csv.reader(file, delimiter=' ')
            for row in csv_reader:
                if not row or row[0].startswith('#') or len(row) != 8:
                    continue

                tx, ty, tz = [float(x) for x in row[1:4]]
                qx, qy, qz, qw = [float(x) for x in row[4:]]

                gt_cam_ts.append(torch.tensor([tx, ty, tz]).to(config.device))
                gt_cam_quats.append(torch.tensor([qw, qx, qy, qz]).to(config.device))

        gt_cam_t = torch.stack(gt_cam_ts)
        gt_cam_quat = torch.stack(gt_cam_quats)

        gt_Se3_world_to_cam = Se3(Quaternion(gt_cam_quat), gt_cam_t)

        # TILL HERE FINISHED
        gt_Se3_cam_to_world = gt_Se3_world_to_cam.inverse()
        Se3_obj_1_to_cam = gt_Se3_cam_to_world[[0]].inverse()

        config.camera_extrinsics = Se3_obj_1_to_cam.inverse().matrix().squeeze().numpy(force=True)
        config.input_frames = sequence_length
        config.segmentation_provider = 'whites'
        config.frame_provider = 'precomputed'
        config.large_images_results_write_frequency = 10

        tracker = OnboardingPipeline(config, write_folder, input_images=image_paths, gt_Se3_cam2obj=gt_Se3_cam_to_world)
        tracker.run_pipeline()

        exit()


if __name__ == "__main__":
    main()

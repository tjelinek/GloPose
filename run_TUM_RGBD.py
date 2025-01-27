import csv

import torch
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from kornia.geometry import Quaternion, Se3

from utils.image_utils import get_nth_video_frame
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i
from utils.runtime_utils import parse_args
from tracker6d import run_tracking_on_sequence
from utils.general import load_config


def main():
    dataset = 'TUM_RGBD'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'rgbd_dataset_freiburg1_360',
            'rgbd_dataset_freiburg1_desk',
            'rgbd_dataset_freiburg1_desk2',
            'rgbd_dataset_freiburg1_floor',
            'rgbd_dataset_freiburg1_plant',
            'rgbd_dataset_freiburg1_room',
            'rgbd_dataset_freiburg1_rpy',
            'rgbd_dataset_freiburg1_teddy',
            'rgbd_dataset_freiburg1_xyz',
            'rgbd_dataset_freiburg2_360_hemisphere',
            'rgbd_dataset_freiburg2_360_kidnap',
            'rgbd_dataset_freiburg2_coke',
            'rgbd_dataset_freiburg2_desk',
            'rgbd_dataset_freiburg2_dishes',
            'rgbd_dataset_freiburg2_flowerbouquet',
            'rgbd_dataset_freiburg2_flowerbouquet_brownbackground',
            'rgbd_dataset_freiburg2_large_no_loop',
            'rgbd_dataset_freiburg2_large_with_loop',
            'rgbd_dataset_freiburg2_metallic_sphere',
            'rgbd_dataset_freiburg2_metallic_sphere2',
            'rgbd_dataset_freiburg2_pioneer_360',
            'rgbd_dataset_freiburg2_pioneer_slam',
            'rgbd_dataset_freiburg2_pioneer_slam2',
            'rgbd_dataset_freiburg2_pioneer_slam3',
            'rgbd_dataset_freiburg2_rpy',
            'rgbd_dataset_freiburg2_xyz',
            'rgbd_dataset_freiburg3_cabinet',
            'rgbd_dataset_freiburg3_large_cabinet',
            'rgbd_dataset_freiburg3_long_office_household',
            'rgbd_dataset_freiburg3_sitting_halfsphere',
            'rgbd_dataset_freiburg3_sitting_rpy',
            'rgbd_dataset_freiburg3_sitting_static',
            'rgbd_dataset_freiburg3_sitting_xyz',
            'rgbd_dataset_freiburg3_teddy',
            'rgbd_dataset_freiburg3_walking_halfsphere',
            'rgbd_dataset_freiburg3_walking_rpy',
            'rgbd_dataset_freiburg3_walking_static',
            'rgbd_dataset_freiburg3_walking_xyz',
        ]

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

        config.write_folder = write_folder
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
        Se3_obj_1_to_obj_i = Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(gt_Se3_cam_to_world)
        Se3_obj_1_to_cam = gt_Se3_cam_to_world[[0]].inverse()

        config.camera_extrinsics = Se3_obj_1_to_cam.inverse().matrix().squeeze().numpy(force=True)
        config.input_frames = sequence_length
        config.segmentation_provider = 'whites'
        config.frame_provider = 'precomputed'

        run_tracking_on_sequence(config, write_folder, gt_texture=None, gt_mesh=None,
                                 gt_obj_1_to_obj_i_Se3=Se3_obj_1_to_obj_i, images_paths=image_paths,
                                 gt_Se3_obj_1_to_cam=Se3_obj_1_to_cam)

        exit()


if __name__ == "__main__":
    main()

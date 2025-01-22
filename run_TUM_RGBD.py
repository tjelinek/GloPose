import pickle

import torch
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from kornia.geometry import Quaternion, Se3

from utils.image_utils import get_nth_video_frame
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i
from utils.runtime_utils import run_tracking_on_sequence, parse_args
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
        sequence_folder = config.default_data_folder / 'BEHAVE' / 'train'

        video_path = sequence_folder / f'{sequence}.mp4'
        gt_pkl_name = sequence_folder / f'{sequence}_gt.pkl'
        object_seg_video_path = sequence_folder / f'{sequence}_mask_obj.mp4'

        with open(gt_pkl_name, "rb") as f:
            gt_annotations = pickle.load(f)
            cam_to_obj_rotations = torch.from_numpy(gt_annotations['obj_rot']).to(config.device)
            cam_to_obj_translations = torch.from_numpy(gt_annotations['obj_trans']).to(config.device)
            sequence_length = cam_to_obj_rotations.shape[0]

        Se3_cam_to_obj = Se3(Quaternion.from_matrix(cam_to_obj_rotations), cam_to_obj_translations)
        Se3_obj_1_to_obj_i = Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(Se3_cam_to_obj)
        Se3_obj_1_to_cam = Se3_cam_to_obj[[0]].inverse()

        config.camera_extrinsics = Se3_obj_1_to_cam.inverse().matrix().squeeze().numpy(force=True)
        config.input_frames = sequence_length
        config.segmentation_provider = 'SAM2'
        config.frame_provider = 'precomputed'

        first_image = get_nth_video_frame(video_path, 0, mode='rgb')
        first_segment = get_nth_video_frame(object_seg_video_path, 0, mode='grayscale')

        first_segment_resized = first_segment.resize(first_image.size, Image.NEAREST)

        transform = transforms.ToTensor()
        first_segment_tensor = transform(first_segment_resized).squeeze()
        first_image_tensor = transform(first_image).squeeze()

        run_tracking_on_sequence(config, write_folder, gt_texture=None, gt_mesh=None,
                                 gt_obj_1_to_obj_i_Se3=Se3_obj_1_to_obj_i,
                                 video_path=video_path, segmentation_video_path=object_seg_video_path,
                                 initial_segmentation=first_segment_tensor, initial_image=first_image_tensor,
                                 gt_Se3_obj_1_to_cam=Se3_obj_1_to_cam)

        exit()


if __name__ == "__main__":
    main()

import pickle

import torch
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from kornia.geometry import Quaternion, Se3

from tracker6d import Tracker6D
from utils.dataset_sequences import get_behave_sequences
from utils.image_utils import get_nth_video_frame
from utils.runtime_utils import parse_args
from utils.general import load_config


def main():
    dataset = 'BEHAVE'
    args = parse_args()
    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = get_behave_sequences(config.default_data_folder / 'BEHAVE' / 'train')

    sequence = sequences[0]
    config = load_config(args.config)

    if config.gt_flow_source == 'GenerateSynthetic':
        exit()

    experiment_name = args.experiment
    config.experiment_name = experiment_name
    config.sequence = sequence
    config.dataset = dataset
    config.image_downsample = 1.0
    config.large_images_results_write_frequency = 20

    config.skip_indices *= 10

    if args.output_folder is not None:
        write_folder = Path(args.output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / sequence

    sequence_folder = config.default_data_folder / 'BEHAVE' / 'train'

    video_path = sequence_folder / f'{sequence}.mp4'
    gt_pkl_name = sequence_folder / f'{sequence}_gt.pkl'
    object_seg_video_path = sequence_folder / f'{sequence}_mask_obj.mp4'

    with open(gt_pkl_name, "rb") as f:
        gt_annotations = pickle.load(f)
        cam_to_obj_rotations = torch.from_numpy(gt_annotations['obj_rot']).to(config.device)
        cam_to_obj_translations = torch.from_numpy(gt_annotations['obj_trans']).to(config.device)
        sequence_length = cam_to_obj_rotations.shape[0]

    gt_Se3_cam2obj = Se3(Quaternion.from_matrix(cam_to_obj_rotations), cam_to_obj_translations)
    gt_Se3_world2cam = gt_Se3_cam2obj.inverse()
    gt_Se3_cam2obj = {i: gt_Se3_cam2obj[i] for i in range(sequence_length)}
    gt_Se3_world2cam = {i: gt_Se3_world2cam[i] for i in range(sequence_length)}

    Se3_obj_1_to_cam = gt_Se3_cam2obj[0].inverse()

    config.camera_extrinsics = Se3_obj_1_to_cam.inverse().matrix().numpy(force=True)
    config.input_frames = sequence_length
    config.segmentation_provider = 'SAM2'
    config.frame_provider = 'precomputed'

    first_image = get_nth_video_frame(video_path, 0, mode='rgb')
    first_segment = get_nth_video_frame(object_seg_video_path, 0, mode='grayscale')

    first_segment_resized = first_segment.resize(first_image.size, Image.NEAREST)

    transform = transforms.ToTensor()
    first_segment_tensor = transform(first_segment_resized).squeeze()
    first_image_tensor = transform(first_image).squeeze()

    tracker = Tracker6D(config, write_folder, video_path, gt_Se3_cam2obj=gt_Se3_cam2obj, gt_Se3_world2cam=gt_Se3_world2cam,
                        initial_segmentation=first_segment_tensor)
    tracker.run_pipeline()


if __name__ == "__main__":
    main()

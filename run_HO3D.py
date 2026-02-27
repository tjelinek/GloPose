import time
from pathlib import Path

import numpy as np
import torch
from kornia.geometry import Quaternion, Se3, PinholeCamera

from data_providers.frame_provider import PrecomputedSegmentationProvider
from eval.eval_onboarding import evaluate_onboarding
from onboarding_pipeline import OnboardingPipeline
from utils.dataset_sequences import get_ho3d_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'HO3D'
    split = 'train'  # 'evaluation' or 'train'

    args = parse_args()
    config = load_config(args.config)
    train_sequences, test_sequences = get_ho3d_sequences(config.ho3d_data_folder)
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        if split == 'train':
            sequences = train_sequences[0:1]
        else:
            sequences = test_sequences[3:4]

    for sequence in sequences:

        if args.sequences is not None and len(args.sequences) > 0:
            if sequence in train_sequences:
                split = 'train'
            else:
                split = 'evaluation'

        with exception_logger():
            config = load_config(args.config)

            config.skip_indices *= 10

            if config.gt_flow_source == 'GenerateSynthetic':
                exit()

            experiment_name = args.experiment
            config.experiment_name = experiment_name
            config.sequence = sequence
            config.object_id = sequence
            config.dataset = dataset
            config.image_downsample = 1.0
            config.frame_provider_config.erode_segmentation = True

            if args.output_folder is not None:
                write_folder = Path(args.output_folder) / dataset / sequence
            else:
                write_folder = config.default_results_folder / experiment_name / dataset / sequence

            t0 = time.time()

            sequence_folder = config.ho3d_data_folder / split / sequence
            image_folder = sequence_folder / 'rgb'
            segmentation_folder = sequence_folder / 'seg'
            if not segmentation_folder.exists():
                segmentation_folder = sequence_folder / 'segmentation_rendered'
            depth_folder = sequence_folder / 'depth'
            meta_folder = sequence_folder / 'meta'

            gt_images_list = [file for file in sorted(image_folder.iterdir()) if file.is_file()]
            gt_segmentations_list = [file for file in sorted(segmentation_folder.iterdir()) if file.is_file()]
            gt_depths_list = [file for file in sorted(depth_folder.iterdir()) if file.is_file()]

            gt_translations = []
            gt_rotations = []
            cam_intrinsics_list = []

            nones = set()
            for i, file in enumerate(sorted(meta_folder.iterdir())):
                if file.suffix not in ['.pkl']:
                    continue

                data = np.load(file, allow_pickle=True)
                data_dict = {key: data[key] for key in data}

                cam_K = data_dict['camMat']
                gt_rot = data_dict['objRot']
                gt_trans = data_dict['objTrans']

                if cam_K is None or gt_rot is None or gt_trans is None:
                    nones.add(i)
                    continue

                cam_intrinsics_list.append(data_dict['camMat'])
                gt_rotations.append(data_dict['objRot'].squeeze())
                gt_translations.append(data_dict['objTrans'])

            eerr0 = set(i for i in range(len(gt_rotations)) if len(gt_rotations[i].shape) < 1)
            eert0 = set(i for i in range(len(gt_rotations)) if len(gt_translations[i].shape) < 1)

            invalid_indices = eerr0 | eert0 | nones
            valid_indices = [i for i in range(len(gt_rotations)) if i not in invalid_indices]

            # Filter the lists to include only valid elements
            filtered_gt_rotations = [gt_rotations[i] for i in valid_indices]
            filtered_gt_translations = [gt_translations[i] for i in valid_indices]
            gt_images_list = [gt_images_list[i] for i in valid_indices]
            gt_segmentations_list = [gt_segmentations_list[i] for i in valid_indices]

            config.input_frames = len(gt_images_list)

            cam2obj_rotations = torch.from_numpy(np.array(filtered_gt_rotations)).to(config.device)
            cam2obj_translations = torch.from_numpy(np.array(filtered_gt_translations)).to(config.device)
            cam2obj_translations *= 1000.0  # Scaling from mm to m

            Se3_cam2obj = Se3(Quaternion.from_axis_angle(cam2obj_rotations), cam2obj_translations)
            Se3_obj2cam = Se3_cam2obj.inverse()
            Se3_cam2obj_dict = {i: Se3_cam2obj[i] for i in range(config.input_frames)}
            Se3_obj2cam_dict = {i: Se3_obj2cam[i] for i in range(config.input_frames)}

            print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

            T_obj_to_cam = Se3_cam2obj.inverse().matrix().squeeze()

            config.camera_intrinsics = cam_intrinsics_list[0]
            config.camera_extrinsics = T_obj_to_cam.numpy(force=True)


            config.segmentation_provider = 'SAM2'
            config.frame_provider = 'precomputed'

            first_segment_tensor = \
                PrecomputedSegmentationProvider.get_initial_segmentation(gt_images_list, gt_segmentations_list,
                                                                         segmentation_channel=1)

            image_h, image_w = (torch.tensor(int(s * config.image_downsample))[None].to(config.device)
                                for s in first_segment_tensor.shape[-2:])

            gt_pinhole_params = {}
            for i in range(config.input_frames):
                obj2cam = Se3_obj2cam_dict[i].matrix()[None]
                cam_K_tensor = torch.from_numpy(cam_intrinsics_list[i])[None].to(config.device)
                gt_pinhole_params[i] = PinholeCamera(cam_K_tensor, obj2cam, image_h, image_w)

            tracker = OnboardingPipeline(config, write_folder, input_images=gt_images_list,
                                         gt_Se3_cam2obj=Se3_cam2obj_dict,
                                         gt_Se3_world2cam=Se3_obj2cam_dict, gt_pinhole_params=gt_pinhole_params,
                                         input_segmentations=gt_segmentations_list, depth_paths=gt_depths_list,
                                         initial_segmentation=first_segment_tensor)
            view_graph = tracker.run_pipeline()
            evaluate_onboarding(view_graph, Se3_obj2cam_dict, config, write_folder)


if __name__ == "__main__":
    main()

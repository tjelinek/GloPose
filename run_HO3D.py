import numpy as np
import torch
import time
from pathlib import Path

from kornia.geometry import Quaternion, Se3

from utils.data_utils import get_initial_image_and_segment
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D
from utils.general import load_config


def main():
    dataset = 'HO3D'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'ABF10', 'BB10', 'GPMF10', 'GSF10', 'MC1', 'MDF10', 'ND2', 'ShSu12', 'SiBF12', 'SM3', 'SMu41',
            'ABF11', 'BB11', 'GPMF11', 'GSF11', 'MC2', 'MDF11', 'SB10', 'ShSu13', 'SiBF13', 'SM4', 'SMu42',
            'ABF12', 'BB12', 'GPMF12', 'GSF12', 'MC4', 'MDF12', 'SB12', 'ShSu14', 'SiBF14', 'SM5', 'SS1',
            'ABF13', 'BB13', 'GPMF13', 'GSF13', 'MC5', 'MDF13', 'SB14', 'SiBF10', 'SiS1', 'SMu1', 'SS2',
            'ABF14', 'BB14', 'GPMF14', 'GSF14', 'MC6', 'MDF14', 'ShSu10', 'SiBF11', 'SM2', 'SMu40', 'SS3',

        ]

    sequence = sequences[0]
    config = load_config(args.config)
    config.skip_indices = 15

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
    t0 = time.time()

    sequence_folder = config.default_data_folder / 'HO3D' / 'train' / sequence
    image_folder = sequence_folder / 'rgb'
    segmentation_folder = sequence_folder / 'seg'
    depth_folder = sequence_folder / 'depth'
    meta_folder = sequence_folder / 'meta'

    gt_images_list = [file for file in sorted(image_folder.iterdir()) if file.is_file()]
    gt_segmentations_list = [file for file in sorted(segmentation_folder.iterdir()) if file.is_file()]
    gt_depths_list = [file for file in sorted(depth_folder.iterdir()) if file.is_file()]

    gt_translations = []
    gt_rotations = []
    cam_intrinsics_list = []
    for i, file in enumerate(sorted(meta_folder.iterdir())):
        if file.suffix != '.npz':
            continue

        data = np.load(file, allow_pickle=True)
        data_dict = {key: data[key] for key in data}
        cam_intrinsics_list.append(data_dict['camMat'])
        gt_rotations.append(data_dict['objRot'].squeeze())
        gt_translations.append(data_dict['objTrans'])

    eerr0 = set(i for i in range(len(gt_rotations)) if len(gt_rotations[i].shape) < 1)
    eert0 = set(i for i in range(len(gt_rotations)) if len(gt_translations[i].shape) < 1)

    valid_indices = [i for i in range(len(gt_rotations)) if i not in (eerr0 | eert0)]

    # Filter the lists to include only valid elements
    filtered_gt_rotations = [gt_rotations[i] for i in valid_indices]
    filtered_gt_translations = [gt_translations[i] for i in valid_indices]
    gt_images_list = [gt_images_list[i] for i in valid_indices]
    gt_segmentations_list = [gt_segmentations_list[i] for i in valid_indices]

    config.input_frames = len(gt_images_list)

    cam2obj_rotations = torch.from_numpy(np.array(filtered_gt_rotations)).to(config.device)
    cam2obj_translations = torch.from_numpy(np.array(filtered_gt_translations)).to(config.device)

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

    first_image_tensor, first_segment_tensor = get_initial_image_and_segment(gt_images_list, gt_segmentations_list,
                                                                             segmentation_channel=1)

    sfb = Tracker6D(config, write_folder, images_paths=gt_images_list, gt_Se3_cam2obj=Se3_cam2obj_dict,
                    gt_Se3_world2cam=Se3_obj2cam_dict, segmentation_paths=gt_segmentations_list,
                    initial_image=first_image_tensor, initial_segmentation=first_segment_tensor,
                    depth_paths=gt_depths_list)
    sfb.run_filtering_with_reconstruction()


if __name__ == "__main__":
    main()

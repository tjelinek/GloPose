import numpy as np
import torch
import time
import torchvision.transforms as transforms
from pathlib import Path

from PIL import Image
from kornia.geometry import Quaternion, Se3

from utils.runtime_utils import run_tracking_on_sequence, parse_args
from utils.general import load_config


def main():
    dataset = 'HO3D'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'ABF10',  'BB10',  'GPMF10',  'GSF10',  'MC1',  'MDF10',  'ND2',     'ShSu12',  'SiBF12',  'SM3',    'SMu41',
            'ABF11',  'BB11',  'GPMF11',  'GSF11',  'MC2',  'MDF11',  'SB10',    'ShSu13',  'SiBF13',  'SM4',    'SMu42',
            'ABF12',  'BB12',  'GPMF12',  'GSF12',  'MC4',  'MDF12',  'SB12',    'ShSu14',  'SiBF14',  'SM5',    'SS1',
            'ABF13',  'BB13',  'GPMF13',  'GSF13',  'MC5',  'MDF13',  'SB14',    'SiBF10',  'SiS1',    'SMu1',   'SS2',
            'ABF14',  'BB14',  'GPMF14',  'GSF14',  'MC6',  'MDF14',  'ShSu10',  'SiBF11',  'SM2',     'SMu40',  'SS3',

        ]

    skip_indices = 15

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
        t0 = time.time()

        sequence_folder = config.default_data_folder / 'HO3D' / 'train' / sequence
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'seg'
        meta_folder = sequence_folder / 'meta'

        gt_images_list = [file for file in sorted(image_folder.iterdir()) if file.is_file()]
        gt_segmentations_list = [file for file in sorted(segmentation_folder.iterdir()) if file.is_file()]

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
        valid_indices = valid_indices[::skip_indices]

        # Filter the lists to include only valid elements
        filtered_gt_rotations = [gt_rotations[i] for i in valid_indices]
        filtered_gt_translations = [gt_translations[i] for i in valid_indices]
        gt_images_list = [gt_images_list[i] for i in valid_indices]
        gt_segmentations_list = [gt_segmentations_list[i] for i in valid_indices]

        config.input_frames = len(gt_images_list)

        rotations_array = torch.from_numpy(np.array(filtered_gt_rotations)).cuda()
        translations_array = torch.from_numpy(np.array(filtered_gt_translations)).cuda()

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        quat_frame1 = Quaternion.from_axis_angle(torch.from_numpy(gt_rotations[0])[None])
        trans_frame1 = torch.from_numpy(gt_translations[0])[None]
        Se3_cam_to_obj = Se3(quat_frame1, trans_frame1)
        T_cam_to_obj = Se3_cam_to_obj.matrix().squeeze()

        config.camera_intrinsics = cam_intrinsics_list[0]
        config.camera_extrinsics = T_cam_to_obj.numpy(force=True)

        config.segmentation_provider = 'SAM2'
        config.frame_provider = 'precomputed'

        first_segment = Image.open(gt_segmentations_list[0])
        first_image = Image.open(gt_images_list[0])

        first_segment_resized = first_segment.resize(first_image.size, Image.NEAREST)

        transform = transforms.ToTensor()
        first_segment_tensor = transform(first_segment_resized)[1].squeeze()  # Green channel is the obj segmentation
        first_image_tensor = transform(first_image).squeeze()

        run_tracking_on_sequence(config, write_folder, gt_texture=None, gt_mesh=None,
                                 gt_cam_to_obj_rotations=rotations_array,
                                 gt_cam_to_obj_translations=translations_array, images_paths=gt_images_list,
                                 segmentation_paths=gt_segmentations_list, initial_segmentation=first_segment_tensor,
                                 initial_image=first_image_tensor)

        exit()

if __name__ == "__main__":
    main()

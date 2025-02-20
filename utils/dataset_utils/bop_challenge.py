import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera


def get_pinhole_params(json_file_path) -> List[PinholeCamera]:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    pinhole_cameras = []
    for key, value in json_data.items():
        cam_K = torch.tensor(value['cam_K']).view(3, 3)

        cam_R_w2c = torch.tensor(value['cam_R_w2c']).view(3, 3)
        cam_t_w2c = torch.tensor(value['cam_t_w2c'])

        width = torch.tensor(value['width'])
        height = torch.tensor(value['height'])

        cam_w2c_Se3 = Se3(Quaternion.from_matrix(cam_R_w2c), cam_t_w2c)

        pinhole_camera = PinholeCamera(cam_K.unsqueeze(0), cam_w2c_Se3.matrix().unsqueeze(0),
                                       height.unsqueeze(0), width.unsqueeze(0))

        pinhole_cameras.append(pinhole_camera)

    return pinhole_cameras


def read_obj2cam_Se3_from_gt(pose_json_path, device: str) -> Dict[int, Dict[int, Se3]]:
    dict_gt_Se3_obj2cam = defaultdict(dict)
    with open(pose_json_path, 'r') as file:
        pose_json = json.load(file)
        for frame, data in pose_json.items():
            frame = int(frame)
            for entry in data:
                obj_id = entry['obj_id']
                R_obj_to_cam = entry['cam_R_m2c']
                R_m2c = torch.tensor(np.array(R_obj_to_cam).reshape(3, 3), device=device)

                cam_t_m2c = entry['cam_t_m2c']
                t_m2c = torch.tensor(cam_t_m2c, device=device)

                gt_Se3_obj2cam = Se3(Quaternion.from_matrix(R_m2c), t_m2c).to(torch.float32)
                dict_gt_Se3_obj2cam[obj_id][frame] = gt_Se3_obj2cam

    return dict_gt_Se3_obj2cam


def load_gt_images_and_segmentations(image_folder: Path, segmentation_folder: Path, object_id: int = 1):
    """Load ground truth images and segmentation files, filtering by object ID."""
    object_id_str = f"{object_id - 1:06d}"  # Ensure it's a zero-padded 6-digit string

    gt_segs = {
        int(file.stem.split('_')[0]): file
        for file in sorted(segmentation_folder.iterdir())
        if file.stem.endswith(object_id_str)  # Dynamically filter by object ID
    }

    gt_images = {
        int(file.stem): file
        for file in sorted(image_folder.iterdir())
        if file.is_file()
    }

    return gt_images, gt_segs


def get_sequence_folder(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, onboarding_type: str = None, direction: str = None):
    """Returns the sequence folder path based on sequence type and onboarding type."""
    if sequence_type == 'onboarding':
        if onboarding_type not in ['dynamic', 'static']:
            raise ValueError(f'Unknown onboarding type {onboarding_type}')

        if onboarding_type == 'dynamic':
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / sequence
        elif onboarding_type == 'static' and direction:
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / f'{sequence}_{direction}'

    elif sequence_type in ['test', 'val', 'train']:
        return bop_folder / dataset / sequence_type / sequence

    raise ValueError(f'Unknown sequence type: {sequence_type}')


def get_bop_images_and_segmentations(bop_folder, dataset, sequence, sequence_type, onboarding_type):
    """Loads images and segmentations from BOP dataset based on sequence type."""
    sequence_starts = [0]

    if sequence_type == 'onboarding' and onboarding_type == 'static':
        sequence_folder_down = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, "down")
        sequence_folder_up = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, "up")

        gt_images, gt_segs = load_gt_images_and_segmentations(sequence_folder_down / 'rgb', sequence_folder_down / 'mask_visib')
        gt_images2, gt_segs2 = load_gt_images_and_segmentations(sequence_folder_up / 'rgb', sequence_folder_up / 'mask_visib')

        sequence_starts.append(len(gt_images))

        for frame in gt_images2.keys():
            gt_images[len(gt_images) + frame] = gt_images2[frame]

        for frame in gt_segs2.keys():
            gt_segs[len(gt_segs) + frame] = gt_segs2[frame]

    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'
        gt_images, gt_segs = load_gt_images_and_segmentations(image_folder, segmentation_folder)

    return gt_images, gt_segs, sequence_starts


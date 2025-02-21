import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera


def get_pinhole_params(json_file_path) -> Dict[int, PinholeCamera]:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    pinhole_cameras = {}
    for frame_str, value in json_data.items():
        frame_int = int(frame_str)
        frame_data = json_data[frame_str]
        cam_K = torch.tensor(frame_data['cam_K']).view(3, 3)

        if 'cam_R_w2c' in frame_data and 'cam_t_w2c' in frame_data:
            cam_R_w2c = torch.tensor(frame_data['cam_R_w2c']).view(3, 3)
            cam_t_w2c = torch.tensor(frame_data['cam_t_w2c'])
            cam_w2c_Se3 = Se3(Quaternion.from_matrix(cam_R_w2c), cam_t_w2c)
        else:
            cam_w2c_Se3 = Se3.identity()

        width = torch.tensor(frame_data['width'])
        height = torch.tensor(frame_data['height'])

        pinhole_camera = PinholeCamera(cam_K.unsqueeze(0), cam_w2c_Se3.matrix().unsqueeze(0),
                                       height.unsqueeze(0), width.unsqueeze(0))

        pinhole_cameras[frame_int] = pinhole_camera

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


def get_sequence_folder(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, onboarding_type: str = None,
                        direction: str = None):
    """Returns the sequence folder path based on sequence type and onboarding type."""
    if sequence_type == 'onboarding':

        if onboarding_type == 'dynamic':
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / sequence
        elif onboarding_type == 'static' and direction in ['up', 'down']:
            return bop_folder / dataset / f'onboarding_{onboarding_type}' / f'{sequence}_{direction}'
        else:
            raise ValueError(f'Unknown onboarding type {onboarding_type} or direction {direction}')

    elif sequence_type in ['test', 'val', 'train']:
        return bop_folder / dataset / sequence_type / sequence
    else:
        raise ValueError(f'Unknown sequence type: {sequence_type}')


def extract_gt_Se3_cam2obj(pose_json_path: Path, object_id: int = None, device: str = 'cpu') -> Dict[int, Se3]:

    dict_gt_Se3_obj2cam = read_obj2cam_Se3_from_gt(pose_json_path, device)

    if object_id is None:
        obj_ids = sorted(dict_gt_Se3_obj2cam.keys())
        object_id = obj_ids[0]
    dict_gt_Se3_obj2cam = dict_gt_Se3_obj2cam[object_id]
    gt_Se3_obj2cam_frames = dict_gt_Se3_obj2cam.keys()
    gt_Se3_cam2obj = {frame: dict_gt_Se3_obj2cam[frame].inverse() for frame in gt_Se3_obj2cam_frames}

    return gt_Se3_cam2obj


def get_bop_images_and_segmentations(bop_folder, dataset, sequence, sequence_type, onboarding_type):
    """Loads images and segmentations from BOP dataset based on sequence type."""
    sequence_starts = [0]

    if sequence_type == 'onboarding' and onboarding_type == 'static':
        sequence_folder_down = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                                   "down")
        sequence_folder_up = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, "up")

        gt_images, gt_segs = load_gt_images_and_segmentations(sequence_folder_down / 'rgb',
                                                              sequence_folder_down / 'mask_visib')
        gt_images2, gt_segs2 = load_gt_images_and_segmentations(sequence_folder_up / 'rgb',
                                                                sequence_folder_up / 'mask_visib')

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


def read_gt_Se3_cam2obj_transformations(bop_folder: Path, dataset: str, sequence: str, sequence_type: str,
                                        onboarding_type: str, sequence_starts: List[int], device: str):
    if onboarding_type == 'dynamic':
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_gt.json'
        gt_Se3_cam2obj = extract_gt_Se3_cam2obj(pose_json_path, device=device)
    elif onboarding_type == 'static':
        sequence_down_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type,
                                                   onboarding_type, 'down')
        pose_json_path_down = sequence_down_folder / 'scene_gt.json'

        sequence_up_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type,
                                                 onboarding_type, 'up')
        pose_json_path_up = sequence_up_folder / 'scene_gt.json'

        gt_Se3_cam2obj_down = extract_gt_Se3_cam2obj(pose_json_path_down, device=device)
        gt_Se3_cam2obj_up = extract_gt_Se3_cam2obj(pose_json_path_up, device=device)

        gt_Se3_cam2obj = gt_Se3_cam2obj_down
        gt_Se3_cam2obj = gt_Se3_cam2obj | {frm + sequence_starts[1]: gt_Se3_cam2obj_up[frm]
                                           for frm in gt_Se3_cam2obj_up.keys()}
    else:
        raise ValueError("Unknown onboarding type.")
    return gt_Se3_cam2obj


def read_pinhole_params(bop_folder: Path, dataset: str, sequence: str, sequence_type: str,
                        onboarding_type: str, sequence_starts: List[int]):
    if onboarding_type == 'dynamic':
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_camera.json'
        pinhole_params = get_pinhole_params(pose_json_path)
    elif onboarding_type == 'static':
        sequence_down_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type,
                                                   onboarding_type, 'down')
        pose_json_path_down = sequence_down_folder / 'scene_camera.json'

        sequence_up_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type,
                                                 onboarding_type, 'up')
        pose_json_path_up = sequence_up_folder / 'scene_camera.json'

        pinhole_params_down = get_pinhole_params(pose_json_path_down)
        pinhole_params_up = get_pinhole_params(pose_json_path_up)

        pinhole_params = pinhole_params_down
        pinhole_params = pinhole_params | {frm + sequence_starts[1]: pinhole_params_up[frm]
                                           for frm in pinhole_params_up.keys()}
    else:
        raise ValueError("Unknown onboarding type.")
    return pinhole_params

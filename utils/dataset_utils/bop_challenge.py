import json
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera, rotation_matrix_to_axis_angle


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


def read_obj_to_cam_transformations_from_gt(pose_json_path) -> (
        Tuple)[Dict[int, Dict[int, torch.Tensor]], Dict[int, Dict[int, torch.Tensor]]]:
    gt_translations_obj_to_cam = defaultdict(dict)
    gt_rotations_obj_to_cam = defaultdict(dict)
    with open(pose_json_path, 'r') as file:
        pose_json = json.load(file)
        for frame, data in pose_json.items():
            frame = int(frame)
            for entry in data:
                obj_id = entry['obj_id']
                R_obj_to_cam = entry['cam_R_m2c']
                R_m2c = torch.tensor(np.array(R_obj_to_cam).reshape(3, 3))
                r_m2c = rotation_matrix_to_axis_angle(R_m2c).numpy()

                cam_t_m2c = entry['cam_t_m2c']
                t_m2c = np.array(cam_t_m2c)

                gt_rotations_obj_to_cam[obj_id][frame] = r_m2c
                gt_translations_obj_to_cam[obj_id][frame] = t_m2c
    return gt_rotations_obj_to_cam, gt_translations_obj_to_cam

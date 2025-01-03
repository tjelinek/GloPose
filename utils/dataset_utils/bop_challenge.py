import json
from pathlib import Path
from typing import List

import torch
import torchvision.transforms.functional as F
from PIL import Image
from kornia.geometry import Se3, Quaternion, PinholeCamera

from data_structures.keyframe_buffer import FrameObservation
from data_structures.keyframe_graph import KeyframeGraph


def bop_t_to_torch_tensor(json_t_value: list[float]) -> torch.Tensor:
    return torch.tensor(json_t_value).unsqueeze(0)


def bop_R_to_torch_tensor(json_R_value: list[float]) -> torch.Tensor:
    return torch.tensor(json_R_value).reshape(3, 3).unsqueeze(0)


def get_bop_segmentation_tensor(segmentation_path: Path) -> torch.Tensor:
    segmentation = Image.open(segmentation_path)
    segmentation_tensor = F.pil_to_tensor(segmentation)[None, None].to(torch.bool).to(torch.float32).cuda()

    return segmentation_tensor


def get_bop_image_tensor(rgb_path: Path) -> torch.Tensor:
    rgb_image = Image.open(rgb_path)
    rgb_image_tensor = F.pil_to_tensor(rgb_image).float()[None, None]
    return rgb_image_tensor


def perform_onboarding(sequence_path: Path) -> KeyframeGraph:
    scene_gt_json_path = sequence_path / 'scene_gt.json'
    scene_cam_json_path = sequence_path / 'scene_camera.json'
    rgb_folder_path = sequence_path / 'rgb/'
    segmentation_path = sequence_path / 'mask_visib'

    with open(scene_gt_json_path, 'r') as json_file:
        scene_gt_json_data = json.load(json_file)

    rgb_image_paths_sorted = list(sorted(rgb_folder_path.iterdir()))
    segmentation_paths_sorted = list(sorted(segmentation_path.iterdir()))

    Se3_obj_to_cam = get_Se3_world_to_cam_from_bop_json(scene_gt_json_data)
    pinhole_params = get_pinhole_params(scene_cam_json_path)

    pose_icosphere = KeyframeGraph(30)

    for json_item, rgb_path, segmentation_path in zip(scene_gt_json_data.items(), rgb_image_paths_sorted,
                                                      segmentation_paths_sorted):
        frame_i = int(json_item[0])
        frame_data = json_item[1][0]

        frame_cam_t = bop_t_to_torch_tensor(frame_data['cam_t_m2c'])
        frame_cam_R = bop_R_to_torch_tensor(frame_data['cam_R_m2c'])

        frame_cam_Se3 = Se3(Quaternion.from_matrix(frame_cam_R), frame_cam_t)

        if pose_icosphere.check_inserting_new_node(frame_cam_Se3):
            rgb_image = get_bop_image_tensor(rgb_path)
            segmentation = get_bop_segmentation_tensor(segmentation_path)

            frame_observation = FrameObservation(observed_image=rgb_image, observed_segmentation=segmentation,
                                                 observed_image_features=None)

            pose_icosphere.insert_new_reference(frame_observation, frame_cam_Se3, frame_i,
                                                pinhole_params[frame_i])

    return pose_icosphere


def get_Se3_world_to_cam_from_bop_json(json_data) -> Se3:
    json_data_first_frame = json_data['0'][0]
    cam_t_first_frame = bop_t_to_torch_tensor(json_data_first_frame['cam_t_m2c'])
    cam_R_first_frame = bop_R_to_torch_tensor(json_data_first_frame['cam_R_m2c'])
    Se3_cam_to_world = Se3(Quaternion.from_matrix(cam_R_first_frame), cam_t_first_frame)
    Se3_world_to_cam = Se3_cam_to_world.inverse()

    return Se3_world_to_cam


def load_intrinsics_as_tensor(json_file_path) -> torch.Tensor:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    cam_K_matrices = []
    for key, value in json_data.items():
        cam_K = value['cam_K']
        cam_K_reshaped = torch.tensor(cam_K).view(3, 3)
        cam_K_matrices.append(cam_K_reshaped)

    cam_K_tensor = torch.stack(cam_K_matrices)

    return cam_K_tensor


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
import json
import warnings
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera

from tracker_config import TrackerConfig
from utils.data_utils import get_scale_from_meter, get_scale_to_meter
from utils.image_utils import decode_rle_list
from utils.math_utils import scale_Se3


def get_pinhole_params(json_file_path: Path, scale: float = 1.0, device='cpu') -> Dict[int, PinholeCamera]:
    with open(json_file_path, 'r') as json_file:
        json_data = json.load(json_file)

    no_image_shape_available = False
    pinhole_cameras = {}
    for frame_str, value in json_data.items():
        frame_int = int(frame_str)
        frame_data = json_data[frame_str]
        cam_K = torch.tensor(frame_data['cam_K'], device=device).view(3, 3)
        cam_w2c = Se3.identity(device=device).matrix()

        if frame_data.get('width') is not None and frame_data.get('height') is not None:
            width = torch.tensor(frame_data['width'], device=device)
            height = torch.tensor(frame_data['height'], device=device)
        else:
            width, height = torch.tensor(0, device=device), torch.tensor(0, device=device)
            no_image_shape_available = True

        pinhole_camera = PinholeCamera(cam_K.unsqueeze(0), cam_w2c.unsqueeze(0),
                                       height.unsqueeze(0), width.unsqueeze(0))
        pinhole_camera = pinhole_camera.scale(torch.tensor(scale, device=device).unsqueeze(0))

        pinhole_cameras[frame_int] = pinhole_camera

    if no_image_shape_available:
        warnings.warn(f"The image shape is not available in the {json_file_path} file. ")

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


def load_gt_images(image_folder: Path):
    """Load ground truth images."""
    gt_images = {
        int(file.stem): file
        for file in sorted(image_folder.iterdir())
        if file.is_file()
    }

    return gt_images


def load_gt_segmentations(segmentation_folder: Path, object_id: int = None):
    """Load segmentation files, filtering by object ID."""
    object_id_str = f"{object_id:06d}" if object_id is not None else None  # Ensure it's a zero-padded 6-digit string

    gt_segs = {
        int(file.stem.split('_')[0]): file
        for file in sorted(segmentation_folder.iterdir())
        if object_id is None or file.stem.endswith(object_id_str)  # Dynamically filter by object ID
    }

    return gt_segs


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


def extract_gt_Se3_cam2obj(pose_json_path: Path, scale_factor: float, scene_object_object_id: int = None,
                           object_id: int = None, device: str = 'cpu') -> Dict[int, Se3]:
    dict_gt_Se3_obj2cam = read_obj2cam_Se3_from_gt(pose_json_path, device)

    assert not (scene_object_object_id is not None and object_id is not None)
    obj_ids = sorted(dict_gt_Se3_obj2cam.keys())

    if scene_object_object_id is None and object_id is None:
        object_id = obj_ids[0]
    elif scene_object_object_id is not None and object_id is None:
        object_id = obj_ids[scene_object_object_id]
    else:
        assert object_id in obj_ids

    dict_gt_Se3_obj2cam = dict_gt_Se3_obj2cam[object_id]
    gt_Se3_obj2cam_frames = dict_gt_Se3_obj2cam.keys()
    gt_Se3_cam2obj = {frame: scale_Se3(dict_gt_Se3_obj2cam[frame].inverse(), scale_factor)
                      for frame in gt_Se3_obj2cam_frames}

    return gt_Se3_cam2obj


def extract_object_id(pose_json_path: Path, scene_object_object_id: int = None) -> Dict[int, int]:

    dict_gt_Se3_obj2cam = read_obj2cam_Se3_from_gt(pose_json_path, 'cpu')

    obj_ids = sorted(dict_gt_Se3_obj2cam.keys())

    if scene_object_object_id is None:
        object_id = obj_ids[0]
    else:
        object_id = obj_ids[scene_object_object_id]

    return {1: object_id}


def load_static_onboarding_parts(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str,
        static_onboarding_sequence: Optional[str],
        loader_fn: Callable[[Path], Optional[dict]],
        sequence_starts: Optional[List[int]] = None
) -> dict:
    folder_down = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'down')
    folder_up = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'up')

    data_down = loader_fn(folder_down) if folder_down.exists() else None
    data_up = loader_fn(folder_up) if folder_up.exists() else None

    if static_onboarding_sequence == 'both':
        assert data_down is not None and data_up is not None and sequence_starts is not None
        merged_data = data_down.copy()
        merged_data.update({k + sequence_starts[1]: v for k, v in data_up.items()})
        return merged_data
    elif static_onboarding_sequence == 'down':
        assert data_down is not None
        return data_down
    elif static_onboarding_sequence == 'up':
        assert data_up is not None
        return data_up
    else:
        raise ValueError(f'Unknown static onboarding sequence type: {static_onboarding_sequence}')


def get_bop_images_and_segmentations(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str = None,
        static_onboarding_sequence: Optional[str] = None,
        scene_obj_id: int = None,
) -> Tuple[Dict[int, Path], Dict[int, Path], Optional[Dict[int, Path]], Optional[List[int]]]:
    """Loads images and segmentations from BOP dataset based on sequence type."""
    sequence_starts = [0]

    if sequence_type == 'onboarding' and onboarding_type == 'static':
        down_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'down')
        up_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'up')

        images_down = load_gt_images(down_folder / 'rgb')
        images_up = load_gt_images(up_folder / 'rgb')
        segs_down = load_gt_segmentations(down_folder / 'mask_visib')
        segs_up = load_gt_segmentations(up_folder / 'mask_visib')

        depth_down_folder = down_folder / 'depth'
        depth_up_folder = up_folder / 'depth'
        depths_down = None
        depths_up = None
        if depth_down_folder.exists():
            depths_down = load_gt_images(depth_down_folder)
        if depth_up_folder.exists():
            depths_up = load_gt_images(depth_up_folder)

        if static_onboarding_sequence == 'both':
            assert images_down is not None and images_up is not None
            assert segs_down is not None and segs_up is not None
            sequence_starts.append(len(images_down))

            merged_images = images_down
            merged_segmentations = images_down
            merged_depths = depths_down

            offset = sequence_starts[1]
            for frame, img in images_up.items():
                merged_images[offset + frame] = img
            for frame, seg in segs_up.items():
                merged_segmentations[offset + frame] = seg
            if merged_depths is not None:
                for frame, depth in depths_up.items():
                    merged_depths[offset + frame] = depth

            return merged_images, merged_segmentations, merged_depths, sequence_starts

        elif static_onboarding_sequence == 'down':
            return images_down, segs_down, depths_down, sequence_starts

        elif static_onboarding_sequence == 'up':
            return images_up, segs_up, depths_up, sequence_starts

        else:
            raise ValueError(f'Unknown static onboarding sequence type: {static_onboarding_sequence}')
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'
        depth_folder = sequence_folder / 'depth'
        gt_images = load_gt_images(image_folder)
        gt_segs = load_gt_segmentations(segmentation_folder, object_id=scene_obj_id)
        gt_depths = None
        if depth_folder.exists():
            gt_depths = load_gt_images(depth_folder)
        return gt_images, gt_segs, gt_depths, None


def read_gt_Se3_cam2obj_transformations(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, scale_factor,
                                        onboarding_type: str = None, sequence_starts: Optional[List[int]] = None,
                                        static_onboarding_sequence: Optional[str] = None, scene_obj_id: int = None,
                                        device: str = 'cpu') -> Dict[int, Se3]:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: extract_gt_Se3_cam2obj(p / 'scene_gt.json', scale_factor, device=device),
            sequence_starts=sequence_starts
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_gt.json'
        return extract_gt_Se3_cam2obj(pose_json_path, scale_factor, scene_obj_id, device=device)


def read_object_id(bop_folder: Path, dataset: str, sequence: str, sequence_type: str,
                   onboarding_type: str = None, static_onboarding_sequence: Optional[str] = None,
                   scene_obj_id: int = None) -> int:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: extract_object_id(p / 'scene_gt.json'),
            sequence_starts=None
        )[1]
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_gt.json'
        return extract_object_id(pose_json_path, scene_obj_id)[1]


def read_pinhole_params(bop_folder: Path, dataset: str, sequence: str, sequence_type: str, scale,
                        onboarding_type: str = None, static_onboarding_sequence: Optional[str] = None,
                        sequence_starts: Optional[List[int]] = None, device='cpu') -> dict[int, PinholeCamera]:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: get_pinhole_params(p / 'scene_camera.json', scale, device=device),
            sequence_starts=sequence_starts
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_camera.json'
        return get_pinhole_params(pose_json_path, scale, device=device)


def add_extrinsics_to_pinhole_params(pinhole_params: Dict[int, PinholeCamera], gt_Se3_world2cam: dict[int, Se3]) -> (
        Dict)[int, PinholeCamera]:
    for frm_i in pinhole_params.keys():
        pinhole = pinhole_params[frm_i]
        gt_T_world2cam = gt_Se3_world2cam[frm_i].matrix().unsqueeze(0)
        pinhole_params[frm_i] = PinholeCamera(pinhole.intrinsics, gt_T_world2cam, pinhole.width, pinhole.height)

    return pinhole_params


def get_gop_camera_intrinsics(json_path: Path, image_id: int):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if str(image_id) not in data:
        raise ValueError(f"Image ID {image_id} not found in the JSON file.")

    cam_K = data[str(image_id)]['cam_K']
    return np.array(cam_K).reshape(3, 3)


def read_gt_Se3_world2cam(pose_json_path: Path, input_scale='m', output_scale='m', device: str = 'cpu') \
        -> dict[int, Se3]:
    data = json.loads(pose_json_path.read_text())
    scale = get_scale_to_meter(input_scale) * get_scale_from_meter(output_scale)
    result = {}
    for frame_id_str, frame_data in data.items():
        R = torch.tensor(frame_data['cam_R_w2c'], dtype=torch.float32, device=device).reshape(3, 3)
        t = torch.tensor(frame_data['cam_t_w2c'], dtype=torch.float32, device=device).reshape(3) * scale
        result[int(frame_id_str)] = Se3(Quaternion.from_matrix(R), t)
    return result


def read_depth_scales(pose_json_path: Path) -> dict[int, float]:
    data = json.loads(pose_json_path.read_text())
    return {int(k): v['depth_scale'] for k, v in data.items()}


def read_static_onboarding_world2cam(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: Optional[str] = None,
        static_onboarding_sequence: Optional[str] = None,
        sequence_starts: Optional[List[int]] = None,
        device: str = 'cpu'
) -> dict[int, Se3]:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: read_gt_Se3_world2cam(p / 'scene_camera.json', device=device),
            sequence_starts=sequence_starts
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_camera.json'
        return read_gt_Se3_world2cam(pose_json_path, device=device)


def read_dynamic_onboarding_depth_scales(
        bop_folder: Path,
        dataset: str,
        sequence: str,
        sequence_type: str,
        onboarding_type: str
) -> dict[int, float]:
    sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
    pose_json_path = sequence_folder / 'scene_camera.json'
    return read_depth_scales(pose_json_path)


def get_default_detections_per_scene_and_image(default_detections_file: Path) -> Dict[Tuple[int, int], List]:
    with open(default_detections_file, 'r') as f:
        default_detections_data = json.load(f)
        default_detections_scene_im_dict = defaultdict(list)
        for i, item in enumerate(default_detections_data):
            im_id: int = item['image_id']
            scene_id: int = item['scene_id']
            default_detections_scene_im_dict[(im_id, scene_id)].append(item)
    return default_detections_scene_im_dict


Detection = namedtuple('Detection', ['object_id', 'segmentation_mask', 'score'])


def get_default_detections_for_image(default_detections_data_scene_im_dict: Dict[Tuple[int, int], List], scene_id: int,
                                     im_id: int, device: str = 'cpu') -> List[Detection]:
    detections_for_image = []  # Initialize as list, not dict
    for detections_data in default_detections_data_scene_im_dict[(im_id, scene_id)]:
        segmentation_rle_format = detections_data['segmentation']

        mask = decode_rle_list(segmentation_rle_format)
        mask_tensor = torch.tensor(mask, device=device)
        detections_data['segmentation_tensor'] = mask_tensor

        detections_for_image.append(detections_data)

    detections_for_image.sort(key=lambda x: (x['score'], x['category_id']), reverse=True)

    sorted_detections = [Detection(object_id=detection['category_id'],
                                   segmentation_mask=detection['segmentation_tensor'],
                                   score=detection['score'])
                         for detection in detections_for_image]

    return sorted_detections


def set_config_for_bop_onboarding(config: TrackerConfig, sequence: str):
    sequence_name_split = sequence.split('_')
    if len(sequence_name_split) == 3:
        if sequence_name_split[2] == 'down':
            config.bop_config.onboarding_type = 'static'
            config.bop_config.static_onboarding_sequence = 'down'
            config.similarity_transformation = 'kabsch'
        elif sequence_name_split[2] == 'up':
            config.bop_config.onboarding_type = 'static'
            config.bop_config.static_onboarding_sequence = 'up'
            config.similarity_transformation = 'kabsch'
        elif sequence_name_split[2] == 'dynamic':
            config.bop_config.onboarding_type = 'dynamic'
            config.similarity_transformation = 'depths'
            config.frame_provider_config.erode_segmentation = True
            config.run_only_on_frames_with_known_pose = False
            config.skip_indices *= 4
        config.sequence = '_'.join(sequence_name_split[:2])

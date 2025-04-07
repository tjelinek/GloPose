import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Callable

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion, PinholeCamera

from data_providers.flow_provider import RoMaFlowProviderDirect
from data_providers.frame_provider import PrecomputedFrameProvider, PrecomputedSegmentationProvider
from data_structures.view_graph import ViewGraph, load_view_graph
from pose.glomap import predict_poses
from tracker_config import TrackerConfig
from utils.image_utils import get_target_shape


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
    static_onboarding_sequence: Optional[str] = None
):
    """Loads images and segmentations from BOP dataset based on sequence type."""
    sequence_starts = [0]

    def load_fn(folder: Path):
        return load_gt_images_and_segmentations(folder / 'rgb', folder / 'mask_visib')

    if sequence_type == 'onboarding' and onboarding_type == 'static':
        data_down = load_fn(get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'down'))
        data_up = load_fn(get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type, 'up'))

        if static_onboarding_sequence == 'both':
            assert data_down is not None and data_up is not None
            sequence_starts.append(len(data_down[0]))

            merged_images = data_down[0]
            merged_segmentations = data_down[1]

            offset = sequence_starts[1]
            for frame, img in data_up[0].items():
                merged_images[offset + frame] = img
            for frame, seg in data_up[1].items():
                merged_segmentations[offset + frame] = seg

            return merged_images, merged_segmentations, sequence_starts

        elif static_onboarding_sequence == 'down':
            return data_down[0], data_down[1], sequence_starts

        elif static_onboarding_sequence == 'up':
            return data_up[0], data_up[1], sequence_starts

        else:
            raise ValueError(f'Unknown static onboarding sequence type: {static_onboarding_sequence}')
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'
        gt_images, gt_segs = load_gt_images_and_segmentations(image_folder, segmentation_folder)
        return gt_images, gt_segs, None


def read_gt_Se3_cam2obj_transformations(
    bop_folder: Path,
    dataset: str,
    sequence: str,
    sequence_type: str,
    onboarding_type: str = None,
    sequence_starts: Optional[List[int]] = None,
    static_onboarding_sequence: Optional[str] = None,
    device: str = 'cpu'
) -> Dict[int, Se3]:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: extract_gt_Se3_cam2obj(p / 'scene_gt.json', device=device),
            sequence_starts=sequence_starts
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_gt.json'
        return extract_gt_Se3_cam2obj(pose_json_path, device=device)


def read_pinhole_params(
    bop_folder: Path,
    dataset: str,
    sequence: str,
    sequence_type: str,
    onboarding_type: str = None,
    static_onboarding_sequence: Optional[str] = None,
    sequence_starts: Optional[List[int]] = None
) -> dict[int, PinholeCamera]:
    if sequence_type == 'onboarding' and onboarding_type == 'static':
        return load_static_onboarding_parts(
            bop_folder,
            dataset,
            sequence,
            sequence_type,
            onboarding_type,
            static_onboarding_sequence,
            loader_fn=lambda p: get_pinhole_params(p / 'scene_camera.json'),
            sequence_starts=sequence_starts
        )
    else:
        sequence_folder = get_sequence_folder(bop_folder, dataset, sequence, sequence_type, onboarding_type)
        pose_json_path = sequence_folder / 'scene_camera.json'
        return get_pinhole_params(pose_json_path)


def get_gop_camera_intrinsics(json_path: Path, image_id: int):
    with open(json_path, 'r') as f:
        data = json.load(f)

    if str(image_id) not in data:
        raise ValueError(f"Image ID {image_id} not found in the JSON file.")

    cam_K = data[str(image_id)]['cam_K']
    return np.array(cam_K).reshape(3, 3)


def predict_poses_for_bop_challenge(bop_targets_path: Path, view_graph_save_paths: Path, config: TrackerConfig) -> None:
    with bop_targets_path.open('r') as file:
        test_annotations = json.load(file)

    test_dataset_path = bop_targets_path.parent.parent / 'test'

    for item in test_annotations:
        im_id = item['im_id']
        scene_id = item['scene_id']

        scene_folder_name = f'{scene_id:06d}'
        image_id_str = f'{im_id:06d}'
        image_filename = f'{image_id_str}.png'

        path_to_scene = test_dataset_path / scene_folder_name
        path_to_image = path_to_scene / 'rgb' / image_filename
        path_to_camera_intrinsics = path_to_scene / 'scene_camera.json'
        segmentation_paths = path_to_scene / 'mask_visib'

        segmentation_files = sorted(segmentation_paths.glob(f"{image_id_str}_*.png"))
        camera_intrinsics = get_gop_camera_intrinsics(path_to_camera_intrinsics, im_id)

        view_graphs: List[ViewGraph] = []
        for view_graph_dir in view_graph_save_paths.iterdir():
            if view_graph_dir.is_dir():
                view_graph = load_view_graph(view_graph_dir)
                view_graphs.append(view_graph)

        predict_all_poses_in_image(path_to_image, segmentation_files, camera_intrinsics, view_graphs, config)


def predict_all_poses_in_image(image_path: Path, segmentation_paths: List[Path], camera_K: np.ndarray,
                               view_graphs: List[ViewGraph],
                               config: TrackerConfig) -> None:

    target_shape = get_target_shape(image_path, config.image_downsample)
    image = PrecomputedFrameProvider.load_and_downsample_image(image_path, config.image_downsample, config.device)

    flow_provider = RoMaFlowProviderDirect('cpu')

    for segmentation_paths in segmentation_paths:
        segmentation = PrecomputedSegmentationProvider.load_and_downsample_segmentation(segmentation_paths,
                                                                                        target_shape,
                                                                                        config.device)

        # TODO iterate over all view graphs
        predict_poses(image, segmentation, camera_K=camera_K, view_graph=view_graphs[0], flow_provider=flow_provider,
                      config=config)


if __name__ == '__main__':
    _bop_targets_path = Path('/mnt/personal/jelint19/data/bop/hope/hope/test_targets_bop24.json')
    _view_graph_location = Path('/mnt/personal/jelint19/cache/view_graph_cache/hope')

    _config = TrackerConfig()
    predict_poses_for_bop_challenge(_bop_targets_path, _view_graph_location, _config)

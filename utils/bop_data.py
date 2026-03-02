"""BOP dataset path-resolution utilities.

Pure functions for constructing paths that follow BOP on-disk conventions
(zero-padded IDs, subfolder names, dataset-specific aria/quest3 variants).
"""

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# ID formatting
# ---------------------------------------------------------------------------

def format_scene_id(scene_id: int) -> str:
    return f'{scene_id:06d}'


def format_image_id(im_id: int) -> str:
    return f'{im_id:06d}'


# ---------------------------------------------------------------------------
# Folder / file resolution
# ---------------------------------------------------------------------------

def get_scene_folder(split_path: Path, scene_id: int) -> Path:
    return split_path / format_scene_id(scene_id)


def get_rgb_folder_name(split: str = '') -> str:
    if 'aria' in split and 'static' in split:
        return 'rgb'
    if 'quest3' in split and 'static' in split:
        return 'gray1'
    return 'rgb'


def get_segmentation_folder_name(split: str = '') -> str:
    if 'aria' in split and 'static' in split:
        return 'mask_visib_rgb'
    if 'quest3' in split and 'static' in split:
        return 'mask_visib_gray1'
    return 'mask_visib'


def get_scene_gt_filename(split: str = '') -> str:
    if 'aria' in split and 'static' in split:
        return 'scene_gt_rgb.json'
    if 'quest3' in split and 'static' in split:
        return 'scene_gt_gray1.json'
    return 'scene_gt.json'


def get_scene_gt_path(scene_folder: Path, split: str = '') -> Path:
    return scene_folder / get_scene_gt_filename(split)


def get_camera_json_path(scene_folder: Path) -> Path:
    return scene_folder / 'scene_camera.json'


# ---------------------------------------------------------------------------
# Image path with .png/.jpg fallback + hot3d quest3 fallback
# ---------------------------------------------------------------------------

def get_image_path(scene_folder: Path, im_id: int, dataset_name: str = '') -> Path:
    image_id_str = format_image_id(im_id)

    path_png = scene_folder / 'rgb' / f'{image_id_str}.png'
    if path_png.exists():
        return path_png

    path_jpg = scene_folder / 'rgb' / f'{image_id_str}.jpg'
    if dataset_name == 'hot3d':
        if path_jpg.exists():
            return path_jpg
        quest3_folder = Path(str(scene_folder).replace('aria', 'quest3'))
        path_quest3 = quest3_folder / 'rgb' / f'{image_id_str}.jpg'
        if path_quest3.exists():
            return path_quest3
        raise FileNotFoundError(f"Image file not found: {path_quest3}")

    if path_jpg.exists():
        return path_jpg

    raise FileNotFoundError(f"Image file not found: {path_jpg}")


# ---------------------------------------------------------------------------
# Camera intrinsics
# ---------------------------------------------------------------------------

def load_camera_intrinsics(json_path: Path, im_id: int) -> np.ndarray:
    with open(json_path, 'r') as f:
        data = json.load(f)

    key = str(im_id)
    if key not in data:
        raise ValueError(f"Image ID {im_id} not found in {json_path}")

    return np.array(data[key]['cam_K']).reshape(3, 3)


# ---------------------------------------------------------------------------
# Test targets
# ---------------------------------------------------------------------------

def load_test_targets(targets_path: Path) -> list[dict]:
    with targets_path.open('r') as f:
        raw = json.load(f)
    return _group_test_targets_by_image(raw)


def _group_test_targets_by_image(test_annotations: list[dict]) -> list[dict]:
    grouped = OrderedDict()
    for item in test_annotations:
        key = (item['im_id'], item['scene_id'])
        if key not in grouped:
            grouped[key] = {
                'im_id': item['im_id'],
                'scene_id': item['scene_id'],
                'objects': [],
                'objects_counts': [],
            }
        if 'obj_id' in item:
            grouped[key]['objects'].append(item['obj_id'])
            grouped[key]['objects_counts'].append(item['inst_count'])
    return list(grouped.values())


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def get_targets_filename(dataset: str, split: str) -> str:
    if dataset in ('hope', 'handal') and split == 'val':
        return 'val_targets_bop24.json'
    return 'test_targets_bop19.json'


def should_run_evaluation(dataset: str, split: str) -> bool:
    if dataset in ('hope', 'handal') and split != 'val':
        return False
    return True

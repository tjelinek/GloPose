import json
from pathlib import Path
from typing import List, Tuple


def get_handal_sequences(path: Path) -> Tuple[List[str], List[str]]:
    """Scan HANDAL dataset directory for train/test sequences.

    Args:
        path: Path to the HANDAL dataset root (e.g., data_folder / 'HANDAL')

    Returns:
        Tuple of (train_sequences, test_sequences), each formatted as 'category@sequence'
    """
    train_sequences = []
    test_sequences = []

    if not path.exists():
        return train_sequences, test_sequences

    for category_dir in sorted(path.iterdir()):
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        train_dir = category_dir / "train"
        if train_dir.exists():
            for sequence_dir in sorted(train_dir.iterdir()):
                if sequence_dir.is_dir():
                    train_sequences.append(f"{category_name}@{sequence_dir.name}")

        test_dir = category_dir / "test"
        if test_dir.exists():
            for sequence_dir in sorted(test_dir.iterdir()):
                if sequence_dir.is_dir():
                    test_sequences.append(f"{category_name}@{sequence_dir.name}")

    return train_sequences, test_sequences


def get_navi_sequences(path: Path) -> List[str]:
    """Scan NAVI dataset directory for video sequences.

    Args:
        path: Path to the NAVI dataset root (e.g., data_folder / 'NAVI' / 'navi_v1.5')

    Returns:
        Sorted list of sequences formatted as 'object@video-folder'
    """
    video_sequences = []

    if not path.exists():
        return video_sequences

    for object_dir in path.iterdir():
        if object_dir.is_dir():
            for item in object_dir.iterdir():
                if item.is_dir() and item.name.startswith('video-'):
                    video_sequences.append(f"{object_dir.name}@{item.name}")

    return sorted(video_sequences)


def get_ho3d_sequences(path: Path) -> Tuple[List[str], List[str]]:
    """Scan HO3D dataset directory for train/evaluation sequences.

    Args:
        path: Path to the HO3D dataset root (e.g., data_folder / 'HO3D')

    Returns:
        Tuple of (train_sequences, test_sequences)
    """
    train_sequences = []
    test_sequences = []

    train_dir = path / 'train'
    if train_dir.exists():
        train_sequences = sorted(d.name for d in train_dir.iterdir() if d.is_dir())

    eval_dir = path / 'evaluation'
    if eval_dir.exists():
        test_sequences = sorted(d.name for d in eval_dir.iterdir() if d.is_dir())

    return train_sequences, test_sequences


def get_tum_rgbd_sequences(path: Path) -> List[str]:
    """Scan TUM RGB-D dataset directory for sequences.

    Args:
        path: Path to the TUM RGB-D dataset root (e.g., data_folder / 'SLAM' / 'tum_rgbd')

    Returns:
        Sorted list of sequence directory names
    """
    if not path.exists():
        return []

    return sorted(d.name for d in path.iterdir() if d.is_dir())


def get_behave_sequences(path: Path) -> List[str]:
    """Scan BEHAVE dataset directory for video sequences.

    Args:
        path: Path to the BEHAVE train directory (e.g., data_folder / 'BEHAVE' / 'train')

    Returns:
        Sorted list of sequence names (mp4 stems, excluding *_mask_obj)
    """
    if not path.exists():
        return []

    return sorted(
        f.stem for f in path.glob('*.mp4')
        if not f.stem.endswith('_mask_obj')
    )


def get_google_scanned_objects_sequences(path: Path) -> List[str]:
    """Scan GoogleScannedObjects models directory for sequences.

    Args:
        path: Path to the models directory (e.g., data_folder / 'GoogleScannedObjects' / 'models')

    Returns:
        Sorted list of model directory names
    """
    if not path.exists():
        return []

    return sorted(d.name for d in path.iterdir() if d.is_dir())


def get_bop_val_sequences(path: Path) -> List[str]:
    """Scan BOP val directory for scene/object sequences.

    Args:
        path: Path to the BOP val directory (e.g., data_folder / 'bop' / 'handal' / 'val')

    Returns:
        Sorted list of sequences formatted as '{scene_id}_{object_id}'
    """
    if not path.exists():
        return []

    sequences = []
    for scene_dir in sorted(path.iterdir()):
        if not scene_dir.is_dir():
            continue

        scene_id = scene_dir.name
        scene_gt_path = scene_dir / 'scene_gt.json'
        if scene_gt_path.exists():
            with open(scene_gt_path, 'r') as f:
                scene_gt = json.load(f)

            obj_ids = set()
            for frame_data in scene_gt.values():
                for obj_entry in frame_data:
                    obj_ids.add(obj_entry['obj_id'])

            for obj_id in sorted(obj_ids):
                sequences.append(f"{scene_id}_{obj_id:06d}")

    return sequences


def get_bop_onboarding_sequences(path: Path, dataset: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Scan BOP onboarding directories for sequences.

    Args:
        path: Path to the BOP root directory (e.g., data_folder / 'bop')
        dataset: Dataset name (e.g., 'handal', 'hope')

    Returns:
        Tuple of (dynamic, static_up, static_down, static_both) sequence lists
    """
    dynamic_sequences = []
    static_up_sequences = []
    static_down_sequences = []

    dynamic_dir = path / dataset / 'onboarding_dynamic'
    if dynamic_dir.exists():
        for obj_dir in sorted(dynamic_dir.iterdir()):
            if obj_dir.is_dir():
                dynamic_sequences.append(f"{obj_dir.name}_dynamic")

    static_dir = path / dataset / 'onboarding_static'
    up_objects = set()
    down_objects = set()
    if static_dir.exists():
        for obj_dir in sorted(static_dir.iterdir()):
            if not obj_dir.is_dir():
                continue

            name = obj_dir.name
            if name.endswith('_up'):
                base = name[:-3]  # Remove '_up'
                up_objects.add(base)
                static_up_sequences.append(f"{base}_up")
            elif name.endswith('_down'):
                base = name[:-5]  # Remove '_down'
                down_objects.add(base)
                static_down_sequences.append(f"{base}_down")

    # Objects that have both up and down get a 'both' entry
    both_objects = sorted(up_objects & down_objects)
    static_both_sequences = [f"{obj}_both" for obj in both_objects]

    static_up_sequences.sort()
    static_down_sequences.sort()

    return dynamic_sequences, static_up_sequences, static_down_sequences, static_both_sequences


def get_bop_classic_sequences(path: Path, dataset: str, split: str) -> List[str]:
    """Scan BOP classic dataset directory for sequences.

    Args:
        path: Path to the BOP root directory (e.g., data_folder / 'bop')
        dataset: Dataset name (e.g., 'tless', 'lmo', 'icbin')
        split: Split name (e.g., 'train_primesense', 'train')

    Returns:
        Sorted list of sequences formatted as '{dataset}@{split}@{sequence_name}'
    """
    split_dir = path / dataset / split
    if not split_dir.exists():
        return []

    return sorted(
        f"{dataset}@{split}@{d.name}"
        for d in split_dir.iterdir()
        if d.is_dir()
    )

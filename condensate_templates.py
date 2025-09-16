import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
from imblearn.under_sampling import CondensedNearestNeighbour

from utils.bop_challenge import extract_object_id

sys.path.append('./repositories/cnos')
from src.model.dinov2 import descriptor_from_hydra


def perform_condensation_per_dataset(bop_base: Path, cache_base_path: Path, dataset: str, split: str, device='cuda'):
    path_to_dataset = bop_base / dataset
    path_to_split = path_to_dataset / split

    all_images = []
    all_segmentations = []
    object_classes = []
    dino_cls_descriptors = []

    dino_descriptor = descriptor_from_hydra(device)

    sequences = sorted(path_to_split.iterdir())
    for sequence in tqdm(sequences, desc=f"Processing sequences in {dataset}", total=len(sequences)):

        if not sequence.is_dir():
            continue

        rgb_folder = sequence / 'rgb'
        segmentation_folder = sequence / 'mask_visib'
        scene_gt = sequence / 'scene_gt.json'

        object_id = extract_object_id(scene_gt, 0)[1]

        rgb_files = sorted(rgb_folder.iterdir())
        seg_files = sorted(segmentation_folder.iterdir())
        for image_path, seg_path in tqdm(list(zip(rgb_files, seg_files)),
                                         total=len(rgb_files), desc="Processing images"):

            dino_cls_descriptor, dino_dense_descriptor = dino_descriptor.get_detections_from_files(image_path, seg_path)

            all_images.append(image_path)
            all_segmentations.append(seg_path)
            object_classes.append(object_id)
            dino_cls_descriptors.append(dino_cls_descriptor.squeeze().numpy(force=True))

    object_classes = np.array(object_classes)
    dino_cls_descriptors = np.array(dino_cls_descriptors)
    all_images = np.array(all_images)
    all_segmentations = np.array(all_segmentations)

    permutation = np.random.permutation(len(all_images))
    all_images = all_images[permutation]
    all_segmentations = all_segmentations[permutation]
    object_classes = object_classes[permutation]
    dino_cls_descriptors = dino_cls_descriptors[permutation]

    cnn = CondensedNearestNeighbour(random_state=42, n_jobs=8, n_neighbors=1)
    cnn.fit_resample(dino_cls_descriptors, object_classes)
    sample_indices = cnn.sample_indices_

    result_save_path = cache_base_path / dataset / split
    shutil.rmtree(result_save_path, ignore_errors=True)

    for index in sample_indices:

        object_id = object_classes[index]
        obj_save_dir = result_save_path / f'obj_{object_id:06d}'
        images_save_dir = obj_save_dir / 'rgb'
        segmentation_save_dir = obj_save_dir / 'mask_visib'
        images_save_dir.mkdir(parents=True, exist_ok=True)
        segmentation_save_dir.mkdir(parents=True, exist_ok=True)

        image_path = all_images[index]
        segmentation_path = all_segmentations[index]

        shutil.copy2(image_path, images_save_dir / f'{image_path.stem}_{index}{image_path.suffix}')
        shutil.copy2(segmentation_path, segmentation_save_dir / f'{segmentation_path.stem}_{index}'
                                                                f'{segmentation_path.suffix}')


def load_condensed_data(cache_root: Path, dataset: str, split: str) -> Tuple[Dict[int, torch.Tensor], ...]:

    dataset_path = cache_root / dataset / split
    descriptor = descriptor_from_hydra()

    images_dict: Dict[int, Any] = defaultdict(list)
    segmentations_dict: Dict[int, Any] = defaultdict(list)
    cls_descriptors_dict: Dict[int, Any] = defaultdict(list)

    obj_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir() and d.name.startswith('obj_')])

    for obj_dir in obj_dirs:

        obj_id = int(obj_dir.stem.split('_')[1])

        rgb_dir = obj_dir / 'rgb'
        mask_dir = obj_dir / 'mask_visib'

        # Get all image files
        rgb_files = sorted(rgb_dir.glob('*'))
        mask_files = sorted(mask_dir.glob('*'))

        for rgb_file, mask_file in zip(rgb_files, mask_files):
            # Load RGB image
            rgb_img = Image.open(rgb_file).convert('RGB')
            rgb_tensor = transforms.ToTensor()(rgb_img)
            images_dict[obj_id].append(rgb_tensor)

            # Load segmentation mask
            mask_img = Image.open(mask_file).convert('L')  # Grayscale
            mask_array = np.array(mask_img)
            mask_tensor = torch.from_numpy(mask_array)
            segmentations_dict[obj_id].append(mask_tensor)

            cls_descriptor, _ = descriptor.get_detections_from_files(rgb_file, mask_file)
            cls_descriptors_dict[obj_id].append(cls_descriptor)

        images_dict[obj_id] = torch.stack(images_dict[obj_id])
        segmentations_dict[obj_id] = torch.stack(segmentations_dict[obj_id])
        cls_descriptors_dict[obj_id] = torch.stack(cls_descriptors_dict[obj_id])

    return images_dict, segmentations_dict, cls_descriptors_dict


def perform_condensation_for_datasets(bop_base_path: Path, cache_base_path: Path, device='cuda'):

    sequences = [
        ('hope', 'onboarding_static'),
        ('hope', 'onboarding_dynamic'),
        ('handal', 'onboarding_static'),
        ('handal', 'onboarding_dynamic'),
        ('tless', 'train_primesense'),
        ('lmo', 'train'),
        ('icbin', 'train'),
    ]

    for dataset, split in tqdm(sequences, desc="Processing datasets", total=len(sequences)):
        perform_condensation_per_dataset(bop_base_path, cache_base_path, dataset, split, device)


if __name__ == '__main__':

    _cache_base_path = Path('/mnt/personal/jelint19/cache/detections_templates_cache')
    _bop_base = Path('/mnt/personal/jelint19/data/bop')
    _device = 'cuda'

    perform_condensation_for_datasets(_bop_base, _cache_base_path, _device)


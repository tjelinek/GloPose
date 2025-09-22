import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.sparse import issparse
from sklearn import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing
from tqdm import tqdm
from PIL import Image
from imblearn.under_sampling import CondensedNearestNeighbour

from utils.bop_challenge import extract_object_id

sys.path.append('./repositories/cnos')
from src.model.dinov2 import descriptor_from_hydra


def _to_np_f32(X):
    if torch is not None and isinstance(X, torch.Tensor):
        return X.numpy(force=True).astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def _to_np_labels(y):
    if torch is not None and isinstance(y, torch.Tensor):
        return y.numpy(force=True)
    return np.asarray(y)


def imblearn_fitresample_adapted(X, y, n_seeds_S=1, random_state=None):
    X = _to_np_f32(X)
    y = _to_np_labels(y)

    estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=16)

    random_state = np.random.default_rng(random_state)
    idx_under = np.empty((0,), dtype=int)

    estimators_ = []
    for target_class in np.unique(y):
        # Randomly get one sample from the majority class
        # Generate the index to select
        idx_maj = np.flatnonzero(y == target_class)
        idx_maj_sample = idx_maj[
            random_state.integers(
                low=0,
                high=idx_maj.size,
                size=n_seeds_S,
            )
        ]

        # Create the set C - One majority samples and all minority
        C_indices = idx_maj_sample
        C_x = _safe_indexing(X, C_indices)
        C_y = _safe_indexing(y, C_indices)

        # Create the set S - all majority samples
        S_indices = np.flatnonzero(y == target_class)
        S_x = _safe_indexing(X, S_indices)
        S_y = _safe_indexing(y, S_indices)

        # fit knn on C
        estimators_.append(clone(estimator).fit(C_x, C_y))

        good_classif_label = idx_maj_sample.copy()
        # Check each sample in S if we keep it or drop it
        for idx_sam, (x_sam, y_sam) in enumerate(zip(S_x, S_y)):
            # Do not select sample which are already well classified
            if idx_sam in good_classif_label:
                continue

            # Classify on S
            if not issparse(x_sam):
                x_sam = x_sam.reshape(1, -1)
            pred_y = estimators_[-1].predict(x_sam)

            # If the prediction do not agree with the true label
            # append it in C_x
            if y_sam != pred_y:
                # Keep the index for later
                idx_maj_sample = np.append(idx_maj_sample, idx_maj[idx_sam])

                # Update C
                C_indices = np.append(C_indices, idx_maj[idx_sam])
                C_x = _safe_indexing(X, C_indices)
                C_y = _safe_indexing(y, C_indices)

                # fit a knn on C
                estimators_[-1].fit(C_x, C_y)

                # This experimental to speed up the search
                # Classify all the element in S and avoid to test the
                # well classified elements
                pred_S_y = estimators_[-1].predict(S_x)
                good_classif_label = np.unique(
                    np.append(idx_maj_sample, np.flatnonzero(pred_S_y == S_y))
                )

        idx_under = np.concatenate((idx_under, idx_maj_sample), axis=0)

    sample_indices_ = idx_under

    return sample_indices_


def harts_cnn_faiss_original(X, y, random_state=None, max_iterations=100):
    X = _to_np_f32(X)
    y = _to_np_labels(y)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    start = rng.integers(0, n)
    S = np.array([start], dtype=int)
    changed = True
    it = 0
    while changed and it < max_iterations:
        changed = False
        it += 1
        knn = KNeighborsClassifier(n_neighbors=1, n_jobs=16)
        knn.fit(X[S], y[S])
        for i in range(n):
            pred = knn.predict(X[i:i+1])[0]
            if pred != y[i]:
                S = np.append(S, i)
                knn.fit(X[S], y[S])
                changed = True
    return np.sort(np.unique(S))


def harts_cnn_faiss_symmetric(X, y, n_seeds_S=1, random_state=None, max_iterations=100):
    X = _to_np_f32(X)
    y = _to_np_labels(y)
    rng = np.random.default_rng(random_state)
    classes = np.unique(y)
    selected = []
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        idx_rest = np.flatnonzero(y != c)
        if idx_c.size == 0:
            continue
        seeds = idx_c[rng.integers(0, idx_c.size, size=n_seeds_S)]
        C = np.concatenate([idx_rest, seeds])
        S_cls = idx_c
        changed = True
        it = 0
        while changed and it < max_iterations:
            changed = False
            it += 1
            knn = KNeighborsClassifier(n_neighbors=1, n_jobs=16)
            knn.fit(X[C], y[C])
            for s in S_cls:
                pred = knn.predict(X[s:s+1])[0]
                if pred != y[s]:
                    C = np.append(C, s)
                    knn.fit(X[C], y[C])
                    changed = True
        selected.append(np.unique(np.append(seeds, np.intersect1d(C, S_cls))))
    if len(selected) == 0:
        return np.array([], dtype=int)
    return np.sort(np.unique(np.concatenate(selected)))


@torch.inference_mode
def perform_condensation_per_dataset(bop_base: Path, cache_base_path: Path, dataset: str, split: str,
                                     method: str = 'hart_symmetric', descriptors_cache_path: Path = None,
                                     device='cuda'):
    path_to_dataset = bop_base / dataset
    path_to_split = path_to_dataset / split

    all_images = []
    all_segmentations = []
    object_classes = []
    dino_cls_descriptors = []

    dino_descriptor = descriptor_from_hydra(device)

    sequences = sorted(path_to_split.iterdir())
    cnn = CondensedNearestNeighbour(random_state=42, n_jobs=8, n_neighbors=1)

    for sequence in tqdm(sequences, desc=f"Processing sequences in {dataset}", total=len(sequences)):

        if not sequence.is_dir():
            continue

        rgb_folder = sequence / 'rgb'
        segmentation_folder = sequence / 'mask_visib'
        scene_gt = sequence / 'scene_gt.json'

        object_id = extract_object_id(scene_gt, 0)[1]

        rgb_files = sorted(rgb_folder.iterdir())
        seg_files = sorted(segmentation_folder.iterdir())

        # Prepare cache directory for this sequence if caching is enabled
        sequence_cache_dir = None
        if descriptors_cache_path is not None:
            sequence_cache_dir = descriptors_cache_path / dataset / split / sequence.name
            sequence_cache_dir.mkdir(parents=True, exist_ok=True)

        for image_path, seg_path in tqdm(list(zip(rgb_files, seg_files)),
                                         total=len(rgb_files), desc="Processing images"):

            # Check cache and compute descriptor
            cache_file_path = descriptors_cache_path / dataset / split / sequence.name / f'{image_path.stem}.pt' \
                if descriptors_cache_path else None

            if cache_file_path and cache_file_path.exists():
                dino_cls_descriptor = torch.load(cache_file_path, map_location=device)
            else:
                dino_cls_descriptor, dino_dense_descriptor = dino_descriptor.get_detections_from_files(image_path,
                                                                                                       seg_path)
                if cache_file_path:
                    torch.save(dino_cls_descriptor.cpu(), cache_file_path)

            all_images.append(image_path)
            all_segmentations.append(seg_path)
            object_classes.append(object_id)
            dino_cls_descriptors.append(dino_cls_descriptor.squeeze())

    object_classes = torch.tensor(object_classes).to(device)
    dino_cls_descriptors = torch.stack(dino_cls_descriptors)
    all_images = np.array(all_images)
    all_segmentations = np.array(all_segmentations)

    permutation = np.random.permutation(len(all_images))
    all_images = all_images[permutation]
    all_segmentations = all_segmentations[permutation]
    object_classes = object_classes[permutation]
    dino_cls_descriptors = dino_cls_descriptors[torch.tensor(permutation).to(device)]

    if method == "hart_imblearn":
        dino_cls_descriptors = dino_cls_descriptors.numpy(force=True)
        dino_cls_descriptors = np.append(dino_cls_descriptors, np.zeros([1, 1024]), axis=0)
        object_classes = np.append(object_classes.numpy(force=True), -1)
        cnn.fit_resample(dino_cls_descriptors, object_classes)
        sample_indices = cnn.sample_indices_
    elif method == 'hart_imblearn_adapted':
        sample_indices = imblearn_fitresample_adapted(dino_cls_descriptors, object_classes)
    elif method == "hart_symmetric":
        sample_indices = harts_cnn_faiss_symmetric(dino_cls_descriptors, object_classes)
    elif method == 'hart':
        sample_indices = harts_cnn_faiss_original(dino_cls_descriptors, object_classes)
    else:
        raise ValueError(f"Method {method} not recognized")

    result_save_path = cache_base_path / dataset / split
    shutil.rmtree(result_save_path, ignore_errors=True)

    for index in sample_indices:

        if object_classes[index] == -1:
            continue

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


def get_descriptors_for_condensed_templates(path_to_detections: Path, black_background: bool = False) -> \
        Tuple[Dict[int, torch.Tensor], ...]:

    descriptor = descriptor_from_hydra()

    images_dict: Dict[int, Any] = defaultdict(list)
    segmentations_dict: Dict[int, Any] = defaultdict(list)
    cls_descriptors_dict: Dict[int, Any] = defaultdict(list)
    patch_descriptors_dict: Dict[int, Any] = defaultdict(list)

    obj_dirs = sorted([d for d in path_to_detections.iterdir() if d.is_dir() and d.name.startswith('obj_')])

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

            cls_descriptor, patch_descriptor = descriptor.get_detections_from_files(rgb_file, mask_file,
                                                                                    black_background)
            cls_descriptors_dict[obj_id].append(cls_descriptor.squeeze(0))
            patch_descriptors_dict[obj_id].append(patch_descriptor.squeeze(0))

        images_dict[obj_id] = torch.stack(images_dict[obj_id])
        segmentations_dict[obj_id] = torch.stack(segmentations_dict[obj_id])
        cls_descriptors_dict[obj_id] = torch.stack(cls_descriptors_dict[obj_id])
        patch_descriptors_dict[obj_id] = torch.stack(patch_descriptors_dict[obj_id])

    return images_dict, segmentations_dict, cls_descriptors_dict, patch_descriptors_dict


def perform_condensation_for_datasets(bop_base_path: Path, cache_base_path: Path, descriptors_cache_path=None,
                                      device='cuda'):
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
        perform_condensation_per_dataset(bop_base_path, cache_base_path, dataset, split, 'hart_imblearn_adapted',
                                         descriptors_cache_path, device)


if __name__ == '__main__':

    experiment_name = '1nn'
    _cache_base_path = Path('/mnt/personal/jelint19/cache/detections_templates_cache') / experiment_name
    _descriptors_cache_path = Path('/mnt/personal/jelint19/cache/DINOv2_cache')
    _bop_base = Path('/mnt/personal/jelint19/data/bop')
    _device = 'cuda'

    perform_condensation_for_datasets(_bop_base, _cache_base_path, _descriptors_cache_path, _device)

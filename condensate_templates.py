import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from threadpoolctl import threadpool_limits

threadpool_limits(limits=1)

import pickle
import shutil
import sys
import argparse
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as TVF
from scipy.sparse import issparse
from sklearn import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing
from sklearn.utils import check_random_state
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from PIL import Image
from imblearn.under_sampling import CondensedNearestNeighbour
from repositories.cnos.segment_anything.utils.amg import rle_to_mask

from data_structures.template_bank import TemplateBank  # noqa: F401 — re-exported for backward compat
from utils.bop_challenge import extract_object_id
from utils.detection_utils import average_patch_similarity

sys.path.append('./repositories/cnos')
from src.model.dinov2 import descriptor_from_hydra

warnings.filterwarnings('ignore', message='The number of unique classes is greater than 50%', category=UserWarning)


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

    estimator = KNeighborsClassifier(n_neighbors=1, n_jobs=1)

    random_state = check_random_state(random_state)
    target_stats = Counter(y)
    idx_under = np.empty((0,), dtype=int)

    estimators_ = []
    for target_class in np.unique(y):
        idx_maj = np.flatnonzero(y == target_class)
        idx_maj_sample = idx_maj[
            random_state.randint(
                low=0,
                high=target_stats[target_class],
                size=n_seeds_S,
            )
        ]

        # Create the set C - One majority samples and all minority
        C_indices = np.append(np.flatnonzero(y != target_class), idx_maj_sample)
        C_x = _safe_indexing(X, C_indices)
        C_y = _safe_indexing(y, C_indices)

        # Create the set S - all majority samples
        S_indices = idx_maj
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


def harts_cnn_original(
        X: torch.Tensor,
        y: torch.Tensor,
        X_patch: Optional[torch.Tensor] = None,
        min_cls_cosine_similarity: float = 0.15,
        min_avg_patch_cosine_similarity: float = 0.15,
        patch_descriptor_filtering: bool = True,
        use_segmentation: bool = True,
        segmentation_masks: Optional[List[torch.Tensor]] = None,
        random_state: Optional[int] = 42,
        max_iterations: int = 100,
) -> torch.Tensor:
    n, d = X.shape
    device = X.device

    X = F.normalize(X, dim=1)

    rng = torch.Generator(device=device)
    rng.manual_seed(random_state)

    start = torch.randint(0, n, (1,), generator=rng, device=device).item()
    S = torch.tensor([start], dtype=torch.long, device=device)

    changed = True
    it = 0

    pbar = tqdm(total=max_iterations, desc='Hart algorithm iterations')
    while changed and it < max_iterations:
        pbar.update(1)

        changed = False
        it += 1
        for i in range(n):

            cosine_sim = X[i:i + 1] @ X[S].T
            topk_vals, topk_idx = torch.topk(cosine_sim, k=1, dim=1)
            y_pred = y[S][topk_idx.squeeze(1)]

            patchwise_similar = True
            if patch_descriptor_filtering:
                S_pred = S[topk_idx]
                X_patch_C_topk = X_patch[S_pred.cpu().item()].unsqueeze(0) if X_patch is not None else None
                segmentation_C_topk = segmentation_masks[S_pred.item()].unsqueeze(0) if use_segmentation else None

                segmentation_s = segmentation_masks[i].unsqueeze(0) if use_segmentation else None
                X_patch_s = X_patch[i:i + 1]

                avg_patch_sim = average_patch_similarity(X_patch_s, X_patch_C_topk, segmentation_s,
                                                         segmentation_C_topk, use_segmentation)

                patchwise_similar = avg_patch_sim >= min_avg_patch_cosine_similarity

            if y_pred.item() != y[i] or topk_vals.item() < min_cls_cosine_similarity or not patchwise_similar:
                S = torch.cat([S, torch.tensor([i], dtype=torch.long, device=device)], dim=0)

                changed = True

    S = torch.unique(S)
    S, _ = torch.sort(S)

    return S


def harts_cnn_symmetric(
        X: torch.Tensor,
        y: torch.Tensor,
        X_patch: Optional[torch.Tensor] = None,
        min_cls_cosine_similarity: float = 0.15,
        min_avg_patch_cosine_similarity: float = 0.15,
        patch_descriptor_filtering: bool = True,
        use_segmentation: bool = True,
        segmentation_masks: Optional[List[torch.Tensor]] = None,
        random_state: Optional[int] = 42,
        max_iterations: int = 100,
) -> torch.Tensor:
    device = X.device

    generator = torch.Generator(device=device).manual_seed(random_state)

    X_norm = F.normalize(X, dim=1)
    classes = torch.unique(y)
    selected = []
    for c in tqdm(classes, desc='Hart symmetric algorithm classes', total=len(classes)):
        idx_c = torch.nonzero(y == c, as_tuple=True)[0]
        idx_rest = torch.nonzero(y != c, as_tuple=True)[0]
        if idx_c.numel() == 0:
            continue
        seed_indices = torch.randint(0, idx_c.numel(), (random_state,), device=device, generator=generator)
        seeds = idx_c[seed_indices]
        C = torch.cat([idx_rest, seeds])
        S_cls = idx_c
        changed = True
        it = 0
        while changed and it < max_iterations:
            changed = False
            it += 1
            X_C_norm = X_norm[C]
            y_C = y[C]
            for s in S_cls:
                cosine_sim = X_norm[s:s + 1] @ X_C_norm.T
                topk_vals, topk_idx = torch.topk(cosine_sim, k=1, dim=1)
                topk_idx = topk_idx.squeeze(1)
                y_pred = y_C[topk_idx]

                patchwise_similar = True
                if patch_descriptor_filtering:
                    C_pred = C[topk_idx]
                    X_patch_C_topk = X_patch[C_pred.cpu()] if X_patch is not None else None
                    segmentation_C_topk = segmentation_masks[C_pred.item()].unsqueeze(0) if use_segmentation else None

                    segmentation_s = segmentation_masks[s].unsqueeze(0) if use_segmentation else None
                    X_patch_s = X_patch[s:s + 1]

                    avg_patch_sim = average_patch_similarity(X_patch_s, X_patch_C_topk, segmentation_s,
                                                             segmentation_C_topk, use_segmentation)

                    patchwise_similar = avg_patch_sim >= min_avg_patch_cosine_similarity

                if (y_pred.item() != y[s] or topk_vals.item() < min_cls_cosine_similarity or not patchwise_similar):
                    C = torch.cat([C, s.view(1)])
                    changed = True
        C_set = set(C.tolist())
        S_cls_set = set(S_cls.tolist())
        intersection = torch.tensor(list(C_set & S_cls_set), dtype=torch.long, device=device)
        selected_for_class = torch.unique(torch.cat([seeds, intersection]))
        selected.append(selected_for_class)
    if len(selected) == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    result = torch.unique(torch.cat(selected))
    S, _ = torch.sort(result)

    return S


def _l2n(x, eps=1e-12):
    # L2-normalize vectors along the last dimension
    # Adds epsilon clamp to avoid division by zero
    if isinstance(x, torch.Tensor):
        n = torch.norm(x, dim=-1, keepdim=True).clamp_min(eps)
        return x / n
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n < eps] = eps
    return x / n


def _fit_whitener(X, out_dim=0, eps=1e-6):
    # Compute dataset mean and whitening projection (PCA-whitening)
    X = _to_np_f32(X)
    X = _l2n(X)  # Normalize features before PCA
    mu = X.mean(0, keepdims=True)  # Global mean
    Xc = X - mu
    # Perform SVD on mean-centered data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # Optionally keep only top 'out_dim' components for dimensionality reduction
    if out_dim and out_dim > 0 and out_dim < Vt.shape[0]:
        Vt = Vt[:out_dim]
        S = S[:out_dim]
    # Compute whitening scales (inverse std for each principal component)
    scale = 1.0 / np.sqrt((S ** 2) / max(1, X.shape[0] - 1) + eps)
    # Compute whitening matrix: projection + scaling
    W = (Vt.T * scale).astype(np.float32)
    # Return mean and whitening matrix
    return mu.astype(np.float32), W


def _apply_whitener(X, mu, W):
    # Apply whitening: subtract mean, project, and re-normalize
    if isinstance(X, torch.Tensor):
        Xw = (X - mu) @ W
        Xw = _l2n(Xw)
        return Xw.to(torch.float32)
    else:
        X = _to_np_f32(X)
        Xw = (X - mu) @ W
        Xw = _l2n(Xw)
        return Xw.astype(np.float32)


def _csls_avg(X, k=10):
    # Compute average cosine similarity to k nearest neighbors (for CSLS)
    X = _to_np_f32(X)
    S = X @ X.T  # Pairwise cosine similarities
    np.fill_diagonal(S, -np.inf)  # Ignore self-similarity
    # Find indices of top-k neighbors per sample
    idx = np.argpartition(-S, kth=min(k, S.shape[1] - 1) - 1, axis=1)[:, :k]
    # Average their similarities
    avgs = (S[np.arange(S.shape[0])[:, None], idx]).mean(axis=1)
    return avgs.astype(np.float32)


def _compute_stats(X, y, csls_k=10):
    # Compute per-template and per-class statistics for OOD gating and normalization
    X = _to_np_f32(X)
    X = _l2n(X)
    # Compute per-template local CSLS neighborhood averages
    csls_avg = _csls_avg(X, k=csls_k) if X.shape[0] > 1 else np.zeros((X.shape[0],), dtype=np.float32)
    # Estimate tied covariance using Ledoit–Wolf shrinkage (robust in high-d)
    clf = LedoitWolf().fit(X)
    Sigma_inv = np.linalg.pinv(clf.covariance_).astype(np.float32)
    # Compute class means for Mahalanobis distance
    classes = np.unique(y)
    mu_c = {}
    for c in classes:
        mu_c[int(c)] = X[y == c].mean(0).astype(np.float32)
    # Return all computed stats in torch format
    return {
        'template_csls_avg': torch.from_numpy(csls_avg),
        'sigma_inv': torch.from_numpy(Sigma_inv),
        'class_means': {k: torch.from_numpy(v) for k, v in mu_c.items()},
    }


def perform_condensation_per_dataset(bop_base: Path, cache_base_path: Path, dataset: str, split: str,
                                     method: str = 'hart_symmetric', descriptor_model='dinov2',
                                     descriptor_mask_detections=True, descriptors_cache_path: Path = None,
                                     min_cls_cosine_similarity: float = 0.15,
                                     min_avg_patch_cosine_similarity: float = 0.15, device='cuda', whiten_dim: int = 0,
                                     csls_k: int = 10, onboarding_augmentations_path: Path = None,
                                     train_pbr_augmentations_path: Path = None,
                                     augment_with_split_detections: bool = False,
                                     augment_with_train_pbr_detections: bool = False,
                                     augmentations_detector: str = None, patch_descriptors_filtering: bool = False):
    path_to_dataset = bop_base / dataset
    path_to_split = path_to_dataset / split

    all_images = []
    all_segmentations = []
    object_classes = []
    dino_cls_descriptors = []
    dino_patch_descriptors = []

    dino_descriptor = descriptor_from_hydra(descriptor_model, descriptor_mask_detections, device=device)

    sequences = sorted(path_to_split.iterdir())
    cnn = CondensedNearestNeighbour(random_state=42, n_jobs=8, n_neighbors=1)

    for sequence in tqdm(sequences, desc=f"Processing sequences in {dataset}", total=len(sequences)):

        if not sequence.is_dir():
            continue

        if 'aria' in split and 'static' in split:
            rgb_folder = sequence / 'rgb'
            segmentation_folder = sequence / 'mask_visib_rgb'
            scene_gt = sequence / 'scene_gt_rgb.json'
        elif 'quest3' in split and 'static' in split:
            rgb_folder = sequence / 'gray1'
            segmentation_folder = sequence / 'mask_visib_gray1'
            scene_gt = sequence / 'scene_gt_gray1.json'
        else:
            rgb_folder = sequence / 'rgb'
            segmentation_folder = sequence / 'mask_visib'
            scene_gt = sequence / 'scene_gt.json'

        object_id = extract_object_id(scene_gt, 0)[1]

        rgb_files = sorted(rgb_folder.iterdir())
        seg_files = sorted(segmentation_folder.iterdir())

        if descriptors_cache_path is not None:
            sequence_cache_dir = descriptors_cache_path / dataset / split / sequence.name
            sequence_cache_dir.mkdir(parents=True, exist_ok=True)

        for image_path, seg_path in tqdm(list(zip(rgb_files, seg_files)),
                                         total=len(rgb_files), desc="Processing images"):

            # Check cache and compute descriptor
            cache_file_path = descriptors_cache_path / dataset / split / sequence.name / f'{image_path.stem}.pt' \
                if descriptors_cache_path else None

            descriptors_loaded_successfully = False
            if cache_file_path and cache_file_path.exists():
                loaded_descriptors = torch.load(cache_file_path, map_location=device, weights_only=True)
                if type(loaded_descriptors) is tuple:
                    dino_cls_descriptor, dino_dense_descriptor = loaded_descriptors
                    descriptors_loaded_successfully = True

            if not descriptors_loaded_successfully:
                dino_cls_descriptor, dino_dense_descriptor = dino_descriptor.get_detections_from_files(image_path,
                                                                                                       seg_path)
                if cache_file_path:
                    dino_cls_descriptor_to_save = dino_cls_descriptor.detach().clone().cpu()
                    dino_dense_descriptor_to_save = dino_dense_descriptor.detach().clone().cpu()

                    payload = (dino_cls_descriptor_to_save, dino_dense_descriptor_to_save)
                    torch.save(payload, cache_file_path)

            seg_img = Image.open(seg_path)
            seg_array = TVF.functional.pil_to_tensor(seg_img).to(torch.bool).squeeze(0).cpu()

            all_images.append(image_path)
            all_segmentations.append(seg_array)
            object_classes.append(object_id)
            dino_cls_descriptors.append(dino_cls_descriptor)
            dino_patch_descriptors.append(dino_dense_descriptor.cpu())

    num_images = len(all_images)

    X_cls_pbr = None
    if (train_pbr_augmentations_path is not None and train_pbr_augmentations_path.exists()
            and augment_with_train_pbr_detections):
        path_to_pbr = path_to_split.parent / 'train_pbr'
        X_cls_pbr, X_patch_pbr, y_pbr, image_paths_pbr, masks_pbr = \
            get_detections_descriptors(augmentations_detector, dataset, path_to_pbr, descriptor_model,
                                       train_pbr_augmentations_path, skip=10, device=device)
        dino_cls_descriptors.extend(X_cls_pbr)
        dino_patch_descriptors.extend(X_patch_pbr)
        object_classes.extend(y_pbr)
        all_images.extend(image_paths_pbr)
        all_segmentations.extend(masks_pbr)

    X_cls_onboarding = None
    if (onboarding_augmentations_path is not None and onboarding_augmentations_path.exists()
            and augment_with_split_detections):
        X_cls_onboarding, X_patch_onboarding, y_onboarding, image_paths_onboarding, masks_onboarding = \
            get_detections_descriptors(augmentations_detector, dataset, path_to_split, descriptor_model,
                                       onboarding_augmentations_path, device=device)
        dino_cls_descriptors.extend(X_cls_onboarding)
        dino_patch_descriptors.extend(X_patch_onboarding)
        object_classes.extend(y_onboarding)
        all_images.extend(image_paths_onboarding)
        all_segmentations.extend(masks_onboarding)

    object_classes = torch.tensor(object_classes).to(device)
    dino_cls_descriptors = torch.cat(dino_cls_descriptors)
    dino_patch_descriptors = torch.cat(dino_patch_descriptors)

    permutation = np.random.permutation(len(all_images))
    permutation_tensor = torch.tensor(permutation).to(device)
    all_images = [all_images[i] for i in permutation]
    all_segmentations = [all_segmentations[i] for i in permutation]
    object_classes = object_classes[permutation]
    dino_cls_descriptors = dino_cls_descriptors[permutation_tensor]
    dino_patch_descriptors = dino_patch_descriptors[permutation_tensor.cpu()]

    X_cls_np = dino_cls_descriptors.numpy(force=True)
    X_cls_np = _l2n(X_cls_np).astype(np.float32)
    y_np = object_classes.numpy(force=True)

    mu_w, W_w = _fit_whitener(X_cls_np, out_dim=min(whiten_dim, X_cls_np.shape[1]))
    if whiten_dim and whiten_dim > 0:
        X_for_selection = _apply_whitener(X_cls_np, mu_w, W_w)
        Xw_np = _apply_whitener(X_cls_np, mu_w, W_w)
    else:
        mu_w, W_w = None, None
        Xw_np = X_cls_np
        X_for_selection = X_cls_np

    stats = _compute_stats(Xw_np, y_np, csls_k=csls_k)

    if method == "hart_imblearn":
        dino_cls_descriptors_np = X_for_selection
        dino_cls_dimension = dino_cls_descriptors_np.shape[1]
        dino_cls_descriptor_np = np.append(dino_cls_descriptors_np, np.zeros([1, dino_cls_dimension]), axis=0)
        object_classes = np.append(object_classes.cpu().numpy(), -1)
        cnn.fit_resample(dino_cls_descriptor_np, object_classes)
        sample_indices = cnn.sample_indices_
    elif method == 'hart_imblearn_adapted':
        sample_indices = imblearn_fitresample_adapted(torch.from_numpy(X_for_selection), object_classes)
    elif method == "hart_symmetric":
        sample_indices = harts_cnn_symmetric(torch.from_numpy(X_for_selection).to(device), object_classes,
                                             X_patch=dino_patch_descriptors,
                                             patch_descriptor_filtering=patch_descriptors_filtering,
                                             use_segmentation=descriptor_mask_detections,
                                             min_cls_cosine_similarity=min_cls_cosine_similarity,
                                             min_avg_patch_cosine_similarity=min_avg_patch_cosine_similarity,
                                             segmentation_masks=all_segmentations).numpy(force=True)
    elif method == 'hart':
        sample_indices = harts_cnn_original(
            torch.from_numpy(X_for_selection).to(device),
            object_classes,
            X_patch=dino_patch_descriptors,
            patch_descriptor_filtering=patch_descriptors_filtering,
            use_segmentation=descriptor_mask_detections,
            min_cls_cosine_similarity=min_cls_cosine_similarity,
            min_avg_patch_cosine_similarity=min_avg_patch_cosine_similarity,
            segmentation_masks=all_segmentations
        ).numpy(force=True)
    else:
        raise ValueError(f"Method {method} not recognized")

    result_save_path = cache_base_path / dataset / split
    shutil.rmtree(result_save_path, ignore_errors=True)

    saved_indices = []
    saved_labels = []

    for index in tqdm(sample_indices, desc="Condensed templates saved", total=len(sample_indices)):
        if object_classes[index] == -1:
            continue
        object_id = int(object_classes[index].item())
        obj_save_dir = result_save_path / f'obj_{object_id:06d}'
        images_save_dir = obj_save_dir / 'rgb'
        segmentation_save_dir = obj_save_dir / 'mask_visib'
        descriptors_save_dir = obj_save_dir / 'descriptors'
        images_save_dir.mkdir(parents=True, exist_ok=True)
        segmentation_save_dir.mkdir(parents=True, exist_ok=True)
        descriptors_save_dir.mkdir(parents=True, exist_ok=True)

        image_path = all_images[index]
        segmentation_tensor = all_segmentations[index]

        new_image_name = Path(f'{image_path.stem}_{index}{image_path.suffix}')
        new_seg_name = Path(f'{image_path.stem}_{index}.png')
        descriptor_name = f'{new_image_name.stem}.pt'
        shutil.copy2(image_path, images_save_dir / new_image_name)
        torchvision.utils.save_image(segmentation_tensor.float(), segmentation_save_dir / new_seg_name, )

        cls_descriptor_to_save = dino_cls_descriptors[index].detach().cpu().clone()
        torch.save(cls_descriptor_to_save, descriptors_save_dir / descriptor_name)
        saved_indices.append(int(index))
        saved_labels.append(object_id)

    stats_dir = result_save_path
    stats_dir.mkdir(parents=True, exist_ok=True)

    if len(saved_indices) > 0:
        idx = torch.tensor(saved_indices, dtype=torch.long)
        y_sel = torch.tensor(saved_labels, dtype=torch.long)

        template_csls_avg_condensed_dict = defaultdict(list)
        template_csls_avg_all_dict = defaultdict(list)
        for i in range(len(idx)):
            template_csls_avg_condensed_dict[y_sel[i].item()].append(stats['template_csls_avg'][idx[i]])
        for i in range(len(object_classes)):
            if object_classes[i] == -1:
                continue
            template_csls_avg_all_dict[object_classes[i].item()].append(stats['template_csls_avg'][i])
        template_csls_avg_condensed_dict = {k: torch.stack(template_csls_avg_condensed_dict[k])
                                            for k in template_csls_avg_condensed_dict.keys()}
        template_csls_avg_all_dict = {k: torch.stack(template_csls_avg_all_dict[k])
                                      for k in template_csls_avg_all_dict.keys()}

        template_csls_avg = stats['template_csls_avg']
        payload = {
            'whitening_mean': None if mu_w is None else torch.from_numpy(mu_w.squeeze(0)),
            'whitening_W': None if W_w is None else torch.from_numpy(W_w),
            'template_indices': idx,
            'template_labels': y_sel,
            'template_csls_avg': template_csls_avg,
            'template_csls_avg_all_dict': template_csls_avg_all_dict,
            'template_csls_avg_condensed_dict': template_csls_avg_condensed_dict,
            'sigma_inv': stats['sigma_inv'],
            'class_means': {int(k): v for k, v in stats['class_means'].items()},
            'orig_onboarding_images': num_images,
            'orig_pbr_images': len(X_cls_pbr) if X_cls_pbr is not None else 0,
            'orig_onboarding_sam_detections': len(X_cls_onboarding) if X_cls_onboarding is not None else 0,
        }

        torch.save(payload, stats_dir / 'csls_stats.pt')


def get_detections_descriptors(augmentations_detector: str, dataset: str, path_to_split: Path, descriptor_model: str,
                               onboarding_augmentations_path: Path, skip: int = 1, device='cpu') -> Any:
    X_cls = []
    X_patch = []
    y_pbr = []
    masks = []
    image_paths = []

    all_augmentations_sequences = sorted(onboarding_augmentations_path.iterdir())
    for sequence in tqdm(all_augmentations_sequences, total=len(all_augmentations_sequences),
                         desc=f'{onboarding_augmentations_path.stem} descriptors of {dataset}'):
        descriptor_dir = sequence / f'cnos_{augmentations_detector}_detections_{descriptor_model}'

        descriptors_files = sorted(descriptor_dir.iterdir())[::skip]
        for descriptor_file in descriptors_files:
            with open(descriptor_file, "rb") as pickle_file:
                payload = pickle.load(pickle_file)

            path_to_sequence = path_to_split / sequence.name
            image_file = next(
                (path_to_sequence / 'rgb' / f'{descriptor_file.stem}{ext}'
                 for ext in ('.jpg', '.png')
                 if (path_to_sequence / 'rgb' / f'{descriptor_file.stem}{ext}').exists()),
                None
            )

            if image_file is None:
                continue

            detection_masks_rle = payload['masks']
            detection_masks_array = \
                [torch.from_numpy(rle_to_mask(rle_mask)).cpu() for rle_mask in detection_masks_rle]

            masks.extend(detection_masks_array)
            X_cls.append(torch.from_numpy(payload['descriptors']).to(device))
            X_patch.append(torch.from_numpy(payload['patch_descriptors']).cpu())
            y_pbr.extend(payload['detections_object_ids'])
            image_paths.extend([image_file] * len(detection_masks_array))

    return X_cls, X_patch, y_pbr, image_paths, masks


def get_descriptors_for_condensed_templates(path_to_detections: Path, descriptor_name: str,
                                            cosine_similarity_quantile: float, mahalanobis_quantile: float,
                                            force_recompute_descriptors: bool = True, device: str = 'cuda') \
        -> TemplateBank:
    descriptor = descriptor_from_hydra(model=descriptor_name, device=device)

    images_dict: Dict[int, Any] = defaultdict(list)
    segmentations_dict: Dict[int, Any] = defaultdict(list)
    cls_descriptors_dict: Dict[int, Any] = defaultdict(list)
    patch_descriptors_dict: Dict[int, Any] = defaultdict(list)

    obj_dirs = sorted([d for d in path_to_detections.iterdir() if d.is_dir() and d.name.startswith('obj_')])

    stats_file = path_to_detections / 'csls_stats.pt'
    stats = torch.load(stats_file, map_location=device, weights_only=True)
    mu_w = stats.get('whitening_mean', None)
    W_w = stats.get('whitening_W', None)
    sigma_inv = stats.get('sigma_inv', None)
    class_means = stats.get('class_means', None)
    csls_avg = stats.get('template_csls_avg', None)

    apply_whitening = (mu_w is not None) and (W_w is not None)
    if apply_whitening:
        mu_w = mu_w.to(device=device, dtype=torch.float32).view(1, -1)
        W_w = W_w.to(device=device, dtype=torch.float32)
    if sigma_inv is not None:
        sigma_inv = sigma_inv.to(device=device, dtype=torch.float32)
    if class_means is not None:
        class_means = {int(k): v.to(device=device, dtype=torch.float32) for k, v in class_means.items()}

    for obj_dir in tqdm(obj_dirs, desc="Loading templates", total=len(obj_dirs)):

        obj_id = int(obj_dir.stem.split('_')[1]) if 'obj' in obj_dir.stem else int(obj_dir.stem)

        rgb_dir = obj_dir / 'rgb'
        mask_dir = obj_dir / 'mask_visib'
        descriptor_dir = obj_dir / 'descriptors'

        # Get all image files
        rgb_files = sorted(rgb_dir.glob('*'))
        mask_files = sorted(mask_dir.glob('*'))

        for rgb_file, mask_file in tqdm(zip(rgb_files, mask_files),
                                        desc=f"Templates for {obj_dir.stem}",
                                        total=len(rgb_files),
                                        leave=False):
            # Load RGB image
            # rgb_img = Image.open(rgb_file).convert('RGB')
            # rgb_tensor = TVF.ToTensor()(rgb_img).cpu()

            descriptor_file = descriptor_dir / f'{rgb_file.stem}.pt'

            images_dict[obj_id].append(rgb_file)

            # Load segmentation mask
            mask_img = Image.open(mask_file).convert('L')  # Grayscale
            mask_array = np.array(mask_img)
            mask_tensor = torch.from_numpy(mask_array).to(device)
            segmentations_dict[obj_id].append(mask_tensor)

            if descriptor_file.exists() and not force_recompute_descriptors:
                cls_descriptor = torch.load(descriptor_file, map_location=device, weights_only=True)
            else:
                cls_descriptor, patch_descriptor = descriptor.get_detections_from_files(rgb_file, mask_file)

            x = cls_descriptor.squeeze(0).to(device, dtype=torch.float32)
            x = _l2n(x)

            if apply_whitening:
                x = _apply_whitener(x, mu_w, W_w)

            cls_descriptors_dict[obj_id].append(x.squeeze(0).detach())
            patch_descriptors_dict[obj_id].append(patch_descriptor.squeeze(0).detach())

        cls_descriptors_dict[obj_id] = torch.stack(cls_descriptors_dict[obj_id])
        patch_descriptors_dict[obj_id] = torch.stack(patch_descriptors_dict[obj_id])

    template_thresholds: Dict[int, torch.Tensor] = {}
    for obj_id, X in cls_descriptors_dict.items():
        if X.shape[0] <= 1:
            template_thresholds[obj_id] = None
            continue
        S = X @ X.T
        diag = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        S = S - 1e9 * diag
        vals, _ = torch.sort(S, dim=1, descending=True)
        per_template_vals = vals[:, 0: max(1, min(vals.shape[1] - 1, X.shape[0] - 1))]
        q = torch.quantile(per_template_vals, q=cosine_similarity_quantile, dim=1, keepdim=False)
        template_thresholds[obj_id] = q

    mahalanobis_thresholds: Optional[Dict[int, torch.Tensor]] = {}
    mahalanobis_threshold_global: Optional[torch.Tensor] = None
    if (sigma_inv is not None) and (class_means is not None):
        all_m = []
        for obj_id, X in cls_descriptors_dict.items():
            mu = class_means.get(int(obj_id), X.mean(dim=0, keepdim=True))
            diff = X - mu
            m = (diff @ sigma_inv * diff).sum(dim=1)
            mahalanobis_thresholds[obj_id] = torch.quantile(m, mahalanobis_quantile)
            all_m.append(m)
        if len(all_m) > 0:
            all_m = torch.cat(all_m, dim=0)
            mahalanobis_threshold_global = torch.quantile(all_m, mahalanobis_quantile)
    else:
        mahalanobis_thresholds = None
        mahalanobis_threshold_global = None

    return TemplateBank(
        images=images_dict,
        masks=segmentations_dict,
        cls_desc=cls_descriptors_dict,
        patch_desc=patch_descriptors_dict,
        template_thresholds=template_thresholds,
        whitening_mean=mu_w,
        whitening_W=W_w,
        sigma_inv=sigma_inv,
        class_means=class_means,
        maha_thresh_per_class=mahalanobis_thresholds,
        maha_thresh_global=mahalanobis_threshold_global,
        template_csls_avg=csls_avg,
    )


def perform_condensation_for_datasets(bop_base_path: Path, cache_base_path: Path, method: str,
                                      descriptors_cache_path=None, descriptor_model='dinov2', device='cuda',
                                      whiten_dim: int = 0, csls_k: int = 10, store_stats: bool = False):
    sequences = [
        ('hot3d', 'object_ref_aria_static_scenewise'),
        ('hot3d', 'object_ref_quest3_static_scenewise'),
        # ('hot3d', 'object_ref_aria_dynamic_scenewise'),
        # ('hot3d', 'object_ref_quest3_dynamic_scenewise'),
        ('hope', 'onboarding_static'),
        ('hope', 'onboarding_dynamic'),
        ('handal', 'onboarding_static'),
        ('handal', 'onboarding_dynamic'),
        ('tless', 'train_primesense'),
        ('lmo', 'train'),
        ('icbin', 'train'),
    ]

    for dataset, split in tqdm(sequences, desc="Processing datasets", total=len(sequences)):
        perform_condensation_per_dataset(bop_base_path, cache_base_path, dataset, split, method, descriptor_model,
                                         descriptors_cache_path=descriptors_cache_path, device=device,
                                         whiten_dim=whiten_dim, csls_k=csls_k)


def main():
    parser = argparse.ArgumentParser(description='Perform template condensation for BOP datasets')
    parser.add_argument('--method', type=str, required=False,
                        choices=['hart', 'hart_symmetric', 'hart_imblearn', 'hart_imblearn_adapted'],
                        default='hart', help='Condensation method to use')
    parser.add_argument('--descriptor', type=str, required=False,
                        choices=['dinov2', 'dinov3'], default='dinov2',
                        help='Descriptor model to use')
    parser.add_argument('--dataset', type=str, required=False,
                        choices=['hope', 'handal', 'tless', 'lmo', 'icbin', 'hot3d'],
                        default='hope', help='Dataset to process')
    parser.add_argument('--split', type=str, required=False,
                        default='onboarding_static', help='Dataset split to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--whiten_dim', type=int, default=0,
                        help='PCA-whitening output dim; 0 disables whitening')
    parser.add_argument('--descriptor_mask_detections', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--csls_k', type=int, default=10,
                        help='k for CSLS avg computation in stats')
    parser.add_argument('--augment_with_split_detections', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--augment_with_train_pbr_detections', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--augmentations_detector', type=str, default='sam2')
    parser.add_argument('--min_cls_cosine_similarity', type=float, default=0.15)
    parser.add_argument('--min_avg_patch_cosine_similarity', type=float, default=0.15)
    parser.add_argument('--patch_descriptors_filtering', type=lambda x: bool(int(x)), default=True)

    args = parser.parse_args()

    # Define paths
    experiment_name = f'1nn-{args.method}-{args.descriptor}'
    if args.whiten_dim > 0:
        experiment_name += f'-whitening_{args.whiten_dim}'
    if args.descriptor_mask_detections < 1:
        experiment_name += f'_nonMaskedBG'
    if args.augment_with_split_detections:
        experiment_name += f'_aug-split'
    if args.augment_with_train_pbr_detections:
        experiment_name += f'_aug-pbr'
    if args.min_cls_cosine_similarity > 0:
        experiment_name += f'_min-cls-sim-{args.min_cls_cosine_similarity}'
    if args.patch_descriptors_filtering:
        experiment_name += f'_min-patch-sim-{args.min_avg_patch_cosine_similarity}'
    cache_base_path = Path('/mnt/personal/jelint19/cache/detections_templates_cache') / experiment_name
    descriptors_cache_path = Path(f'/mnt/personal/jelint19/cache/{args.descriptor}_cache/bop')
    detections_cache_path = Path(f'/mnt/personal/jelint19/cache/detections_cache/{args.dataset}')
    onboarding_augmentations_path = detections_cache_path / f'{args.split}'
    train_pbr_augmentations_path = detections_cache_path / 'train_pbr'
    bop_base = Path('/mnt/data/vrg/public_datasets/bop')

    print(f"Processing {args.dataset}/{args.split} with method {args.method} and descriptor {args.descriptor}")

    # Perform condensation for a single dataset/split
    perform_condensation_per_dataset(bop_base, cache_base_path, args.dataset, args.split, args.method, args.descriptor,
                                     descriptor_mask_detections=args.descriptor_mask_detections,
                                     descriptors_cache_path=descriptors_cache_path,
                                     min_cls_cosine_similarity=args.min_cls_cosine_similarity,
                                     min_avg_patch_cosine_similarity=args.min_avg_patch_cosine_similarity,
                                     device=args.device, whiten_dim=args.whiten_dim, csls_k=args.csls_k,
                                     onboarding_augmentations_path=onboarding_augmentations_path,
                                     train_pbr_augmentations_path=train_pbr_augmentations_path,
                                     augment_with_split_detections=args.augment_with_split_detections,
                                     augment_with_train_pbr_detections=args.augment_with_train_pbr_detections,
                                     augmentations_detector=args.augmentations_detector,
                                     patch_descriptors_filtering=args.patch_descriptors_filtering)


if __name__ == '__main__':
    main()

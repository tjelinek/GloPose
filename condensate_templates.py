import shutil
import sys
import argparse
import warnings
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.sparse import issparse
from sklearn import clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import _safe_indexing
from sklearn.utils import check_random_state
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from PIL import Image
from imblearn.under_sampling import CondensedNearestNeighbour

from utils.bop_challenge import extract_object_id

sys.path.append('./repositories/cnos')
from src.model.dinov2 import descriptor_from_hydra

warnings.filterwarnings('ignore', message='The number of unique classes is greater than 50%', category=UserWarning)


@dataclass
class TemplateBank:
    images: Dict[int, torch.Tensor] = None
    masks: Dict[int, torch.Tensor] = None
    cls_desc: Dict[int, torch.Tensor] = None
    patch_desc: Dict[int, torch.Tensor] = None
    template_thresholds: Dict[int, torch.Tensor] = None
    whitening_mean: Optional[torch.Tensor] = None
    whitening_W: Optional[torch.Tensor] = None
    sigma_inv: Optional[torch.Tensor] = None
    class_means: Optional[Dict[int, torch.Tensor]] = None
    maha_thresh_per_class: Optional[Dict[int, torch.Tensor]] = None
    maha_thresh_global: Optional[torch.Tensor] = None


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


def harts_cnn_original(X, y, random_state=None, max_iterations=100):
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
            pred = knn.predict(X[i:i + 1])[0]
            if pred != y[i]:
                S = np.append(S, i)
                knn.fit(X[S], y[S])
                changed = True
    return np.sort(np.unique(S))


def harts_cnn_symmetric(X, y, n_seeds_S=1, random_state=None, max_iterations=100):
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
                pred = knn.predict(X[s: s + 1])[0]
                if pred != y[s]:
                    C = np.append(C, s)
                    knn.fit(X[C], y[C])
                    changed = True
        selected.append(np.unique(np.append(seeds, np.intersect1d(C, S_cls))))
    if len(selected) == 0:
        return np.array([], dtype=int)
    return np.sort(np.unique(np.concatenate(selected)))


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
    idx = np.argpartition(-S, kth=min(k, S.shape[1]-1)-1, axis=1)[:, :k]
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
                                     descriptors_cache_path: Path = None, device='cuda', whiten_dim: int = 0,
                                     csls_k: int = 10):
    path_to_dataset = bop_base / dataset
    path_to_split = path_to_dataset / split

    all_images = []
    all_segmentations = []
    object_classes = []
    dino_cls_descriptors = []

    dino_descriptor = descriptor_from_hydra(descriptor_model, device)

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

            if cache_file_path and cache_file_path.exists():
                dino_cls_descriptor = torch.load(cache_file_path, map_location=device, weights_only=True)
            else:
                dino_cls_descriptor, dino_dense_descriptor = dino_descriptor.get_detections_from_files(image_path,
                                                                                                       seg_path)
                if cache_file_path:
                    torch.save(dino_cls_descriptor.detach().cpu(), cache_file_path)

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

    X_np = dino_cls_descriptors.cpu().numpy()
    X_np = _l2n(X_np).astype(np.float32)
    y_np = object_classes.numpy(force=True)

    mu_w, W_w = _fit_whitener(X_np, out_dim=min(whiten_dim, X_np.shape[1]))
    if whiten_dim and whiten_dim > 0:
        X_for_selection = _apply_whitener(X_np, mu_w, W_w)
        Xw_np = _apply_whitener(X_np, mu_w, W_w)
    else:
        mu_w, W_w = None, None
        Xw_np = X_np
        X_for_selection = X_np

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
        sample_indices = harts_cnn_symmetric(torch.from_numpy(X_for_selection), object_classes)
    elif method == 'hart':
        sample_indices = harts_cnn_original(torch.from_numpy(X_for_selection), object_classes)
    else:
        raise ValueError(f"Method {method} not recognized")

    result_save_path = cache_base_path / dataset / split
    shutil.rmtree(result_save_path, ignore_errors=True)

    saved_indices = []
    saved_labels = []

    for index in sample_indices:
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
        segmentation_path = all_segmentations[index]

        new_image_name = Path(f'{image_path.stem}_{index}{image_path.suffix}')
        new_seg_name = Path(f'{segmentation_path.stem}_{index}{segmentation_path.suffix}')
        descriptor_name = f'{new_image_name.stem}.pt'
        shutil.copy2(image_path, images_save_dir / new_image_name)
        shutil.copy2(segmentation_path, segmentation_save_dir / new_seg_name)

        cls_descriptor_to_save = dino_cls_descriptors[index].detach().cpu().clone()
        torch.save(cls_descriptor_to_save, descriptors_save_dir / descriptor_name)
        saved_indices.append(int(index))
        saved_labels.append(object_id)

    stats_dir = result_save_path
    stats_dir.mkdir(parents=True, exist_ok=True)
    if len(saved_indices) > 0:
        idx = np.array(saved_indices, dtype=int)
        y_sel = np.array(saved_labels, dtype=int)
        payload = {
            'whitening_mean': None if mu_w is None else torch.from_numpy(mu_w.squeeze(0)),
            'whitening_W': None if W_w is None else torch.from_numpy(W_w),
            'template_indices': torch.tensor(idx, dtype=torch.long),
            'template_labels': torch.tensor(y_sel, dtype=torch.long),
            'template_csls_avg': stats['template_csls_avg'],
            'sigma_inv': stats['sigma_inv'],
            'class_means': {int(k): v for k, v in stats['class_means'].items()},
        }

        torch.save(payload, stats_dir / 'csls_stats.pt')


def get_descriptors_for_condensed_templates(path_to_detections: Path, descriptor_name: str, device: str = 'cuda',
                                            threshold_quantile: float = 0.05, default_threshold: float = 0.0,
                                            mahalanobis_quantile: float = 0.95,
                                            force_recompute_descriptors: bool = True) -> TemplateBank:
    descriptor = descriptor_from_hydra(model=descriptor_name)

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
            rgb_img = Image.open(rgb_file).convert('RGB')
            rgb_tensor = transforms.ToTensor()(rgb_img).to(device)

            descriptor_file = descriptor_dir / f'{rgb_file.stem}.pt'

            images_dict[obj_id].append(rgb_tensor)

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

            cls_descriptors_dict[obj_id].append(x.squeeze(0))
            patch_descriptors_dict[obj_id].append(patch_descriptor.squeeze(0))

        images_dict[obj_id] = torch.stack(images_dict[obj_id])
        segmentations_dict[obj_id] = torch.stack(segmentations_dict[obj_id])
        cls_descriptors_dict[obj_id] = torch.stack(cls_descriptors_dict[obj_id])
        patch_descriptors_dict[obj_id] = torch.stack(patch_descriptors_dict[obj_id])

    template_thresholds: Dict[int, torch.Tensor] = {}
    for obj_id, X in cls_descriptors_dict.items():
        if X.shape[0] <= 1:
            template_thresholds[obj_id] = torch.full((X.shape[0],), float(default_threshold), device=X.device)
            continue
        S = X @ X.T
        diag = torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
        S = S - 1e9 * diag
        vals, _ = torch.sort(S, dim=1, descending=True)
        per_template_vals = vals[:, 0: max(1, min(vals.shape[1]-1, X.shape[0]-1))]
        q = torch.quantile(per_template_vals, q=threshold_quantile, dim=1, keepdim=False)
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
    parser.add_argument('--csls_k', type=int, default=10,
                        help='k for CSLS avg computation in stats')

    args = parser.parse_args()

    # Define paths
    experiment_name = f'1nn-{args.method}-{args.descriptor}'
    if args.whiten_dim > 0:
        experiment_name += f'-whitening_{args.whiten_dim}'
    cache_base_path = Path('/mnt/personal/jelint19/cache/detections_templates_cache') / experiment_name
    descriptors_cache_path = Path(f'/mnt/personal/jelint19/cache/{args.descriptor}_cache/bop')
    bop_base = Path('/mnt/personal/jelint19/data/bop')

    print(f"Processing {args.dataset}/{args.split} with method {args.method} and descriptor {args.descriptor}")

    # Perform condensation for single dataset/split
    perform_condensation_per_dataset(bop_base, cache_base_path, args.dataset, args.split, args.method, args.descriptor,
                                     descriptors_cache_path, args.device, whiten_dim=args.whiten_dim,
                                     csls_k=args.csls_k)


if __name__ == '__main__':
    main()

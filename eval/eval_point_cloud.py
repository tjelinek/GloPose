import numpy as np
from pathlib import Path

import trimesh
from scipy.spatial import KDTree


def sample_points_from_mesh(mesh_path: Path, num_points: int = 100_000) -> np.ndarray:
    """Load a mesh and uniformly sample points from its surface.

    Args:
        mesh_path: Path to a mesh file (PLY, OBJ, etc.).
        num_points: Number of points to sample.

    Returns:
        (N, 3) array of sampled surface points.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    points, _ = mesh.sample(num_points, return_index=True)
    return np.asarray(points, dtype=np.float64)


def extract_reconstruction_points(reconstruction) -> np.ndarray:
    """Extract 3D point coordinates from a pycolmap Reconstruction.

    Args:
        reconstruction: A pycolmap.Reconstruction object (already aligned).

    Returns:
        (M, 3) array of reconstructed 3D points.
    """
    return np.stack([p.xyz for p in reconstruction.points3D.values()])


def compute_reconstruction_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    fscore_thresholds_mm: tuple = (1.0, 2.0, 5.0),
    max_dist_mm: float = 20.0,
    gt_mesh_unit: str = 'mm',
    pred_points_unit: str = 'mm',
) -> dict:
    """Compute accuracy, completeness, overall, and F-score metrics.

    Both point clouds are converted to millimetres before evaluation.

    Args:
        pred_points: (M, 3) predicted (reconstructed) points.
        gt_points: (N, 3) ground-truth surface points.
        fscore_thresholds_mm: Thresholds in mm for F-score computation.
        max_dist_mm: Clamping distance in mm for accuracy/completeness.
        gt_mesh_unit: Unit of gt_points ('mm', 'cm', or 'm').
        pred_points_unit: Unit of pred_points ('mm', 'cm', or 'm').

    Returns:
        Dict with accuracy_mm, completeness_mm, overall_mm,
        fscore_{τ}mm for each threshold, and point counts.
    """
    unit_to_mm = {'mm': 1.0, 'cm': 10.0, 'm': 1000.0}
    pred = pred_points * unit_to_mm[pred_points_unit]
    gt = gt_points * unit_to_mm[gt_mesh_unit]

    tree_gt = KDTree(gt)
    tree_pred = KDTree(pred)

    # Accuracy: pred → GT
    dist_pred_to_gt, _ = tree_gt.query(pred)
    dist_pred_to_gt = np.minimum(dist_pred_to_gt, max_dist_mm)
    accuracy = float(np.mean(dist_pred_to_gt))

    # Completeness: GT → pred
    dist_gt_to_pred, _ = tree_pred.query(gt)
    dist_gt_to_pred = np.minimum(dist_gt_to_pred, max_dist_mm)
    completeness = float(np.mean(dist_gt_to_pred))

    overall = (accuracy + completeness) / 2.0

    result = {
        'accuracy_mm': accuracy,
        'completeness_mm': completeness,
        'overall_mm': overall,
        'num_pred_points': len(pred),
        'num_gt_points': len(gt),
    }

    for tau in fscore_thresholds_mm:
        precision = float(np.mean(dist_pred_to_gt <= tau))
        recall = float(np.mean(dist_gt_to_pred <= tau))
        if precision + recall > 0:
            fscore = 2.0 * precision * recall / (precision + recall)
        else:
            fscore = 0.0
        result[f'fscore_{tau:.0f}mm'] = fscore

    return result


def compute_pose_auc(
    errors: np.ndarray,
    thresholds: tuple = (5, 10, 30),
) -> dict:
    """Compute AUC of pose error at given thresholds.

    Adapted from VGGT ``calculate_auc_np``: builds a histogram of errors
    in integer-degree bins up to the maximum threshold, computes the
    cumulative sum, and returns the mean (= AUC) at each threshold.

    Args:
        errors: 1-D array of per-frame errors in degrees.
        thresholds: Degree thresholds at which to report AUC.

    Returns:
        Dict mapping ``auc_at_{t}`` to float AUC values.
    """
    max_threshold = max(thresholds)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(errors, bins=bins)
    num_frames = float(len(errors))
    if num_frames == 0:
        return {f'auc_at_{t}': 0.0 for t in thresholds}
    normalized = histogram.astype(float) / num_frames
    cumsum = np.cumsum(normalized)

    result = {}
    for t in thresholds:
        # AUC up to threshold t = mean of cumsum[0:t]
        result[f'auc_at_{t}'] = float(np.mean(cumsum[:t]))
    return result

import csv
import json
from pathlib import Path

import numpy as np

from data_structures.types import Detection, PoseEstimate


def detection_to_bop_record(det: Detection, scene_id: int, image_id: int, time: float) -> dict:
    """Convert a Detection to a BOP COCO detection record (JSON-ready dict).

    Args:
        det: Single detection with object_id, score, bbox_xywh.
        scene_id: BOP scene identifier (integer).
        image_id: BOP image identifier (integer).
        time: Inference time in seconds for this image.

    Returns:
        Dict matching BOP COCO detection format.

    Raises:
        TypeError: If object_id cannot be converted to int (non-BOP dataset).
    """
    return {
        'scene_id': scene_id,
        'image_id': image_id,
        'category_id': int(det.object_id),
        'bbox': det.bbox_xywh,
        'score': det.score,
        'time': time,
    }


def pose_to_bop_record(pose: PoseEstimate, scene_id: int, image_id: int, time: float) -> dict:
    """Convert a PoseEstimate to a BOP pose result record.

    Translation is converted from meters to millimeters (BOP convention).

    Args:
        pose: Single pose estimate with Se3_obj2cam.
        scene_id: BOP scene identifier (integer).
        image_id: BOP image identifier (integer).
        time: Inference time in seconds for this image.

    Returns:
        Dict with keys: scene_id, im_id, obj_id, score, R (ndarray 3x3), t (ndarray 3x1), time.
    """
    R = pose.R.detach().cpu().numpy().astype(np.float64)
    t_mm = (pose.t_meters * 1000.0).detach().cpu().numpy().astype(np.float64).reshape(3, 1)
    return {
        'scene_id': scene_id,
        'im_id': image_id,
        'obj_id': int(pose.object_id),
        'score': pose.score,
        'R': R,
        't': t_mm,
        'time': time,
    }


def write_bop_detection_json(records: list[dict], path: Path) -> None:
    """Write detection results as BOP COCO JSON.

    Args:
        records: List of dicts from detection_to_bop_record().
        path: Output JSON file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(records, f)


def write_bop_pose_csv(records: list[dict], path: Path) -> None:
    """Write pose results as BOP CSV.

    Format: scene_id,im_id,obj_id,score,R,t,time
    R is 9 space-separated floats (row-major), t is 3 space-separated floats (mm).

    Args:
        records: List of dicts from pose_to_bop_record().
        path: Output CSV file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time'])
        for rec in records:
            R_str = ' '.join(f'{v:.6f}' for v in rec['R'].flatten())
            t_str = ' '.join(f'{v:.6f}' for v in rec['t'].flatten())
            writer.writerow([
                rec['scene_id'],
                rec['im_id'],
                rec['obj_id'],
                f'{rec["score"]:.6f}',
                R_str,
                t_str,
                f'{rec["time"]:.3f}',
            ])
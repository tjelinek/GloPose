from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from kornia.geometry import So3, Quaternion
from pycolmap import Reconstruction


def update_global_statistics(csv_per_frame_stats: Path, csv_per_sequence_stats: Path, num_keyframes: int,
                             reconstruction: Reconstruction, dataset: str, sequence: str, reconstruction_success: bool,
                             pose_alignment_success: bool):
    # Read the input CSV file containing reconstruction data
    if not csv_per_frame_stats.exists():
        print(f"Error: Input file {csv_per_frame_stats} does not exist.")
        return

    df = pd.read_csv(csv_per_frame_stats)
    sequence_df = df[(df['dataset'] == dataset) & (df['sequence'] == sequence)]

    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    stats = {
        'dataset': dataset,
        'sequence': sequence,
        'num_keyframes': num_keyframes,
        'colmap_registered_keyframes': reconstruction.num_reg_images() if reconstruction is not None else None,
        'mean_rotation_error': None,
        'rot_accuracy_at_5_deg': None,
        'mean_translation_error': None,
        'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
        'note': str()
    }

    if not pose_alignment_success:
        stats['note'] = ('Pose alignment with the reference pose failed.'
                         'This happens when COLMAP does not register the 1st image.')
    elif not reconstruction_success:
        stats['note'] = 'SfM reconstruction failed.'
    else:
        for _, row in sequence_df.iterrows():
            # Skip if ground truth is not available
            if row['gt_rotation'] is None or row['gt_translation'] is None:
                continue

            gt_rot_val = eval(row['gt_rotation'])
            gt_trans_val = eval(row['gt_translation'])
            pred_rot_val = eval(row['pred_rotation'])
            pred_trans_val = eval(row['pred_translation'])

            gt_rot_matrix = torch.Tensor(gt_rot_val) if gt_rot_val is not None else None
            pred_rot_matrix = torch.Tensor(pred_rot_val) if pred_rot_val is not None else None
            gt_trans = torch.Tensor(gt_trans_val) if gt_trans_val is not None else None
            pred_trans = torch.Tensor(pred_trans_val) if pred_trans_val is not None else None

            if gt_rot_matrix is None or pred_rot_matrix is None or gt_trans is None or pred_trans is None:
                continue

            gt_So3 = So3(Quaternion.from_matrix(gt_rot_matrix))
            pred_So3 = So3(Quaternion.from_matrix(pred_rot_matrix))

            rel_rot: So3 = gt_So3.inverse() * pred_So3

            rotation_error_deg = torch.rad2deg(torch.linalg.norm(rel_rot.q.to_axis_angle())).item()
            rotation_errors.append(rotation_error_deg)

            translation_error = torch.linalg.norm(gt_trans - pred_trans).item()
            translation_errors.append(translation_error)

        rotation_errors_np = np.asarray(rotation_errors)
        translation_errors_np = np.asarray(translation_errors)

        # Update the existing stats dictionary instead of creating a new one
        stats.update({
            'mean_rotation_error': np.mean(rotation_errors_np),
            'rot_accuracy_at_5_deg': np.sum(rotation_errors_np <= 5) / len(rotation_errors_np),
            'mean_translation_error': np.mean(translation_errors_np)
        })

    stats_df = pd.DataFrame([stats])

    # Write to CSV
    if csv_per_sequence_stats.exists():
        existing_df = pd.read_csv(csv_per_sequence_stats)

        filtered_df = existing_df[~existing_df.set_index(['dataset', 'sequence']).index.isin(
            stats_df.set_index(['dataset', 'sequence']).index)]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        updated_df.to_csv(csv_per_sequence_stats, index=False)
    else:
        stats_df.to_csv(csv_per_sequence_stats, index=False)

    print(f"Statistics written to {csv_per_sequence_stats}")
    return stats_df

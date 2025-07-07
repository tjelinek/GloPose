from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from kornia.geometry import Se3, So3, Quaternion
from pycolmap import Reconstruction

from pose.glomap import get_image_Se3_world2cam


def evaluate_reconstruction(
        reconstruction,
        ground_truth_poses: Dict[int, Se3],
        image_name_to_frame_id: Dict[str, int],
        csv_output_path: Path,
        dataset: str,
        sequence: str
):
    stats = []

    for image in reconstruction.images.values():
        image_frame_id = image_name_to_frame_id[image.name]

        Se3_obj2cam_pred = get_image_Se3_world2cam(image, 'cpu')

        # Get ground truth pose for this frame
        Se3_world2cam_gt = ground_truth_poses.get(image_frame_id)

        # Ground-truth rotation and translation
        gt_rotation = Se3_world2cam_gt.rotation.matrix().tolist() if Se3_world2cam_gt is not None else None
        gt_translation = Se3_world2cam_gt.translation.tolist() if Se3_world2cam_gt is not None else None

        pred_rotation = Se3_obj2cam_pred.rotation.matrix().tolist()
        pred_translation = Se3_obj2cam_pred.translation.tolist()

        # Add stats for the current image frame
        stats.append({
            'dataset': dataset,
            'sequence': sequence,
            'image_frame_id': image_frame_id,
            'gt_rotation': gt_rotation,
            'gt_translation': gt_translation,
            'pred_rotation': pred_rotation,
            'pred_translation': pred_translation
        })

    # Convert stats to a Pandas DataFrame
    stats_df = pd.DataFrame(stats)

    # If the CSV file exists, append; otherwise, create a new one
    if csv_output_path.exists():
        existing_df = pd.read_csv(csv_output_path)
        filtered_df = existing_df[~((existing_df['dataset'] == dataset) &
                                    (existing_df['sequence'] == sequence))]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        updated_df.to_csv(csv_output_path, index=False)
    else:
        stats_df.to_csv(csv_output_path, index=False)


def update_dataset_reconstructions_stats(
        csv_per_frame_stats: Path,
        csv_per_sequence_stats: Path,
        num_keyframes: int,
        input_frames: int,
        reconstruction: Reconstruction,
        dataset: str,
        sequence: str,
        reconstruction_success: bool,
        pose_alignment_success: bool
):
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
        'input_frames': input_frames,
        'colmap_registered_keyframes': reconstruction.num_reg_images() if reconstruction is not None else None,
        'mean_rotation_error': None,
        'mean_translation_error': None,
        'rot_accuracy_at_2_deg': None,
        'rot_accuracy_at_5_deg': None,
        'rot_accuracy_at_10_deg': None,
        'trans_accuracy_at_0_05': None,
        'trans_accuracy_at_0_10': None,
        'trans_accuracy_at_0_50': None,
        'reconstruction_success': reconstruction_success,
        'alignment_success': pose_alignment_success,
        'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S"),
        'note': str()
    }

    if not pose_alignment_success:
        stats['note'] = ('Pose alignment with the reference pose failed. '
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

        if rotation_errors and translation_errors:
            rotation_errors_np = np.asarray(rotation_errors)
            translation_errors_np = np.asarray(translation_errors)

            # Update the existing stats dictionary
            stats.update({
                'mean_rotation_error': np.mean(rotation_errors_np),
                'mean_translation_error': np.mean(translation_errors_np),
                'rot_accuracy_at_2_deg': np.sum(rotation_errors_np <= 2) / len(rotation_errors_np),
                'rot_accuracy_at_5_deg': np.sum(rotation_errors_np <= 5) / len(rotation_errors_np),
                'rot_accuracy_at_10_deg': np.sum(rotation_errors_np <= 10) / len(rotation_errors_np),
                'trans_accuracy_at_0_05': np.sum(translation_errors_np <= 0.05) / len(translation_errors_np),
                'trans_accuracy_at_0_10': np.sum(translation_errors_np <= 0.10) / len(translation_errors_np),
                'trans_accuracy_at_0_50': np.sum(translation_errors_np <= 0.50) / len(translation_errors_np),
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

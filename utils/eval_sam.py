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


def update_iou_frame_statistics(csv_iou_frame_stats: Path, iou: np.ndarray,
                                dataset: str, sequence: str, start_frame: int = 0):
    """
    Updates the per-frame IoU statistics CSV file.

    Args:
        csv_iou_frame_stats: Path to the frame-level IoU CSV file
        iou: 1D numpy array containing IoU values for each frame
        dataset: Dataset name
        sequence: Sequence name
        start_frame: Starting frame number (default: 0)
    """
    # Create frame-level statistics
    frame_stats = []
    for i, iou_val in enumerate(iou):
        frame_stats.append({
            'dataset': dataset,
            'sequence': sequence,
            'frame': start_frame + i,
            'iou': float(iou_val)
        })

    frame_df = pd.DataFrame(frame_stats)

    # Write to CSV
    if csv_iou_frame_stats.exists():
        existing_df = pd.read_csv(csv_iou_frame_stats)

        # Remove existing entries for this dataset-sequence combination
        filtered_df = existing_df[~((existing_df['dataset'] == dataset) &
                                    (existing_df['sequence'] == sequence))]
        updated_df = pd.concat([filtered_df, frame_df], ignore_index=True)
        updated_df.to_csv(csv_iou_frame_stats, index=False)
    else:
        frame_df.to_csv(csv_iou_frame_stats, index=False)

    print(f"Frame statistics written to {csv_iou_frame_stats}")
    return frame_df


def _compute_iou_statistics_from_values(iou_values: np.ndarray, failure_threshold: float = 0.5):
    """
    Helper function to compute IoU statistics from raw IoU values.

    Args:
        iou_values: Array of IoU values
        failure_threshold: Threshold below which IoU is considered a failure

    Returns:
        Dictionary containing computed statistics
    """
    # Basic statistics
    avg_iou = np.mean(iou_values)

    # Percentage of frames with IoU > thresholds
    thresholds = [0.8, 0.9, 0.95, 0.99, 0.999]
    threshold_stats = {}
    for thresh in thresholds:
        key = f'pct_frames_iou_gt_{str(thresh)}'
        threshold_stats[key] = np.sum(iou_values > thresh) / len(iou_values) * 100

    # Failure analysis
    is_failure = iou_values <= failure_threshold

    # Frames to first failure
    if np.any(is_failure):
        frames_to_first_failure = np.argmax(is_failure)
    else:
        frames_to_first_failure = 'No failure'  # No failure occurred

    # Failure duration analysis
    failure_durations = []
    in_failure = False
    failure_start = 0

    for i, failed in enumerate(is_failure):
        if failed and not in_failure:
            # Start of a failure period
            in_failure = True
            failure_start = i
        elif not failed and in_failure:
            # End of a failure period
            in_failure = False
            failure_durations.append(i - failure_start)

    # Handle case where sequence ends in failure
    if in_failure:
        failure_durations.append(len(is_failure) - failure_start)

    avg_failure_duration = np.mean(failure_durations) if failure_durations else 0.0

    # Recovery analysis (recovery = failure followed by success)
    num_recoveries = 0
    for i in range(1, len(is_failure)):
        if is_failure[i - 1] and not is_failure[i]:
            num_recoveries += 1

    stats = {
        'num_frames': len(iou_values),
        'avg_iou': avg_iou,
        'frames_to_first_failure': frames_to_first_failure,
        'avg_failure_duration': avg_failure_duration,
        'num_recoveries': num_recoveries,
        **threshold_stats
    }

    return stats


def compute_sequence_iou_statistics(csv_iou_frame_stats: Path, csv_iou_sequence_stats: Path):
    """
    Computes sequence-level statistics from frame-level IoU data.

    Args:
        csv_iou_frame_stats: Path to the folder containing frame-level IoU CSV files
        csv_iou_sequence_stats: Path to the sequence-level statistics CSV file
    """
    if not csv_iou_frame_stats.exists():
        print(f"Error: Input folder {csv_iou_frame_stats} does not exist.")
        return

    if not csv_iou_frame_stats.is_dir():
        print(f"Error: {csv_iou_frame_stats} is not a directory.")
        return

    # Find all CSV files in the folder
    csv_files = list(csv_iou_frame_stats.glob("*.csv"))

    if not csv_files:
        print(f"Error: No CSV files found in {csv_iou_frame_stats}")
        return

    sequence_stats = []

    # Process each CSV file
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        for (dataset, sequence), group in df.groupby(['dataset', 'sequence']):
            iou_values = group['iou'].values
            stats = _compute_iou_statistics_from_values(iou_values)

            # Add metadata
            stats.update({
                'dataset': dataset,
                'sequence': sequence,
                'file_source': csv_file.name,
                'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
            })

            sequence_stats.append(stats)

    stats_df = pd.DataFrame(sequence_stats)

    # Create output directory if it doesn't exist
    csv_iou_sequence_stats.parent.mkdir(parents=True, exist_ok=True)

    stats_df.to_csv(csv_iou_sequence_stats, index=False)

    print(f"Sequence statistics written to {csv_iou_sequence_stats}")
    print(f"Processed {len(csv_files)} CSV files, computed statistics for {len(sequence_stats)} sequences")

    return stats_df


def compute_dataset_iou_statistics(csv_iou_sequence_stats: Path, csv_iou_dataset_stats: Path):
    """
    Computes dataset-level statistics from sequence-level statistics.

    Args:
        csv_iou_sequence_stats: Path to the sequence-level statistics CSV file
        csv_iou_dataset_stats: Path to the dataset-level statistics CSV file
    """
    if not csv_iou_sequence_stats.exists():
        print(f"Error: Input file {csv_iou_sequence_stats} does not exist.")
        return

    df = pd.read_csv(csv_iou_sequence_stats)
    dataset_stats = []

    # Group by dataset
    for dataset, group in df.groupby('dataset'):
        # Weight averages by number of frames per sequence
        weights = group['num_frames'].values

        # Metrics to aggregate with weighted average
        weighted_metrics = [
            'avg_iou', 'pct_frames_iou_gt_0.8', 'pct_frames_iou_gt_0.9',
            'pct_frames_iou_gt_0.95', 'pct_frames_iou_gt_0.99', 'pct_frames_iou_gt_0.999'
        ]

        stats = {
            'dataset': dataset,
            'num_sequences': len(group),
            'total_frames': group['num_frames'].sum(),
            'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
        }

        # Add weighted averages
        for metric in weighted_metrics:
            prefix = 'avg_' if not metric.startswith('avg_') else ''
            stats[f'{prefix}{metric}'] = np.average(group[metric].values, weights=weights)

        # Handle failure statistics
        failure_mask = group['frames_to_first_failure'] != 'No failure'
        sequences_with_failure = group[failure_mask]

        stats['num_sequences_with_failure'] = len(sequences_with_failure)
        stats['pct_sequences_with_failure'] = (len(sequences_with_failure) / len(group)) * 100

        stats['avg_failures_per_seq'] = np.mean(group['num_recoveries'].values),

        # Average failure duration only for sequences that had failures
        if len(sequences_with_failure) > 0:
            stats['avg_failure_duration_when_occurred'] = np.mean(sequences_with_failure['avg_failure_duration'].values)
        else:
            stats['avg_failure_duration_when_occurred'] = 0.0

        dataset_stats.append(stats)

    stats_df = pd.DataFrame(dataset_stats)
    stats_df.to_csv(csv_iou_dataset_stats, index=False)

    print(f"Dataset statistics written to {csv_iou_dataset_stats}")
    return stats_df


if __name__ == '__main__':

    data_root = Path('/mnt/personal/jelint19/results/FlowTracker/sam2_eval')
    csv_per_frame_results_folder = data_root / 'sam_stats'
    csv_per_sequence_iou_stats = data_root / 'stats_iou_per_sequence.csv'
    csv_per_dataset_iou_stats = data_root / 'stats_iou_per_dataset.csv'

    compute_sequence_iou_statistics(csv_per_frame_results_folder, csv_per_sequence_iou_stats)
    compute_dataset_iou_statistics(csv_per_sequence_iou_stats, csv_per_dataset_iou_stats)
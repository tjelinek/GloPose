from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from kornia.geometry import Se3, So3, Quaternion
from pycolmap import Reconstruction

from pose.glomap import get_image_Se3_world2cam


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """
    Round all numeric columns in a DataFrame to specified decimal places.

    Args:
        df: Input DataFrame
        decimals: Number of decimal places to round to

    Returns:
        DataFrame with rounded numeric columns
    """
    df_copy = df.copy()
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns
    df_copy[numeric_columns] = df_copy[numeric_columns].round(decimals)
    return df_copy


def get_scale_factor(from_unit: str, to_unit: str) -> float:
    """
    Get the scale factor to convert from one unit to another.

    Args:
        from_unit: Source unit ('m', 'cm', 'mm')
        to_unit: Target unit ('m', 'cm', 'mm')

    Returns:
        Scale factor to multiply values by
    """
    # Convert to meters first, then to target unit
    to_meters = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}
    from_meters = {'m': 1.0, 'cm': 100.0, 'mm': 1000.0}

    if from_unit not in to_meters or to_unit not in from_meters:
        raise ValueError(f"Unsupported units. Use 'm', 'cm', or 'mm'. Got {from_unit} -> {to_unit}")

    return to_meters[from_unit] * from_meters[to_unit]


def get_translation_thresholds_in_output_unit(output_unit: str) -> Dict[str, float]:
    """
    Get translation accuracy thresholds converted to the specified output unit.
    Thresholds are always 0.1, 0.5, 1, 5, 10 cm, but converted to output unit for comparison.

    Args:
        output_unit: Output unit ('m', 'cm', 'mm')

    Returns:
        Dictionary mapping threshold names to values in output unit
    """
    # Fixed thresholds in centimeters
    base_thresholds_cm = {
        '0_1': 0.1,
        '0_5': 0.5,
        '1': 1.0,
        '5': 5.0,
        '10': 10.0
    }

    # Convert to target unit for comparison
    scale_factor = get_scale_factor('cm', output_unit)
    return {k: v * scale_factor for k, v in base_thresholds_cm.items()}


def evaluate_reconstruction(
        reconstruction,
        ground_truth_poses: Dict[int, Se3],
        image_name_to_frame_id: Dict[str, int],
        csv_output_path: Path,
        dataset: str,
        sequence: str,
        input_translation_unit: str = 'mm',
        output_translation_unit: str = 'cm'
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
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(csv_output_path, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(csv_output_path, index=False)


def update_sequence_reconstructions_stats(
        csv_per_frame_stats: Path,
        csv_per_sequence_stats: Path,
        num_keyframes: int,
        input_frames: int,
        reconstruction: Reconstruction,
        dataset: str,
        sequence: str,
        reconstruction_success: bool,
        pose_alignment_success: bool,
        input_translation_unit: str = 'mm',
        output_translation_unit: str = 'cm'
):
    # Read the input CSV file containing reconstruction data
    if not csv_per_frame_stats.exists():
        print(f"Error: Input file {csv_per_frame_stats} does not exist.")
        return

    df = pd.read_csv(csv_per_frame_stats)
    sequence_df = df[(df['dataset'] == dataset) & (df['sequence'] == sequence)]

    rotation_errors: List[float] = []
    translation_errors: List[float] = []

    # Get scale factor for translation unit conversion
    scale_factor = get_scale_factor(input_translation_unit, output_translation_unit)

    # Get translation thresholds converted to output unit for comparison
    translation_thresholds = get_translation_thresholds_in_output_unit(output_translation_unit)

    stats = {
        'dataset': dataset,
        'sequence': sequence,
        'num_keyframes': num_keyframes,
        'input_frames': input_frames,
        'colmap_registered_keyframes': reconstruction.num_reg_images() if reconstruction is not None else None,
        'mean_rotation_error': None,
        f'mean_translation_error_{output_translation_unit}': None,
        'rot_accuracy_at_2_deg': None,
        'rot_accuracy_at_5_deg': None,
        'rot_accuracy_at_10_deg': None,
        'trans_accuracy_at_0_1_cm': None,
        'trans_accuracy_at_0_5_cm': None,
        'trans_accuracy_at_1_cm': None,
        'trans_accuracy_at_5_cm': None,
        'trans_accuracy_at_10_cm': None,
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

            # Convert translation error to output unit
            translation_error = torch.linalg.norm(gt_trans - pred_trans).item() * scale_factor
            translation_errors.append(translation_error)

        if rotation_errors and translation_errors:
            rotation_errors_np = np.asarray(rotation_errors)
            translation_errors_np = np.asarray(translation_errors)

            # Update the existing stats dictionary
            stats.update({
                'mean_rotation_error': np.mean(rotation_errors_np),
                f'mean_translation_error_{output_translation_unit}': np.mean(translation_errors_np),
                'rot_accuracy_at_2_deg': np.sum(rotation_errors_np <= 2) / len(rotation_errors_np),
                'rot_accuracy_at_5_deg': np.sum(rotation_errors_np <= 5) / len(rotation_errors_np),
                'rot_accuracy_at_10_deg': np.sum(rotation_errors_np <= 10) / len(rotation_errors_np),
                'trans_accuracy_at_0_1_cm': np.sum(translation_errors_np <= translation_thresholds['0_1']) / len(
                    translation_errors_np),
                'trans_accuracy_at_0_5_cm': np.sum(translation_errors_np <= translation_thresholds['0_5']) / len(
                    translation_errors_np),
                'trans_accuracy_at_1_cm': np.sum(translation_errors_np <= translation_thresholds['1']) / len(
                    translation_errors_np),
                'trans_accuracy_at_5_cm': np.sum(translation_errors_np <= translation_thresholds['5']) / len(
                    translation_errors_np),
                'trans_accuracy_at_10_cm': np.sum(translation_errors_np <= translation_thresholds['10']) / len(
                    translation_errors_np),
            })

    stats_df = pd.DataFrame([stats])

    # Write to CSV
    if csv_per_sequence_stats.exists():
        existing_df = pd.read_csv(csv_per_sequence_stats)

        filtered_df = existing_df[~existing_df.set_index(['dataset', 'sequence']).index.isin(
            stats_df.set_index(['dataset', 'sequence']).index)]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(csv_per_sequence_stats, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(csv_per_sequence_stats, index=False)

    print(f"Statistics written to {csv_per_sequence_stats}")
    return stats_df


def update_experiment_statistics(
        experiment_name: str,
        dataset_stats_files: Dict[str, Path],
        experiment_stats_file: Path,
        output_translation_unit: str = 'cm'
):
    """
    Aggregate dataset statistics into experiment-level statistics.

    Args:
        experiment_name: Name of the experiment
        dataset_stats_files: Dict mapping dataset names to their stats CSV files
        experiment_stats_file: Path to output experiment statistics CSV
        output_translation_unit: Unit for translation measurements
    """
    experiment_stats = {'experiment': experiment_name}

    for dataset_name, stats_file in dataset_stats_files.items():
        if not stats_file.exists():
            print(f"Warning: Dataset stats file {stats_file} does not exist. Skipping {dataset_name}.")
            continue

        df = pd.read_csv(stats_file)

        # Calculate aggregated statistics for this dataset
        dataset_prefix = f"{dataset_name}_"

        # Basic counts
        experiment_stats[f"{dataset_prefix}num_sequences"] = len(df)
        experiment_stats[f"{dataset_prefix}total_keyframes"] = df['num_keyframes'].sum()
        experiment_stats[f"{dataset_prefix}total_input_frames"] = df['input_frames'].sum()
        experiment_stats[f"{dataset_prefix}total_registered_keyframes"] = df['colmap_registered_keyframes'].sum()

        # Success rates
        experiment_stats[f"{dataset_prefix}reconstruction_success_rate"] = df['reconstruction_success'].sum() / len(df)
        experiment_stats[f"{dataset_prefix}alignment_success_rate"] = df['alignment_success'].sum() / len(df)

        # Only calculate accuracy metrics for successful reconstructions
        successful_df = df[df['reconstruction_success'] & df['alignment_success']]

        if len(successful_df) > 0:
            # Mean errors
            experiment_stats[f"{dataset_prefix}mean_rotation_error"] = successful_df['mean_rotation_error'].mean()
            experiment_stats[f"{dataset_prefix}mean_translation_error_{output_translation_unit}"] = successful_df[
                f'mean_translation_error_{output_translation_unit}'].mean()

            # Rotation accuracy at different thresholds
            experiment_stats[f"{dataset_prefix}rot_accuracy_at_2_deg"] = successful_df['rot_accuracy_at_2_deg'].mean()
            experiment_stats[f"{dataset_prefix}rot_accuracy_at_5_deg"] = successful_df['rot_accuracy_at_5_deg'].mean()
            experiment_stats[f"{dataset_prefix}rot_accuracy_at_10_deg"] = successful_df['rot_accuracy_at_10_deg'].mean()

            # Translation accuracy at different thresholds
            experiment_stats[f"{dataset_prefix}trans_accuracy_at_0_1_cm"] = successful_df[
                'trans_accuracy_at_0_1_cm'].mean()
            experiment_stats[f"{dataset_prefix}trans_accuracy_at_0_5_cm"] = successful_df[
                'trans_accuracy_at_0_5_cm'].mean()
            experiment_stats[f"{dataset_prefix}trans_accuracy_at_1_cm"] = successful_df['trans_accuracy_at_1_cm'].mean()
            experiment_stats[f"{dataset_prefix}trans_accuracy_at_5_cm"] = successful_df['trans_accuracy_at_5_cm'].mean()
            experiment_stats[f"{dataset_prefix}trans_accuracy_at_10_cm"] = successful_df[
                'trans_accuracy_at_10_cm'].mean()
        else:
            # No successful reconstructions for this dataset
            for metric in ['mean_rotation_error', f'mean_translation_error_{output_translation_unit}',
                           'rot_accuracy_at_2_deg', 'rot_accuracy_at_5_deg', 'rot_accuracy_at_10_deg',
                           'trans_accuracy_at_0_1_cm', 'trans_accuracy_at_0_5_cm',
                           'trans_accuracy_at_1_cm', 'trans_accuracy_at_5_cm',
                           'trans_accuracy_at_10_cm']:
                experiment_stats[f"{dataset_prefix}{metric}"] = None

    # Add timestamp
    experiment_stats['timestamp'] = datetime.now().strftime("%d.%m.%Y, %H:%M:%S")

    # Convert to DataFrame
    stats_df = pd.DataFrame([experiment_stats])

    # Write to CSV
    if experiment_stats_file.exists():
        existing_df = pd.read_csv(experiment_stats_file)
        # Remove existing entry for this experiment
        filtered_df = existing_df[existing_df['experiment'] != experiment_name]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(experiment_stats_file, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(experiment_stats_file, index=False)

    print(f"Experiment statistics written to {experiment_stats_file}")
    return stats_df


def generate_dataset_reconstruction_statistics(
        keyframe_stats_file: Path,
        per_sequence_stats_file: Path,
        dataset_name: str,
        output_translation_unit: str = 'cm'
):
    """
    Generate dataset-level statistics from sequence-level statistics.

    Args:
        keyframe_stats_file: Path to sequence statistics CSV
        per_sequence_stats_file: Path to output dataset statistics CSV
        dataset_name: Name of the dataset
        output_translation_unit: Unit for translation measurements
    """
    if not keyframe_stats_file.exists():
        print(f"Error: Sequence stats file {keyframe_stats_file} does not exist.")
        return

    df = pd.read_csv(keyframe_stats_file)
    dataset_df = df[df['dataset'] == dataset_name]

    if len(dataset_df) == 0:
        print(f"No data found for dataset {dataset_name}")
        return

    # Calculate dataset-level statistics
    dataset_stats = {
        'dataset': dataset_name,
        'num_sequences': len(dataset_df),
        'total_keyframes': dataset_df['num_keyframes'].sum(),
        'total_input_frames': dataset_df['input_frames'].sum(),
        'total_registered_keyframes': dataset_df['colmap_registered_keyframes'].sum(),
        'reconstruction_success_rate': dataset_df['reconstruction_success'].sum() / len(dataset_df),
        'alignment_success_rate': dataset_df['alignment_success'].sum() / len(dataset_df),
        'timestamp': datetime.now().strftime("%d.%m.%Y, %H:%M:%S")
    }

    # Only calculate accuracy metrics for successful reconstructions
    successful_df = dataset_df[dataset_df['reconstruction_success'] & dataset_df['alignment_success']]

    if len(successful_df) > 0:
        dataset_stats.update({
            'mean_rotation_error': successful_df['mean_rotation_error'].mean(),
            f'mean_translation_error_{output_translation_unit}': successful_df[
                f'mean_translation_error_{output_translation_unit}'].mean(),
            'rot_accuracy_at_2_deg': successful_df['rot_accuracy_at_2_deg'].mean(),
            'rot_accuracy_at_5_deg': successful_df['rot_accuracy_at_5_deg'].mean(),
            'rot_accuracy_at_10_deg': successful_df['rot_accuracy_at_10_deg'].mean(),
            'trans_accuracy_at_0_1_cm': successful_df['trans_accuracy_at_0_1_cm'].mean(),
            'trans_accuracy_at_0_5_cm': successful_df['trans_accuracy_at_0_5_cm'].mean(),
            'trans_accuracy_at_1_cm': successful_df['trans_accuracy_at_1_cm'].mean(),
            'trans_accuracy_at_5_cm': successful_df['trans_accuracy_at_5_cm'].mean(),
            'trans_accuracy_at_10_cm': successful_df['trans_accuracy_at_10_cm'].mean(),
        })
    else:
        # No successful reconstructions
        for metric in ['mean_rotation_error', f'mean_translation_error_{output_translation_unit}',
                       'rot_accuracy_at_2_deg', 'rot_accuracy_at_5_deg', 'rot_accuracy_at_10_deg',
                       'trans_accuracy_at_0_1_cm', 'trans_accuracy_at_0_5_cm',
                       'trans_accuracy_at_1_cm', 'trans_accuracy_at_5_cm',
                       'trans_accuracy_at_10_cm']:
            dataset_stats[metric] = None

    # Convert to DataFrame and save
    stats_df = pd.DataFrame([dataset_stats])

    # Write to CSV
    if per_sequence_stats_file.exists():
        existing_df = pd.read_csv(per_sequence_stats_file)
        # Remove existing entry for this dataset
        filtered_df = existing_df[existing_df['dataset'] != dataset_name]
        updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
        # Round numeric columns before saving
        updated_df = round_numeric_columns(updated_df)
        updated_df.to_csv(per_sequence_stats_file, index=False)
    else:
        # Round numeric columns before saving
        stats_df = round_numeric_columns(stats_df)
        stats_df.to_csv(per_sequence_stats_file, index=False)

    print(f"Dataset statistics written to {per_sequence_stats_file}")
    return stats_df


if __name__ == '__main__':

    base_experiments_folder = Path('/mnt/personal/jelint19/results/FlowTracker/matchability_thresholds')
    for experiment_folder in sorted(base_experiments_folder.iterdir()):

        if not experiment_folder.is_dir():
            continue

        kf_stats_file = experiment_folder / 'reconstruction_keyframe_stats.csv'
        seq_stats_file = experiment_folder / 'reconstruction_sequence_stats.csv'

        dataset_stats_files = {}
        for dataset_folder in experiment_folder.iterdir():
            if not dataset_folder.is_dir():
                continue

            dataset_name = dataset_folder.stem
            stat_file = generate_dataset_reconstruction_statistics(kf_stats_file, seq_stats_file, dataset_folder.name)
            if stat_file is not None:
                dataset_stats_files[dataset_name] = stat_file

        experiment_name = experiment_folder.stem
        experiment_stats_file_path = base_experiments_folder / 'experiment_stats.csv'
        update_experiment_statistics(experiment_name, dataset_stats_files, experiment_stats_file_path)
        pass

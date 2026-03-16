"""Collect and display reconstruction results across experiments.

Usage:
    python scripts/collect_experiment_results.py                             # all experiments
    python scripts/collect_experiment_results.py onboarding/ufm_c0975r05     # specific ones
    python scripts/collect_experiment_results.py --per-sequence              # sequence-level detail
    python scripts/collect_experiment_results.py --csv results.csv           # export to CSV
    python scripts/collect_experiment_results.py --dataset handal_static_onboarding  # filter by dataset
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.job_runner import get_configurations, get_results_root


DATASET_COLUMNS = [
    'dataset', 'num_sequences', 'mean_input_frames', 'mean_keyframes',
    'mean_colmap_registered_keyframes',
    'reconstruction_success_rate', 'alignment_success_rate',
    'mean_frame_filtering_time', 'mean_reconstruction_time',
    'mean_rotation_error', 'mean_translation_error_cm',
    'rot_accuracy_at_2_deg', 'rot_accuracy_at_5_deg', 'rot_accuracy_at_10_deg',
    'trans_accuracy_at_1_cm', 'trans_accuracy_at_5_cm', 'trans_accuracy_at_10_cm',
    'pose_auc_at_5', 'pose_auc_at_10', 'pose_auc_at_30',
    'accuracy_mm', 'completeness_mm', 'overall_mm',
    'fscore_1mm', 'fscore_2mm', 'fscore_5mm',
]

SEQUENCE_COLUMNS = [
    'dataset', 'sequence', 'num_keyframes', 'input_frames',
    'colmap_registered_keyframes',
    'mean_rotation_error', 'mean_translation_error_cm',
    'rot_accuracy_at_5_deg', 'trans_accuracy_at_5_cm',
    'pose_auc_at_5', 'pose_auc_at_10', 'pose_auc_at_30',
    'accuracy_mm', 'completeness_mm', 'fscore_5mm',
    'reconstruction_success', 'alignment_success',
    'frame_filtering_time', 'reconstruction_time',
]

# Summary columns for the default (compact) view
SUMMARY_COLUMNS = [
    'mean_rotation_error', 'mean_translation_error_cm',
    'rot_accuracy_at_5_deg', 'trans_accuracy_at_5_cm',
    'reconstruction_success_rate', 'alignment_success_rate',
    'pose_auc_at_5', 'pose_auc_at_30',
    'accuracy_mm', 'completeness_mm', 'fscore_5mm',
]


def load_dataset_stats(config_name: str, results_root: Path) -> pd.DataFrame | None:
    csv_path = results_root / config_name / 'reconstruction_dataset_stats.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.insert(0, 'experiment', config_name)
    return df


def load_sequence_stats(config_name: str, results_root: Path) -> pd.DataFrame | None:
    csv_path = results_root / config_name / 'reconstruction_sequence_stats.csv'
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df.insert(0, 'experiment', config_name)
    return df


def main():
    parser = argparse.ArgumentParser(description='Collect experiment results')
    parser.add_argument('configs', nargs='*', help='Config names (default: all from job_runner)')
    parser.add_argument('--per-sequence', action='store_true', help='Show per-sequence results')
    parser.add_argument('--csv', type=str, help='Export results to CSV file')
    parser.add_argument('--dataset', type=str, help='Filter by dataset name (substring match)')
    parser.add_argument('--full', action='store_true', help='Show all columns')
    args = parser.parse_args()

    configurations = args.configs if args.configs else get_configurations()
    results_root = get_results_root()

    all_dfs = []
    missing = []

    for config_name in configurations:
        if args.per_sequence:
            df = load_sequence_stats(config_name, results_root)
        else:
            df = load_dataset_stats(config_name, results_root)

        if df is not None:
            all_dfs.append(df)
        else:
            missing.append(config_name)

    if not all_dfs:
        print('No results found for any experiment.')
        if missing:
            print(f'Missing: {", ".join(missing)}')
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    if args.dataset:
        combined = combined[combined['dataset'].str.contains(args.dataset, case=False, na=False)]

    if not args.full:
        if args.per_sequence:
            available = ['experiment'] + [c for c in SEQUENCE_COLUMNS if c in combined.columns]
        else:
            available = ['experiment'] + [c for c in SUMMARY_COLUMNS if c in combined.columns]
            if 'dataset' in combined.columns:
                available.insert(1, 'dataset')
        combined = combined[[c for c in available if c in combined.columns]]

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.max_colwidth', 40)
    pd.set_option('display.float_format', '{:.3f}'.format)

    if args.csv:
        combined.to_csv(args.csv, index=False)
        print(f'Results written to {args.csv}')
    else:
        print(combined.to_string(index=False))

    if missing:
        print(f'\nNo results for: {", ".join(missing)}')


if __name__ == '__main__':
    main()

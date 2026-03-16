"""Check which experiments have completed, failed, or are missing results.

Usage:
    python scripts/check_experiment_status.py                    # check all experiments from job_runner
    python scripts/check_experiment_status.py onboarding/ufm_c0975r05 reconstruction/vggt  # check specific ones
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.job_runner import get_configurations, get_results_root, get_sequences, Datasets


# Maps Datasets enum → (csv_dataset_name, sequence_suffix_to_strip)
# The CSV dataset name is constructed as {dataset}_{onboarding_type}_onboarding for BOP,
# or the raw dataset name for others.
DATASET_CSV_NAMES = {
    Datasets.BOP_HANDAL_ONBOARDING_STATIC: 'handal_static_onboarding',
    Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: 'handal_dynamic_onboarding',
    Datasets.HOPE_ONBOARDING_STATIC: 'hope_static_onboarding',
    Datasets.HOPE_ONBOARDING_DYNAMIC: 'hope_dynamic_onboarding',
    Datasets.HOT3D_ONBOARDING_STATIC: 'hot3d_static_onboarding',
    Datasets.HOT3D_ONBOARDING_DYNAMIC: 'hot3d_dynamic_onboarding',
    Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: 'bop_classic',
    Datasets.HO3D_train: 'HO3D',
    Datasets.NAVI: 'navi',
}


def get_expected_csv_sequences(dataset_enum: Datasets, sequences: list[str]) -> list[tuple[str, str]]:
    """Return list of (csv_dataset_name, csv_sequence_name) expected in the CSV."""
    csv_dataset = DATASET_CSV_NAMES.get(dataset_enum)
    if csv_dataset is None:
        return []

    results = []
    for seq in sequences:
        parts = seq.split('_')

        if dataset_enum in (Datasets.BOP_HANDAL_ONBOARDING_STATIC, Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC,
                            Datasets.HOPE_ONBOARDING_STATIC, Datasets.HOPE_ONBOARDING_DYNAMIC):
            # e.g., obj_000001_both → csv sequence = obj_000001_both_down (or _both, _up, _dynamic)
            # The sequence name in CSV includes special_hash suffix
            if len(parts) == 3:
                base_seq = f'{parts[0]}_{parts[1]}'
                suffix = parts[2]  # both, up, down, dynamic
                if suffix == 'dynamic':
                    csv_seq = f'{base_seq}_dynamic'
                else:
                    csv_seq = f'{base_seq}_{suffix}'
                results.append((csv_dataset, csv_seq))
            else:
                results.append((csv_dataset, seq))

        elif dataset_enum in (Datasets.HOT3D_ONBOARDING_STATIC, Datasets.HOT3D_ONBOARDING_DYNAMIC):
            # HOT3D: NNNNNN_static or NNNNNN_dynamic
            if len(parts) == 2:
                base_seq = parts[0]
                suffix = parts[1]  # static, dynamic
                csv_seq = f'{base_seq}_{suffix}'
                results.append((csv_dataset, csv_seq))
            else:
                results.append((csv_dataset, seq))

        elif dataset_enum == Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES:
            # Classic: dataset_NNNNNN format, dataset name varies (tless, lmo, icbin)
            # These are handled differently — the CSV dataset name includes the BOP dataset
            # For now, use the sequence as-is since classic sequences include dataset prefix
            results.append((csv_dataset, seq))

        else:
            # HO3D, NAVI: sequence name used as-is
            results.append((csv_dataset, seq))

    return results


def check_experiment(config_name: str, results_root: Path, expected_sequences: dict):
    """Check status of a single experiment."""
    experiment_folder = results_root / config_name
    csv_path = experiment_folder / 'reconstruction_sequence_stats.csv'

    if not experiment_folder.exists():
        return {
            'status': 'NOT_STARTED',
            'total_expected': sum(len(seqs) for seqs in expected_sequences.values()),
            'completed': 0,
            'failed_recon': 0,
            'failed_align': 0,
            'missing': sum(len(seqs) for seqs in expected_sequences.values()),
            'details': [],
        }

    if not csv_path.exists():
        return {
            'status': 'NO_CSV',
            'total_expected': sum(len(seqs) for seqs in expected_sequences.values()),
            'completed': 0,
            'failed_recon': 0,
            'failed_align': 0,
            'missing': sum(len(seqs) for seqs in expected_sequences.values()),
            'details': [],
        }

    df = pd.read_csv(csv_path)
    csv_entries = set(zip(df['dataset'].astype(str), df['sequence'].astype(str)))

    total_expected = 0
    completed = 0
    failed_recon = 0
    failed_align = 0
    missing_list = []

    for dataset_enum, sequences in expected_sequences.items():
        expected_pairs = get_expected_csv_sequences(dataset_enum, sequences)
        total_expected += len(expected_pairs)

        for csv_dataset, csv_seq in expected_pairs:
            # Try to find this entry in the CSV (may not match exactly due to BOP classic naming)
            matching = df[(df['dataset'].astype(str) == csv_dataset) &
                         (df['sequence'].astype(str) == csv_seq)]

            if matching.empty:
                # Try partial match for BOP classic sequences
                if dataset_enum == Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES:
                    matching = df[df['sequence'].astype(str) == csv_seq]

                if matching.empty:
                    missing_list.append(f'{csv_dataset}/{csv_seq}')
                    continue

            row = matching.iloc[0]
            recon_success = str(row.get('reconstruction_success', '')).lower() == 'true'
            align_success = str(row.get('alignment_success', '')).lower() == 'true'

            if not recon_success:
                failed_recon += 1
            elif not align_success:
                failed_align += 1
            else:
                completed += 1

    status = 'COMPLETE' if not missing_list and failed_recon == 0 else 'PARTIAL'
    if total_expected > 0 and completed + failed_recon + failed_align == 0:
        status = 'NO_RESULTS'

    return {
        'status': status,
        'total_expected': total_expected,
        'completed': completed,
        'failed_recon': failed_recon,
        'failed_align': failed_align,
        'missing': len(missing_list),
        'details': missing_list,
    }


def main():
    parser = argparse.ArgumentParser(description='Check experiment status')
    parser.add_argument('configs', nargs='*', help='Config names to check (default: all from job_runner)')
    parser.add_argument('--show-missing', action='store_true', help='List missing sequences')
    args = parser.parse_args()

    configurations = args.configs if args.configs else get_configurations()
    sequences = get_sequences()
    results_root = get_results_root()

    print(f'{"Experiment":<50} {"Status":<12} {"Done":>5} {"Fail":>5} {"NoAlign":>7} {"Miss":>5} {"Total":>6}')
    print('-' * 96)

    for config_name in configurations:
        result = check_experiment(config_name, results_root, sequences)

        status_str = result['status']
        print(f'{config_name:<50} {status_str:<12} {result["completed"]:>5} '
              f'{result["failed_recon"]:>5} {result["failed_align"]:>7} '
              f'{result["missing"]:>5} {result["total_expected"]:>6}')

        if args.show_missing and result['details']:
            for detail in result['details'][:10]:
                print(f'    MISSING: {detail}')
            if len(result['details']) > 10:
                print(f'    ... and {len(result["details"]) - 10} more')


if __name__ == '__main__':
    main()

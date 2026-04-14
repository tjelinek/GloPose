"""Check which experiments have completed, failed, or are missing results.

Usage:
    python scripts/check_experiment_status.py                    # check all experiments from job_runner
    python scripts/check_experiment_status.py onboarding/ufm_c0975r05 reconstruction/vggt  # check specific ones
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.job_runner import get_configurations, get_results_root, get_sequences, subsample_sequences, Datasets


# Maps Datasets enum → csv_dataset_name.
# Note: lookups use _CSV_NAMES_BY_VALUE (keyed by .value) to avoid identity
# mismatches when job_runner.py runs as __main__ vs scripts.job_runner.
DATASET_CSV_NAMES = {
    Datasets.BOP_HANDAL_ONBOARDING_STATIC: 'handal_static_onboarding',
    Datasets.BOP_HANDAL_ONBOARDING_BOTH: 'handal_static_onboarding',
    Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC: 'handal_dynamic_onboarding',
    Datasets.HOPE_ONBOARDING_STATIC: 'hope_static_onboarding',
    Datasets.HOPE_ONBOARDING_BOTH: 'hope_static_onboarding',
    Datasets.HOPE_ONBOARDING_DYNAMIC: 'hope_dynamic_onboarding',
    Datasets.HOT3D_ARIA_ONBOARDING_STATIC: 'hot3d_static_onboarding',
    Datasets.HOT3D_ARIA_ONBOARDING_DYNAMIC: 'hot3d_dynamic_onboarding',
    Datasets.HOT3D_QUEST3_ONBOARDING_STATIC: 'hot3d_static_onboarding',
    Datasets.HOT3D_QUEST3_ONBOARDING_DYNAMIC: 'hot3d_dynamic_onboarding',
    Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES: 'bop_classic',
    Datasets.HO3D_train: 'HO3D',
    Datasets.NAVI: 'navi',
}
_CSV_NAMES_BY_VALUE = {k.value: v for k, v in DATASET_CSV_NAMES.items()}


_BOP_ONBOARDING_DATASETS = {
    Datasets.BOP_HANDAL_ONBOARDING_STATIC.value, Datasets.BOP_HANDAL_ONBOARDING_BOTH.value,
    Datasets.BOP_HANDAL_ONBOARDING_DYNAMIC.value,
    Datasets.HOPE_ONBOARDING_STATIC.value, Datasets.HOPE_ONBOARDING_BOTH.value,
    Datasets.HOPE_ONBOARDING_DYNAMIC.value,
}

_HOT3D_DATASETS = {
    Datasets.HOT3D_ARIA_ONBOARDING_STATIC.value, Datasets.HOT3D_ARIA_ONBOARDING_DYNAMIC.value,
    Datasets.HOT3D_QUEST3_ONBOARDING_STATIC.value, Datasets.HOT3D_QUEST3_ONBOARDING_DYNAMIC.value,
}


def _hot3d_device_for(dataset_enum) -> str:
    """Return 'aria' or 'quest3' for HOT3D dataset enums."""
    if 'ARIA' in dataset_enum.value:
        return 'aria'
    if 'QUEST3' in dataset_enum.value:
        return 'quest3'
    raise ValueError(f'Not a HOT3D dataset: {dataset_enum}')


def get_expected_csv_sequences(dataset_enum, sequences: list[str]) -> list[tuple[str, str]]:
    """Return list of (csv_dataset_name, csv_sequence_name) expected in the CSV.

    Mirrors how runners write rows: CSV dataset = `run.dataset` + `_{onboarding_type}_onboarding`
    (for BOP-style onboarding runs), and CSV sequence = `run.sequence_{run.special_hash}`.
    """
    ds_value = dataset_enum.value
    csv_dataset = _CSV_NAMES_BY_VALUE.get(ds_value)
    if csv_dataset is None:
        return []

    results = []
    for seq in sequences:
        parts = seq.split('_')

        if ds_value in _BOP_ONBOARDING_DATASETS:
            # Job-runner sequence: `obj_NNNNNN_{up|down|both|dynamic}`
            # CSV sequence: `obj_NNNNNN_{up|down|both|dynamic}` (unchanged — special_hash equals suffix)
            results.append((csv_dataset, seq))

        elif ds_value in _HOT3D_DATASETS:
            # Job-runner sequence: `obj_NNNNNN_{up|down}` (aria/quest3 separated at dataset level)
            # CSV sequence: `obj_NNNNNN_{device}_{up|down}` — special_hash is `{device}_{suffix}`
            device = _hot3d_device_for(dataset_enum)
            if len(parts) == 3:
                base_seq = f'{parts[0]}_{parts[1]}'
                suffix = parts[2]
                csv_seq = f'{base_seq}_{device}_{suffix}'
                results.append((csv_dataset, csv_seq))
            else:
                results.append((csv_dataset, seq))

        elif ds_value == Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES.value:
            # Job-runner sequence: `{dataset}@{split}@{scene_id}` e.g. `tless@train_primesense@000001`
            # CSV: dataset = `{dataset}` (not `bop_classic`), sequence = `{scene_id}`
            split = seq.split('@')
            if len(split) == 3:
                results.append((split[0], split[2]))
            else:
                results.append((csv_dataset, seq))

        elif ds_value == Datasets.NAVI.value:
            # Job-runner sequence: `{obj_name}@{video_name}`
            # CSV sequence: `{obj_name}_{video_name}` (@ replaced with _)
            results.append((csv_dataset, seq.replace('@', '_', 1)))

        else:
            # HO3D: sequence name used as-is
            results.append((csv_dataset, seq))

    return results


def check_experiment(config_name: str, results_root: Path, expected_sequences: dict):
    """Check status of a single experiment."""
    experiment_folder = results_root / config_name
    csv_path = experiment_folder / 'reconstruction_sequence_stats.csv'

    all_expected = sum(len(seqs) for seqs in expected_sequences.values())

    if not experiment_folder.exists() or not csv_path.exists():
        return {
            'status': 'NOT_STARTED' if not experiment_folder.exists() else 'NO_CSV',
            'total_expected': all_expected,
            'completed': 0,
            'failed_recon': 0,
            'failed_align': 0,
            'missing': all_expected,
            'details': [],
            'missing_by_dataset': dict(expected_sequences),
        }

    df = pd.read_csv(csv_path)

    total_expected = 0
    completed = 0
    failed_recon = 0
    failed_align = 0
    missing_list = []
    missing_by_dataset: dict[Datasets, list[str]] = {}

    for dataset_enum, sequences in expected_sequences.items():
        expected_pairs = get_expected_csv_sequences(dataset_enum, sequences)
        total_expected += len(expected_pairs)

        for (csv_dataset, csv_seq), orig_seq in zip(expected_pairs, sequences):
            # Try to find this entry in the CSV (may not match exactly due to BOP classic naming)
            matching = df[(df['dataset'].astype(str) == csv_dataset) &
                         (df['sequence'].astype(str) == csv_seq)]

            if matching.empty:
                # Try partial match for BOP classic sequences
                if dataset_enum.value == Datasets.BOP_CLASSIC_ONBOARDING_SEQUENCES.value:
                    matching = df[df['sequence'].astype(str) == csv_seq]

                if matching.empty:
                    missing_list.append(f'{csv_dataset}/{csv_seq}')
                    missing_by_dataset.setdefault(dataset_enum, []).append(orig_seq)
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
        'missing_by_dataset': missing_by_dataset,
    }


def main():
    parser = argparse.ArgumentParser(description='Check experiment status')
    parser.add_argument('configs', nargs='*', help='Config names to check (default: all from job_runner)')
    parser.add_argument('--show-missing', action='store_true', help='List missing sequences')
    parser.add_argument('--quick', action='store_true',
                        help='Expect only the --quick subset (20 per dataset, matching job_runner --quick)')
    args = parser.parse_args()

    configurations = args.configs if args.configs else get_configurations()
    sequences = get_sequences()
    if args.quick:
        sequences = subsample_sequences(sequences, max_per_dataset=20)
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

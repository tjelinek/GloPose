import subprocess
import itertools
import argparse
from pathlib import Path


ALIASES = {
    'method': 'mth',
    'descriptor': 'dsc',
    'descriptor_mask_detections': 'msk',
    'device': 'dev',
    'augment_with_split_detections': 'aug_spl',
    'augment_with_train_pbr_detections': 'aug_pbr',
    'augmentations_detector': 'aug_det',
    'patch_descriptors_filtering': 'ptch_flt',
    'min_cls_cosine_similarity': 'cls_sim',
    'min_avg_patch_cosine_similarity': 'pch_sim',
    'dataset': 'ds',
    'split': 'spl',
}


def is_excluded(config, exclusions):
    for exclusion in exclusions:
        if all(config.get(key) == value for key, value in exclusion):
            return True
    return False


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}".replace('.', '')
    return value


def submit_job(config, failed_jobs_log=None, dry_run=False):
    job_name_parts = [f"{ALIASES.get(key, key)}_{format_value(value)}" for key, value in sorted(config.items())]
    job_name = '@'.join(job_name_parts)

    log_dir = '/mnt/personal/jelint19/results/logs/condensation_jobs'

    python_args = []
    for key, value in config.items():
        python_args.append(f'--{key}={value}')
    if failed_jobs_log and not dry_run:
        python_args.append(f'--failed_jobs_log={failed_jobs_log}')

    cmd = [
        'sbatch',
        '--job-name', job_name,
        '--error', f'{log_dir}/{job_name}.err',
        'scripts/compute_condensations.batch',
    ] + python_args

    if dry_run:
        python_cmd = f"python -m condensate_templates {' '.join(python_args)}"
        print(f"[DRY RUN] {python_cmd}")
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Submitted: {job_name} - {result.stdout.strip()}")
    else:
        print(f"Failed: {job_name} - {result.stderr.strip()}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    failed_jobs_log = '/mnt/personal/jelint19/results/logs/condensation_jobs/failed_jobs.log'
    failed_jobs_path = Path(failed_jobs_log)
    failed_jobs_path.parent.mkdir(parents=True, exist_ok=True)
    failed_jobs_path.write_text('')

    config_space = {
        'method': [
            'hart',
            # 'hart_symmetric',
            # 'hart_imblearn'
        ],
        'descriptor': ['dinov3'],
        'descriptor_mask_detections': [
            # 0,
            1
        ],
        'device': ['cpu'],
        'augment_with_split_detections': [
            # 0,
            1
        ],
        'augment_with_train_pbr_detections': [0, 1],
        'augmentations_detector': ['sam2'],
        'patch_descriptors_filtering': [0, 1],
        'min_cls_cosine_similarity': [0.15, 0.25, 0.5],
        'min_avg_patch_cosine_similarity': [0., 0.15, 0.25, 0.5],
    }

    config_spaces = [
        {
            **config_space,
            'dataset': ['hope'],
            'split': ['onboarding_static', 'onboarding_dynamic'],
        },
        {
            **config_space,
            'dataset': ['handal'],
            'split': ['onboarding_static', 'onboarding_dynamic'],
        },
        {
            **config_space,
            'dataset': ['tless'],
            'split': ['train_primesense'],
        },
        {
            **config_space,
            'dataset': ['lmo'],
            'split': ['train'],
        },
        {
            **config_space,
            'dataset': ['icbin'],
            'split': ['train'],
        },
        {
            **config_space,
            'dataset': ['hot3d'],
            'split': ['object_ref_aria_static_scenewise', 'object_ref_quest3_static_scenewise'],
        },
    ]

    exclusions = [
        [('patch_descriptors_filtering', 0), ('min_avg_patch_cosine_similarity', 0.15)],
        [('patch_descriptors_filtering', 0), ('min_avg_patch_cosine_similarity', 0.25)],
        [('patch_descriptors_filtering', 0), ('min_avg_patch_cosine_similarity', 0.5)],
        [('patch_descriptors_filtering', 1), ('min_avg_patch_cosine_similarity', 0.)],
    ]

    total_jobs = 0
    failed_jobs = 0
    excluded_jobs = 0

    for config_space in config_spaces:
        for values in itertools.product(*config_space.values()):
            config = dict(zip(config_space.keys(), values))
            if is_excluded(config, exclusions):
                excluded_jobs += 1
                continue
            total_jobs += 1
            if submit_job(config, failed_jobs_log=failed_jobs_log, dry_run=args.dry_run) != 0:
                failed_jobs += 1

    print(f"\nTotal jobs submitted: {total_jobs - failed_jobs}/{total_jobs}")
    print(f"Excluded combinations: {excluded_jobs}")
    if failed_jobs > 0:
        print(f"Failed submissions: {failed_jobs}")


if __name__ == '__main__':
    main()
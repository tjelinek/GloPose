import subprocess
import itertools
import argparse


def is_excluded(config, exclusions):
    for exclusion in exclusions:
        if all(config.get(key) == value for key, value in exclusion):
            return True
    return False


def format_value(value):
    if isinstance(value, float):
        return f"{value:.2f}".replace('.', '')
    return value


def submit_job(config, experiment_name=None):
    job_name_parts = [f"{key}_{format_value(value)}" for key, value in sorted(config.items()) if value is not None]
    if experiment_name:
        job_name_parts.append(experiment_name)
    job_name = '_'.join(job_name_parts)

    cmd = [
        'sbatch',
        '--job-name', job_name,
        '--error', f'/mnt/personal/jelint19/results/logs/condensation_jobs/{job_name}.err',
        'scripts/pose_estimator.batch',
    ]

    for key, value in config.items():
        if value is not None:
            cmd.append(f'--{key}={value}')

    if experiment_name:
        cmd.append(f'--experiment_name={experiment_name}')

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Submitted: {job_name} - {result.stdout.strip()}")
    else:
        print(f"Failed: {job_name} - {result.stderr.strip()}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default=None)
    args = parser.parse_args()

    config_space = {
        'descriptor': ['dinov2', 'dinov3'],
        'detector': ['sam', 'fastsam', 'sam2'],
        'use_enhanced_nms': [0, 1],
        'similarity_metric': ['cosine', 'csls'],
        'certainty': [0.15, 0.25, 0.5],
    }

    config_spaces = [
        {
            **config_space,
            'templates_source': ['cnns'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
        },
        {
            **config_space,
            'templates_source': ['prerendered'],
        }
    ]

    exclusions = [
        # [('use_enhanced_nms', 1), ('similarity_metric', 'csls')],
        # [('use_enhanced_nms', 1), ('similarity_metric', 'cosine')],
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
            if submit_job(config, experiment_name=args.experiment_name) != 0:
                failed_jobs += 1

    print(f"\nTotal jobs submitted: {total_jobs - failed_jobs}/{total_jobs}")
    print(f"Excluded combinations: {excluded_jobs}")
    if failed_jobs > 0:
        print(f"Failed submissions: {failed_jobs}")


if __name__ == '__main__':
    main()
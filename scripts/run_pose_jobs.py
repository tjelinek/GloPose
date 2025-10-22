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


def submit_job(config, experiment_folder=None, dry_run=False):
    job_name_parts = [f"{key}_{format_value(value)}" for key, value in sorted(config.items()) if value is not None]
    job_name = '_'.join(job_name_parts)

    log_dir = '/mnt/personal/jelint19/results/logs/pose_estimator'

    python_args = []
    for key, value in config.items():
        if value is not None:
            python_args.append(f'--{key}={value}')
    if experiment_folder:
        python_args.append(f'--experiment_folder={experiment_folder}')

    python_cmd = f"python -m pose.pose_estimator {' '.join(python_args)}"

    if dry_run:
        print(f"[DRY RUN] {python_cmd}")
        return 0

    cmd = [
              'sbatch',
              '--job-name', job_name,
              '--error', f'{log_dir}/{job_name}.err',
              'scripts/pose_estimator.batch',
          ] + python_args

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Submitted: {job_name} - {result.stdout.strip()}")
    else:
        print(f"Failed: {job_name} - {result.stderr.strip()}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_folder', default=None)
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    config_space = {
        'descriptor': [
            # 'dinov2',
            'dinov3'
        ],
        'detector': [
            # 'sam',
            # 'fastsam',
            'sam2'
        ],
        'use_enhanced_nms': [
            0,
            1
        ],
        'descriptor_mask_detections': [
            0,
            1
        ],
    }

    config_spaces = [
        {
            **config_space,
            'templates_source': ['cnns'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
            'similarity_metric': ['cosine', 'csls'],
            'ood_detection_method': ['none'],
        },
        {
            **config_space,
            'templates_source': ['cnns'],
            'aggregation_function': ['max'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
            'similarity_metric': ['cosine', 'csls'],
            'ood_detection_method': ['global_threshold'],
            'confidence_thresh': [0.15, 0.25, 0.5, 0.75],
        },
        {
            **config_space,
            'templates_source': ['cnns'],
            'aggregation_function': ['max'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
            'similarity_metric': ['cosine', 'csls'],
            'ood_detection_method': ['lowe_test'],
            'lowe_ratio_threshold': [1.05, 1.1, 1.25, 1.5],
        },
        {
            **config_space,
            'templates_source': ['cnns'],
            'aggregation_function': ['max'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
            'similarity_metric': ['cosine', 'csls'],
            'ood_detection_method': ['cosine_similarity_quantiles'],
            'cosine_similarity_quantile': [.25, .5, .75],
        },
        {
            **config_space,
            'templates_source': ['cnns'],
            'aggregation_function': ['max'],
            'condensation_source': ['1nn-hart', '1nn-hart_imblearn_adapted', '1nn-hart_imblearn', '1nn-hart_symmetric'],
            'ood_detection_method': ['mahalanobis_ood_detection'],
            'mahalanobis_quantile': [.95, .75, .5, .25],
        },
        {
            **config_space,
            'templates_source': ['prerendered'],
            'aggregation_function': ['max', 'avg_5'],
            'similarity_metric': ['cosine'],
            'ood_detection_method': ['global_threshold'],
            'confidence_thresh': [0.15, 0.25, 0.5],
        }
    ]

    exclusions = [
        [('use_enhanced_nms', 0), ('ood_detection_method', 'none')],
        [('use_enhanced_nms', 0), ('ood_detection_method', 'lowe_test')],
        [('use_enhanced_nms', 0), ('ood_detection_method', 'cosine_similarity_quantiles')],
        [('use_enhanced_nms', 0), ('ood_detection_method', 'mahalanobis_ood_detection')],
        [('descriptor_mask_detections', 0), ('ood_detection_method', 'none')],
        [('descriptor_mask_detections', 0), ('ood_detection_method', 'lowe_test')],
        [('descriptor_mask_detections', 0), ('ood_detection_method', 'cosine_similarity_quantiles')],
        [('descriptor_mask_detections', 0), ('ood_detection_method', 'mahalanobis_ood_detection')],
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
            if submit_job(config, experiment_folder=args.experiment_folder, dry_run=args.dry_run) != 0:
                failed_jobs += 1

    print(f"\nTotal jobs submitted: {total_jobs - failed_jobs}/{total_jobs}")
    print(f"Excluded combinations: {excluded_jobs}")
    if failed_jobs > 0:
        print(f"Failed submissions: {failed_jobs}")


if __name__ == '__main__':
    main()

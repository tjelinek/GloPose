import subprocess
import itertools
import argparse


def submit_job(descriptor, templates_source, condensation_source=None, certainty=None, detector='sam',
               experiment_name=None, use_enhanced_nms=1, similarity_metric='cosine'):

    job_name = f"{descriptor}_{templates_source}_{detector}"
    if condensation_source:
        job_name += f"_{condensation_source}"
    if certainty is not None:
        certainty_str = f"{certainty:.2f}".replace('.', '')
        job_name += f"_cert{certainty_str}"
    if experiment_name:
        job_name += f"_{experiment_name}"

    cmd = [
        'sbatch',
        '--job-name', job_name,
        '--error', f'/mnt/personal/jelint19/results/logs/condensation_jobs/{job_name}.err',
        'scripts/pose_estimator.batch',
        f'--descriptor={descriptor}',
        f'--templates_source={templates_source}',
        f'--detector={detector}',
        f'--use_enhanced_nms={use_enhanced_nms}',
        f'--similarity_metric={similarity_metric}'
    ]

    if condensation_source:
        cmd.append(f'--condensation_source={condensation_source}')

    if certainty is not None:
        cmd.append(f'--certainty={certainty}')

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

    descriptors = [
        'dinov2',
        'dinov3'
    ]
    condensation_sources = [
        '1nn-hart',
        '1nn-hart_imblearn_adapted',
        '1nn-hart_imblearn',
        '1nn-hart_symmetric'
    ]
    certainties = [0.15, 0.25, 0.5]
    detectors = [
        'sam',
        'fastsam',
        'sam2'
    ]
    use_enhanced_nms_values = [
        0,
        1
    ]
    similarity_metrics = [
        'cosine',
        'csls',
        'mahalanobis'
    ]

    total_jobs = 0
    failed_jobs = 0

    for descriptor, condensation_source, certainty, detector, use_enhanced_nms, similarity_metric in itertools.product(
            descriptors, condensation_sources, certainties, detectors, use_enhanced_nms_values, similarity_metrics):
        total_jobs += 1
        if submit_job(descriptor, 'cnns', condensation_source, certainty, detector,
                      experiment_name=args.experiment_name, use_enhanced_nms=use_enhanced_nms,
                      similarity_metric=similarity_metric) != 0:
            failed_jobs += 1

    for descriptor, certainty, detector, use_enhanced_nms, similarity_metric in itertools.product(
            descriptors, certainties, detectors, use_enhanced_nms_values, similarity_metrics):
        total_jobs += 1
        if submit_job(descriptor, 'prerendered', certainty=certainty, detector=detector,
                      experiment_name=args.experiment_name, use_enhanced_nms=use_enhanced_nms,
                      similarity_metric=similarity_metric) != 0:
            failed_jobs += 1

    print(f"\nTotal jobs submitted: {total_jobs - failed_jobs}/{total_jobs}")
    if failed_jobs > 0:
        print(f"Failed submissions: {failed_jobs}")


if __name__ == '__main__':
    main()
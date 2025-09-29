import subprocess
import itertools


def submit_job(descriptor, templates_source, condensation_source=None, certainty=None):
    """Submit a single SLURM job with the specified configuration."""
    cmd = [
        'sbatch',
        'scripts/pose_estimator.batch',
        f'--descriptor={descriptor}',
        f'--templates_source={templates_source}'
    ]

    if condensation_source:
        cmd.append(f'--condensation_source={condensation_source}')

    if certainty is not None:
        cmd.append(f'--certainty={certainty}')

    job_name = f"{descriptor}_{templates_source}"
    if condensation_source:
        job_name += f"_{condensation_source}"
    if certainty is not None:
        certainty_str = f"{certainty:.2f}".replace('.', '')
        job_name += f"_cert{certainty_str}"

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"Submitted: {job_name} - {result.stdout.strip()}")
    else:
        print(f"Failed: {job_name} - {result.stderr.strip()}")

    return result.returncode


def main():
    descriptors = ['dinov2', 'dinov3']
    condensation_sources = [
        '1nn-hart',
        '1nn-hart_imblearn_adapted',
        '1nn-hart_imblearn',
        '1nn-hart_symmetric'
    ]
    certainties = [0.15, 0.25, 0.5]

    # Track success
    total_jobs = 0
    failed_jobs = 0

    # Run all combinations with CNNs
    for descriptor, condensation_source, certainty in itertools.product(descriptors, condensation_sources, certainties):
        total_jobs += 1
        if submit_job(descriptor, 'cnns', condensation_source, certainty) != 0:
            failed_jobs += 1

    # Run prerendered with both descriptors
    for descriptor, certainty in itertools.product(descriptors, certainties):
        total_jobs += 1
        if submit_job(descriptor, 'prerendered', certainty=certainty) != 0:
            failed_jobs += 1

    print(f"\nTotal jobs submitted: {total_jobs - failed_jobs}/{total_jobs}")
    if failed_jobs > 0:
        print(f"Failed submissions: {failed_jobs}")


if __name__ == '__main__':
    main()
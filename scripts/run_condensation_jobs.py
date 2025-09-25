#!/usr/bin/env python3
"""
Script to submit SLURM jobs for all combinations of datasets, descriptors, and methods.
"""

import subprocess
import time
from itertools import product

# Define all combinations to run
METHODS = [
    'hart',
    'hart_symmetric',
    'hart_imblearn',
    'hart_imblearn_adapted'
]

DESCRIPTORS = ['dinov2', 'dinov3']

DATASETS = [
    ('hope', 'onboarding_static'),
    ('hope', 'onboarding_dynamic'),
    ('handal', 'onboarding_static'),
    ('handal', 'onboarding_dynamic'),
    ('tless', 'train_primesense'),
    ('lmo', 'train'),
    ('icbin', 'train'),
]


def submit_job(method, descriptor, dataset, split):
    """Submit a single SLURM job."""
    job_name = f"condensation-{method}-{descriptor}-{dataset}-{split}"

    cmd = [
        'sbatch',
        '--job-name', job_name,
        'scripts/compute_condensations.batch',  # Your SLURM script name
        '--method', method,
        '--descriptor', descriptor,
        '--dataset', dataset,
        '--split', split
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]  # Extract job ID from output
        print(f"Submitted job {job_id}: {job_name}")
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Failed to submit job {job_name}: {e}")
        print(f"Error output: {e.stderr}")
        return None


def main():
    """Submit all job combinations."""
    submitted_jobs = []
    total_jobs = len(METHODS) * len(DESCRIPTORS) * len(DATASETS)

    print(f"Submitting {total_jobs} jobs...")
    print("=" * 50)

    for method, descriptor, (dataset, split) in product(METHODS, DESCRIPTORS, DATASETS):
        job_id = submit_job(method, descriptor, dataset, split)
        if job_id:
            submitted_jobs.append(job_id)

        # Small delay to avoid overwhelming the scheduler
        time.sleep(0.1)

    print("=" * 50)
    print(f"Successfully submitted {len(submitted_jobs)}/{total_jobs} jobs")

    if submitted_jobs:
        print("\nTo monitor job status, use:")
        print(f"squeue -u $USER")
        print("\nTo cancel all jobs, use:")
        print(f"scancel {' '.join(submitted_jobs)}")


if __name__ == '__main__':
    main()
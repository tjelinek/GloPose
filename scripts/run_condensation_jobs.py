#!/usr/bin/env python3
"""
Script to submit SLURM jobs for all combinations of datasets, descriptors, and methods.
"""

import subprocess
import time
from itertools import product
from pathlib import Path

# Define all combinations to run
METHODS = [
    'hart',
    'hart_symmetric',
    'hart_imblearn',
    'hart_imblearn_adapted'
]

DESCRIPTORS = [
    'dinov2',
    'dinov3'
]

WHITEN_DIM = [
    0,
    64,
    128,
    256
]

DATASETS = [
    ('hope', 'onboarding_static'),
    ('hope', 'onboarding_dynamic'),
    ('handal', 'onboarding_static'),
    ('handal', 'onboarding_dynamic'),
    ('tless', 'train_primesense'),
    ('lmo', 'train'),
    ('icbin', 'train'),
    ('hot3d', 'object_ref_aria_static_scenewise'),
    ('hot3d', 'object_ref_quest3_static_scenewise'),
    # ('hot3d', 'object_ref_aria_dynamic_scenewise'),
    # ('hot3d', 'object_ref_quest3_dynamic_scenewise'),
]


def submit_job(method, descriptor, whiten_dim, dataset, split):
    """Submit a single SLURM job."""
    job_name = f"cond-{method}-{descriptor}-{dataset}-{split}_whiten-dim{whiten_dim}"
    log_name = f"condensation_{method}_{descriptor}_{dataset}_{split}_whiten-dim{whiten_dim}"

    log_path = Path('/mnt/personal/jelint19/results/logs/condensation_jobs')
    log_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        'sbatch',
        '--job-name', job_name,
        '--error', f'/mnt/personal/jelint19/results/logs/condensation_jobs/{log_name}.err',
        'scripts/compute_condensations.batch',  # Your SLURM script name
        '--method', method,
        '--descriptor', descriptor,
        '--whiten_dim', str(whiten_dim),
        '--dataset', dataset,
        '--split', split,
        '--device', 'cpu',
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
    total_jobs = len(METHODS) * len(DESCRIPTORS) * len(WHITEN_DIM) * len(DATASETS)

    print(f"Submitting {total_jobs} jobs...")
    print("=" * 50)

    for method, descriptor, whiten_dim, (dataset, split) in product(METHODS, DESCRIPTORS, WHITEN_DIM, DATASETS):
        job_id = submit_job(method, descriptor, whiten_dim, dataset, split)
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
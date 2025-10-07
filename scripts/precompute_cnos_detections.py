import subprocess
from pathlib import Path
import argparse

DATASETS = [
    'lmo',
    'tless',
    'icbin',
    'handal',
    'hope',
    'handal-val',
    'hope-val',
    'hot3d-quest3',
    'hot3d-aria',
    'hot3d-quest3-train',
    'hot3d-aria-train',
]

DETECTORS = [
    'sam',
    'fastsam',
    'sam2']


def submit_job(dataset, detector):
    """Submit a single SLURM job."""
    job_name = f"PrecomputeCNOS_{detector}_{dataset}"
    log_name = f"PrecomputeCNOSDetections_{detector}_{dataset}"

    # Create log directory if it doesn't exist
    log_path = Path('/mnt/personal/jelint19/results/logs/precompute_detections')
    log_path.mkdir(parents=True, exist_ok=True)

    error_log = log_path / f"{log_name}.err"

    # Determine batch script path
    if Path.cwd().name == 'scripts':
        batch_script = 'precompute_cnos_detections_job.batch'
    else:
        batch_script = 'scripts/precompute_cnos_detections_job.batch'

    cmd = [
        'sbatch',
        '--job-name', job_name,
        '--error', str(error_log),
        '--output', str(error_log),  # Redirect stdout to same file as stderr
        batch_script,
        '--dataset', dataset,
        '--detector', detector,
    ]

    print(f"Submitting job for dataset: {dataset}, detector: {detector}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  Job submitted: {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"  Error submitting job: {e.stderr}")


def main():
    parser = argparse.ArgumentParser(
        description='Submit CNOS detection precomputation jobs'
    )
    parser.add_argument(
        '--detector',
        type=str,
        choices=DETECTORS + ['all'],
        default='all',
        help='Detector to use (default: all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=DATASETS + ['all'],
        default='all',
        help='Dataset to process (default: all)'
    )

    args = parser.parse_args()

    # Determine which detectors to run
    detectors = DETECTORS if args.detector == 'all' else [args.detector]

    # Determine which datasets to process
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]

    # Submit jobs
    for detector in detectors:
        for dataset in datasets:
            submit_job(dataset, detector)


if __name__ == '__main__':
    main()

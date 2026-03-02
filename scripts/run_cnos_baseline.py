"""Run original CNOS pipeline for baseline comparison.

Invokes repositories/cnos/run_inference.py with the specified dataset and
rendering type. Used for the 3-way comparison in the paper:
  1. CNOS + BOP onboarding frames (onboarding_static)
  2. CNOS + our keyframes (matchability_images_{split})
  3. GloPose detection

Usage:
    python scripts/run_cnos_baseline.py --dataset hope --rendering_type onboarding_static
    python scripts/run_cnos_baseline.py --dataset handal --rendering_type matchability_images_both
    python scripts/run_cnos_baseline.py --dataset hope --rendering_type onboarding_static --descriptor_model dinov3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Run original CNOS baseline for comparison')
    parser.add_argument('--dataset', required=True,
                        help='BOP dataset name (e.g., hope, handal, tless, lmo, icbin)')
    parser.add_argument('--rendering_type', required=True,
                        help='Template rendering type (e.g., onboarding_static, onboarding_dynamic, '
                             'matchability_images_both, pbr, pyrender)')
    parser.add_argument('--descriptor_model', default=None,
                        help='Descriptor model override (e.g., dinov2, dinov3)')
    parser.add_argument('--aggregation_function', default=None,
                        help='Aggregation function override (e.g., max, mean)')
    parser.add_argument('--confidence_thresh', type=float, default=None,
                        help='Confidence threshold override')
    parser.add_argument('--split', default=None,
                        help='Dataset split (e.g., test, val)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print command without executing')
    parser.add_argument('--extra_overrides', nargs='*', default=[],
                        help='Additional Hydra overrides (e.g., model.matching_config.nms_thresh=0.3)')

    args = parser.parse_args()

    cnos_dir = Path(__file__).parent.parent / 'repositories' / 'cnos'
    if not cnos_dir.exists():
        print(f"Error: CNOS directory not found at {cnos_dir}", file=sys.stderr)
        sys.exit(1)

    # Build Hydra overrides
    overrides = [
        f'dataset_name={args.dataset}',
        f'model.onboarding_config.rendering_type={args.rendering_type}',
    ]

    if args.descriptor_model is not None:
        overrides.append(f'model/descriptor_model={args.descriptor_model}')
    if args.aggregation_function is not None:
        overrides.append(f'model.matching_config.aggregation_function={args.aggregation_function}')
    if args.confidence_thresh is not None:
        overrides.append(f'model.matching_config.confidence_thresh={args.confidence_thresh}')
    if args.split is not None:
        overrides.append(f'split={args.split}')

    overrides.extend(args.extra_overrides)

    cmd = [sys.executable, 'run_inference.py'] + overrides

    print(f"Working directory: {cnos_dir}")
    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("(dry run — not executing)")
        return

    result = subprocess.run(cmd, cwd=str(cnos_dir))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()

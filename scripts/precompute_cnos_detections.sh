#!/bin/bash
# submit_all.sh

datasets=(
  lmo
  tless
  icbin
  handal
  hope
  handal-val
  hope-val
)
#  tudl
#  itodd
#  hb_kinect
#  hb_primesense
#  ycbv

# Parse optional detector argument
detector=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --detector)
      detector="${2:-}"; shift 2 ;;
    --detector=*)
      detector="${1#*=}"; shift ;;
    *)
      echo "Unknown argument: $1" >&2; shift ;;
  esac
done

# Decide which sbatch path to use
if [[ $(basename "$PWD") == "scripts" ]]; then
  sbatch_cmd="sbatch precompute_cnos_detections_job.batch"
else
  sbatch_cmd="sbatch scripts/precompute_cnos_detections_job.batch"
fi

for ds in "${datasets[@]}"; do
  echo "Submitting job for dataset: $ds"

  # Build command with dataset
  cmd_args="--dataset $ds"

  # Add detector if specified
  if [[ -n "${detector}" ]]; then
    cmd_args="$cmd_args --detector $detector"
  fi

  $sbatch_cmd $cmd_args
done

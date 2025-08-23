#!/bin/bash
# submit_all.sh

datasets=(
  lmo
  tless
  tudl
  icbin
  itodd
  hb_kinect
  hb_primesense
  ycbv
  handal
)

# Decide which sbatch path to use
if [[ $(basename "$PWD") == "scripts" ]]; then
  sbatch_cmd="sbatch precompute_cnos_detections_job.batch"
else
  sbatch_cmd="sbatch scripts/precompute_cnos_detections_job.batch"
fi

for ds in "${datasets[@]}"; do
  echo "Submitting job for dataset: $ds"
  $sbatch_cmd --dataset "$ds"
done

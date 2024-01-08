#!/bin/bash

# List of parameters (without the .yaml extension)
params=(
  "config_deep_no_flow_all_frames_keyframes"
  "config_deep_no_flow_gt_mesh"
  "config_deep_no_flow"
  "config_deep_only_flow_gt_mesh"
  "config_deep_only_flow_gt_one_kf"
  "config_deep_only_flow_gt_two_kf"
  "config_deep_only_flow"
  "config_deep"
  "config_deep_with_flow_all_frames_keyframes"
  "config_deep_with_flow_gt_no_rgb"
  "config_deep_with_flow_gt"
  "config_deep_with_flow_no_rgb"
  "config_deep_with_flow"
  )

# Loop through the parameters and run the sbatch command
for param in "${params[@]}"
do
  sbatch job.batch "$param"
done


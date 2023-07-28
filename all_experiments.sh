#!/bin/bash

# List of parameters (without the .yaml extension)
params=(
  "config_deep_no_flow_gt"
  "config_deep_no_flow_stochastically_add_keyframes"
  "config_deep_no_flow_tex_change_reg"
  "config_deep_no_flow"
  "config_deep_only_flow_gt_one_kf"
  "config_deep_only_flow_gt"
  "config_deep_only_iou_and_flow_gt_fewer_frames"
  "config_deep_only_iou_and_flow_gt_max_keyframes"
  "config_deep_with_flow_gt_no_rgb"
  "config_deep_with_flow_gt"
  "config_deep_with_flow_max_keyframes"
  "config_deep_with_flow_stochastically_add_keyframes"
  "config_deep_with_flow_tex_change_reg"
  "config_deep_with_flow"
  "config_deep"
  )

# Loop through the parameters and run the sbatch command
for param in "${params[@]}"
do
  sbatch job.batch "$param"
done


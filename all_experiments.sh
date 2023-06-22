#!/bin/bash

# List of parameters (without the .yaml extension)
params=(
  "config_deep_no_flow_gt.yaml"
  "config_deep_no_flow_no_laplacian.yaml"
  "config_deep_no_flow_stochastically_add_keyframes.yaml"
  "config_deep_no_flow_tex_change_reg.yaml"
  "config_deep_no_flow.yaml"
  "config_deep_only_iou_and_flow_gt_fewer_frames.yaml"
  "config_deep_only_iou_and_flow_gt_max_keyframes.yaml"
  "config_deep_with_flow_gt_no_rgb.yaml"
  "config_deep_with_flow_gt.yaml"
  "config_deep_with_flow_max_keyframes.yaml"
  "config_deep_with_flow_no_laplacian.yaml"
  "config_deep_with_flow_stochastically_add_keyframes.yaml"
  "config_deep_with_flow_tex_change_reg.yaml"
  "config_deep_with_flow.yaml"
  "config_deep.yaml"
  )

# Loop through the parameters and run the sbatch command
for param in "${params[@]}"
do
  sbatch job.batch "$param"
done


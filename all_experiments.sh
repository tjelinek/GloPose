#!/bin/bash

# List of parameters (without the .yaml extension)
params=(
  "config_deep_max_keyframes_no_flow_no_motion_reg"
  "config_deep_max_keyframes_no_flow"
  "config_deep_max_keyframes_no_motion_reg"
  "config_deep_max_keyframes_only_flow_gt"
  "config_deep_max_keyframes"
  "config_deep_no_flow_gt_no_rgb"
  "config_deep_no_flow_gt"
  "config_deep_no_flow_tex_cahnge_reg"
  "config_deep_no_flow"
  "config_deep_only_flow_gt"
  "config_deep_only_iou_and_flow_gt_fewer_frames"
  "config_deep_only_iou_and_flow_gt"
  "config_deep_only_iou_gt"
  "config_deep_tex_change_reg"
  "config_deep_with_flow_gt_no_move_reg"
  "config_deep_with_flow_gt_no_rgb_no_move_reg"
  "config_deep_with_flow_gt_no_rgb"
  "config_deep_with_flow_gt_texture_change_reg_all_keys_keyframes"
  "config_deep_with_flow_gt_texture_change_reg"
  "config_deep_with_flow_gt"
  "config_deep"
)

# Loop through the parameters and run the sbatch command
for param in "${params[@]}"
do
  sbatch job.batch "$param"
done


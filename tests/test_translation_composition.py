from pathlib import Path

import kaolin
import numpy as np
import torch
from kornia.geometry import (Quaternion, rotation_matrix_to_axis_angle, Se3)

from utils.math_utils import (Se3_last_cam_to_world_from_Se3_obj, Se3_obj_from_epipolar_Se3_cam)
from dataset_generators import scenarios
from dataset_generators.track_augmentation import modify_rotations
from flow import flow_unit_coords_to_image_coords, source_to_target_coords_world_coord_system
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from pose.essential_matrix_pose_estimation import estimate_pose_zaragoza
from tracker_config import TrackerConfig
from utils.general import normalize_vertices, get_not_occluded_foreground_points, erode_segment_mask2, \
    tensor_index_to_coordinates_xy

sequence_len = 30
config = TrackerConfig()
config.camera_up = (0, 1, 0)
# config.camera_position = (0, 0, 5)
# config.camera_position = (10, 0, 0)
config.camera_position = (-6, 5.18, 10)
config.input_frames = sequence_len

path = Path('./prototypes/sphere.obj')
mesh = kaolin.io.obj.import_mesh(str(path), with_materials=True)
ivertices = normalize_vertices(mesh.vertices).numpy()
faces = mesh.faces.numpy()
iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

h = 564
w = 300
scenario = scenarios.generate_rotations_xyz(5.0)
scenario_t = scenarios.generate_y_translation(36)
gt_rotation = torch.from_numpy(scenario.rotation_axis_angles)[:sequence_len].cuda()  # * 0
gt_rotation[..., 0] *= -2.
gt_rotation[..., 1] *= 1.
gt_rotation[..., 2] *= -0.5

# gt_translation = torch.zeros(1, 1, sequence_len, 3).cuda()
gt_translation = torch.from_numpy(np.asarray(scenario_t.translations)).cuda()[:sequence_len].to(torch.float32) * 0
gt_translation[..., 0] *= 2.
gt_translation[..., 1] *= 1.
gt_translation[..., 2] *= 0.5

first_frame = 0
source_frame = 6
target_frame = 11

if config.augment_gt_track or True:
    gt_rotation, gt_translation = modify_rotations(gt_rotation, gt_translation)
    gt_rotation = gt_rotation[:sequence_len]
    gt_translation = gt_translation[:sequence_len]


def predict_camera_pose_using_zaragoza(source_frame_, target_frame_, config_, rendering_):
    flow_observation = rendering_.render_flow_for_frame(gt_encoder, source_frame_, target_frame_)
    segmentation = erode_segment_mask2(7, flow_observation.adjusted_segmentation[0])[None]
    src_pts_yx, observed_visible_fg_points_mask = (
        get_not_occluded_foreground_points(flow_observation.observed_flow_occlusion,
                                           segmentation,
                                           config_.occlusion_coef_threshold,
                                           config_.segmentation_mask_threshold))
    flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)
    dst_pts_yx = source_to_target_coords_world_coord_system(src_pts_yx, flow)
    src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
    dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)
    K1 = rendering_.camera_intrinsics
    rot_cam_, t_cam_ = estimate_pose_zaragoza(src_pts_xy, dst_pts_xy, K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2])

    quat_cam_ = Quaternion.from_axis_angle(rot_cam_[None])
    Se3_cam_ = Se3(quat_cam_, t_cam_[None, ..., 0])

    return Se3_cam_


config.rot_init = tuple(gt_rotation[0].numpy(force=True))
config.tran_init = tuple(gt_translation[0].numpy(force=True))

gt_encoder = Encoder(config, ivertices, iface_features, w, h, 3).cuda()
gt_encoder.set_encoder_poses(gt_rotation, gt_translation)

rendering = RenderingKaolin(config, faces, w, h).cuda()

Se3_world_to_cam_1st_frame = rendering.camera_transformation_matrix_Se3()

quat_obj_gt = Quaternion.from_axis_angle(gt_rotation.to(torch.float32))
Se3_obj_gt = Se3(quat_obj_gt, gt_translation)

axis_angle_ref_to_last_gt = gt_rotation[[target_frame]] - gt_rotation[[source_frame]]
quat_ref_to_last_gt = Quaternion.from_axis_angle(axis_angle_ref_to_last_gt)
translation_ref_to_last_gt = gt_translation[[target_frame]] - gt_translation[[source_frame]]

Se3_obj_ref_to_last_gt_prime = Se3(quat_ref_to_last_gt, translation_ref_to_last_gt)

Se3_obj_ref_to_last_gt = Se3_obj_gt[[target_frame]] * Se3_obj_gt[[source_frame]].inverse()

Se3_cam_ref_to_last = predict_camera_pose_using_zaragoza(source_frame, target_frame, config, rendering)
Se3_cam_first_to_ref = predict_camera_pose_using_zaragoza(0, source_frame, config, rendering)
Se3_cam_first_to_last = predict_camera_pose_using_zaragoza(0, target_frame, config, rendering)

Se3_cam_ref_to_last_gt_unsure = Se3_last_cam_to_world_from_Se3_obj(Se3_obj_ref_to_last_gt, Se3_world_to_cam_1st_frame)

Se3_obj_ref_to_last_pred = Se3_obj_from_epipolar_Se3_cam(Se3_cam_ref_to_last, Se3_world_to_cam_1st_frame)

rot_obj_ref_to_last_pred = rotation_matrix_to_axis_angle(Se3_obj_ref_to_last_pred.quaternion.matrix()).squeeze()
t_obj_ref_to_last_pred = Se3_obj_ref_to_last_pred.translation.data.squeeze(-1)  # Shape (1, 3, 1) -> (1, 3)

rot_obj_ref_to_last_gt = rotation_matrix_to_axis_angle(Se3_obj_ref_to_last_gt.quaternion.matrix()).squeeze()

Se3_obj_1st_to_last_pred = Se3_obj_gt[[source_frame]] * Se3_obj_ref_to_last_pred
Se3_obj_1st_to_last_gt = Se3_obj_gt[[target_frame]]

rot_obj_1st_to_last_pred = rotation_matrix_to_axis_angle(Se3_obj_1st_to_last_pred.quaternion.matrix()).squeeze()
rot_obj_1st_to_last_gt = rotation_matrix_to_axis_angle(Se3_obj_1st_to_last_gt.quaternion.matrix()).squeeze()

# TODO It is not clear why do I need to perform inverse but is works
Se3_obj_first_to_last_base = Se3_obj_from_epipolar_Se3_cam(Se3_cam_first_to_last, Se3_world_to_cam_1st_frame)
rot_obj_first_to_last_base = rotation_matrix_to_axis_angle(Se3_obj_first_to_last_base.quaternion.matrix()).squeeze()

print('----------------------------------------')
print(f'T obj  : {Se3_obj_first_to_last_base.translation.squeeze().numpy(force=True).round(3)}')
print(f'Rot cam pred: {torch.rad2deg(rotation_matrix_to_axis_angle(Se3_cam_first_to_last.quaternion.matrix())).numpy(force=True).round(3)}')
print(f'Rot obj ref to last pred: {torch.rad2deg(rot_obj_ref_to_last_pred).numpy(force=True).round(3)}')
print(f'Rot obj ref to last   gt: {torch.rad2deg(rot_obj_ref_to_last_gt).numpy(force=True).round(3)}')
print(f'Rot obj 1st to last pred: {torch.rad2deg(rot_obj_1st_to_last_pred).numpy(force=True).round(3)}')
print(f'Rot obj 1st to last   gt: {torch.rad2deg(rot_obj_1st_to_last_gt).numpy(force=True).round(3)}')
print(f'Rot obj 1st to last base: {torch.rad2deg(rot_obj_first_to_last_base).numpy(force=True).round(3)}')
print('----------------------------------------')

breakpoint()

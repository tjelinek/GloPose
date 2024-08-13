from pathlib import Path

import kaolin
import numpy as np
import torch
from kornia.geometry import Quaternion, axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam
from dataset_generators import scenarios
from flow import flow_unit_coords_to_image_coords, source_to_target_coords_world_coord_system
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from pose.essential_matrix_pose_estimation import estimate_pose_zaragoza
from tracker_config import TrackerConfig
from utils import normalize_vertices, get_not_occluded_foreground_points, erode_segment_mask2, \
    tensor_index_to_coordinates_xy

sequence_len = 2
config = TrackerConfig()
config.camera_up = (0, 1, 0)
config.camera_position = (-6, 5.18, 10)
config.input_frames = sequence_len

path = Path('./prototypes/sphere.obj')
mesh = kaolin.io.obj.import_mesh(str(path), with_materials=True)
ivertices = normalize_vertices(mesh.vertices).numpy()
faces = mesh.faces.numpy()
iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

h = 564
w = 300
scenario = scenarios.generate_rotations_xyz(10.0)
scenario_t = scenarios.generate_xyz_translation(36)
gt_rotation = torch.from_numpy(scenario.rotation_axis_angles)[:sequence_len].cuda()# * 0
gt_rotation[..., 0] *= -2.
gt_rotation[..., 1] *= 1.
gt_rotation[..., 2] *= -0.5

# gt_translation = torch.zeros(1, 1, sequence_len, 3).cuda()
gt_translation = torch.from_numpy(np.asarray(scenario_t.translations)).cuda()[:sequence_len].to(torch.float32) * 0
gt_translation[..., 0] *= 2.
gt_translation[..., 1] *= 1.
gt_translation[..., 2] *= 0.5

gt_encoder = Encoder(config, ivertices, iface_features, w, h, 3).cuda()
gt_encoder.set_encoder_poses(gt_rotation, gt_translation)

rendering = RenderingKaolin(config, faces, w, h).cuda()
W_4x4 = rendering.camera_transformation_matrix_4x4().permute(0, 2, 1)

flow_observation = rendering.render_flow_for_frame(gt_encoder, 0, 1)
segmentation = erode_segment_mask2(7, flow_observation.observed_flow_segmentation[0])[None]

src_pts_yx, observed_visible_fg_points_mask = (
    get_not_occluded_foreground_points(flow_observation.observed_flow_occlusion,
                                       segmentation,
                                       config.occlusion_coef_threshold,
                                       config.segmentation_mask_threshold))

flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)

dst_pts_yx = source_to_target_coords_world_coord_system(src_pts_yx, flow)

src_pts_xy = tensor_index_to_coordinates_xy(src_pts_yx)
dst_pts_xy = tensor_index_to_coordinates_xy(dst_pts_yx)

K1 = rendering.camera_intrinsics

rot_cam, t_cam = estimate_pose_zaragoza(src_pts_xy, dst_pts_xy, K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2])
# rot_cam, t_cam, inlier_mask, _ = estimate_pose_using_2D_2D_E_solver(src_pts_yx, dst_pts_yx, K1, K1, w, h, config, None)

R_cam = axis_angle_to_rotation_matrix(rot_cam[None])
quat_cam = Quaternion.from_matrix(R_cam)

R_obj, t_obj = Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam[None], W_4x4)
rot_obj = rotation_matrix_to_axis_angle(R_obj).squeeze()
t_obj = t_obj[..., 0]  # Shape (1, 3, 1) -> (1, 3)

print('----------------------------------------')
print(f'T_world_to_cam\n{W_4x4}')
print(f'T cam  : {t_cam.squeeze().numpy(force=True).round(3)}')
print(f'T obj  : {t_obj.squeeze().numpy(force=True).round(3)}')
print(f'Rot cam: {torch.rad2deg(rot_cam).numpy(force=True).round(3)}')
print(f'Rot obj: {torch.rad2deg(rot_obj).numpy(force=True).round(3)}')
print('----------------------------------------')

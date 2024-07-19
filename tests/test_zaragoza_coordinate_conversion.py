from pathlib import Path

import kaolin
import numpy as np
import torch
from kornia.geometry import Quaternion, axis_angle_to_rotation_matrix

from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam
from dataset_generators import scenarios
from flow import source_coords_to_target_coords, flow_unit_coords_to_image_coords
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from pose.essential_matrix_pose_estimation import estimate_pose_using_directly_zaragoza, \
    estimate_pose_using_2D_2D_E_solver
from tracker_config import TrackerConfig
from utils import normalize_vertices, get_not_occluded_foreground_points, erode_segment_mask2, \
    tensor_index_to_coordinates_xy

sequence_len = 2
config = TrackerConfig()
config.camera_up = (1, 0, 0)
config.camera_position = (0, 0, 5)
config.input_frames = sequence_len

path = Path('./prototypes/sphere.obj')
mesh = kaolin.io.obj.import_mesh(str(path), with_materials=True)
ivertices = normalize_vertices(mesh.vertices).numpy()
faces = mesh.faces.numpy()
iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

h = 500
w = 500
scenario = scenarios.generate_rotations_xyz(10.0)
scenario_t = scenarios.generate_xyz_translation(36)
gt_rotation = torch.from_numpy(scenario.rotation_axis_angles)[None, :sequence_len].cuda()# * 0
gt_rotation[..., 0] *= 2.
gt_rotation[..., 1] *= 1.
gt_rotation[..., 2] *= 0.5

# gt_translation = torch.zeros(1, 1, sequence_len, 3).cuda()
gt_translation = torch.from_numpy(np.asarray(scenario_t.translations)).cuda()[None, None, :sequence_len].to(torch.float32) * 0
gt_translation[..., 0] *= 2.
gt_translation[..., 1] *= 1.
gt_translation[..., 2] *= 0.5

gt_encoder = Encoder(config, ivertices, iface_features, w, h, 3).cuda()
gt_encoder.set_encoder_poses(gt_rotation, gt_translation)

rendering = RenderingKaolin(config, faces, w, h).cuda()
W_4x4 = rendering.camera_transformation_matrix_4x4()

flow_observation = rendering.render_flow_for_frame(gt_encoder, 0, 1)
segmentation = erode_segment_mask2(7, flow_observation.rendered_flow_segmentation[0])[None]

src_pts_yx, observed_visible_fg_points_mask = (
    get_not_occluded_foreground_points(flow_observation.rendered_flow_occlusion.permute(0, 1, 2, 4, 3),
                                       segmentation.permute(0, 1, 2, 4, 3),
                                       config.occlusion_coef_threshold,
                                       config.segmentation_mask_threshold))

flow = flow_unit_coords_to_image_coords(flow_observation.theoretical_flow)
dst_pts_yx = source_coords_to_target_coords(src_pts_yx.permute(1, 0), flow).permute(1, 0)

src_pts_xy = src_pts_yx[..., [1, 0]]
flow_xy = flow[:, :, [1, 0]].permute(0, 1, 2, 4, 3)
dst_pts_xy = source_coords_to_target_coords(src_pts_xy.permute(1, 0), flow_xy).permute(1, 0)

src_pts_xy2 = tensor_index_to_coordinates_xy(src_pts_yx)
dst_pts_xy2 = tensor_index_to_coordinates_xy(dst_pts_yx)

try:
    assert torch.equal(src_pts_xy2, src_pts_xy) and torch.equal(dst_pts_xy2, dst_pts_xy)
except:
    print(f'dst_pts_yx2\n{dst_pts_xy}')
    print(f'dst_pts_yx2\n{dst_pts_xy2}')

K1 = rendering.camera_intrinsics

rot_cam, t_cam = estimate_pose_zaragoza(src_pts_xy, dst_pts_xy, K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2])
# rot_cam, t_cam, inlier_mask, _ = estimate_pose_using_2D_2D_E_solver(src_pts_yx, dst_pts_yx, K1, K1, w, h, config, None)
src_pts_xy = src_pts_xy2
dst_pts_xy = dst_pts_xy2

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

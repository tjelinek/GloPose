from pathlib import Path

import kaolin
import torch
from kornia.geometry import Quaternion, axis_angle_to_rotation_matrix

from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam
from dataset_generators import scenarios
from flow import source_coords_to_target_coords
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from pose.essential_matrix_pose_estimation import estimate_pose_using_directly_zaragoza
from tracker_config import TrackerConfig
from utils import normalize_vertices, get_not_occluded_foreground_points, pinhole_intrinsics_from_tensor

sequence_len = 2
config = TrackerConfig()
config.input_frames = sequence_len

path = Path('./prototypes/sphere.obj')
mesh = kaolin.io.obj.import_mesh(str(path), with_materials=True)
ivertices = normalize_vertices(mesh.vertices).numpy()
faces = mesh.faces.numpy()
iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

h = 500
w = 500
scenario = scenarios.generate_rotations_z(10.0)
gt_rotation = torch.from_numpy(scenario.rotation_axis_angles)[None, :sequence_len].cuda()
gt_translation = torch.zeros(1, 1, sequence_len, 3).cuda()

gt_encoder = Encoder(config, ivertices, iface_features, w, h, 3).cuda()
gt_encoder.set_encoder_poses(gt_rotation, gt_translation)

rendering = RenderingKaolin(config, faces, w, h).cuda()
W_4x4 = rendering.camera_transformation_matrix_4x4()

flow_observation = rendering.render_flow_for_frame(gt_encoder, 0, 1)

src_pts_yx, observed_visible_fg_points_mask = (
    get_not_occluded_foreground_points(flow_observation.rendered_flow_occlusion,
                                       flow_observation.rendered_flow_segmentation,
                                       config.occlusion_coef_threshold,
                                       config.segmentation_mask_threshold))
dst_pts_yx = source_coords_to_target_coords(src_pts_yx.permute(1, 0), flow_observation.theoretical_flow).permute(1, 0)
K1 = rendering.camera_intrinsics

rot_cam, t_cam, inlier_mask, _ = estimate_pose_using_directly_zaragoza(src_pts_yx, dst_pts_yx,
                                                                       K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2])

R_cam = axis_angle_to_rotation_matrix(rot_cam[None])
quat_cam = Quaternion.from_matrix(R_cam)

R_obj, t_obj = Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam[None], W_4x4)
t_obj = t_obj[..., 0]  # Shape (1, 3, 1) -> (1, 3)

print('----------------------------------------')
print(f'Rot cam: {torch.rad2deg(rot_cam).numpy(force=True)}')
print(f'T cam  : {t_cam.squeeze().numpy(force=True)}')
print(f'Rot obj: {torch.rad2deg(rot_cam).numpy(force=True)}')
print(f'T obj  : {t_obj.squeeze().numpy(force=True)}')
print('----------------------------------------')

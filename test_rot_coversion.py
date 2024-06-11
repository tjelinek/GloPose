import kaolin
import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle

from auxiliary_scripts.math_utils import Rt_obj_from_epipolar_Rt_cam, Rt_epipolar_cam_from_Rt_obj
from dataset_generators import scenarios
from utils import homogenize_3x4_transformation_matrix

camera_translation = torch.tensor([1.0, -2.0, 5.0]).to(torch.float).unsqueeze(0)
camera_up = torch.tensor([0, 1, 0]).to(torch.float).unsqueeze(0)
obj_center = torch.tensor([0, 0, 0]).to(torch.float).unsqueeze(0)

obj_rot_np = np.deg2rad(np.stack(scenarios.generate_rotations_xyz(5).rotations, axis=0))
obj_rot = torch.from_numpy(obj_rot_np).to(torch.float32)
# obj_rot[:, [0, 2]] *= -1.
obj_trans = torch.zeros_like(obj_rot).squeeze().unsqueeze(-1)
obj_trans[:, [2]] = 1.

W_4x3 = kaolin.render.camera.generate_transformation_matrix(camera_position=camera_translation,
                                                            camera_up_direction=camera_up,
                                                            look_at=obj_center)

T_world_to_cam = homogenize_3x4_transformation_matrix(W_4x3.permute(0, 2, 1)).repeat(obj_rot.shape[0], 1, 1)

obj_R = axis_angle_to_rotation_matrix(obj_rot)

R_cam, t_cam = Rt_epipolar_cam_from_Rt_obj(obj_R, obj_trans, T_world_to_cam)

r_cam = rotation_matrix_to_axis_angle(R_cam)

R_obj_prime, t_obj_prime = Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam, T_world_to_cam)

r_obj_prime = rotation_matrix_to_axis_angle(R_obj_prime)


def print_tensor(name, tensor: torch.Tensor, max_vals=80):
    print(f"{name}:\n{tensor[:max_vals].squeeze()}\n\n")


print_tensor("t_obj", obj_trans)
print_tensor("t_cam", t_cam)
print_tensor("t_obj_prime", t_obj_prime)
print_tensor("r_obj", torch.rad2deg(obj_rot))
print_tensor("r_cam", torch.rad2deg(r_cam))
print_tensor("r_obj_prime", torch.rad2deg(r_obj_prime))

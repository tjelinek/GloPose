import kaolin
import numpy as np
import torch
from kornia.geometry import axis_angle_to_rotation_matrix, rotation_matrix_to_axis_angle, Se3, Quaternion

from auxiliary_scripts.math_utils import Se3_epipolar_cam_from_Se3_obj, Se3_obj_from_epipolar_Se3_cam
from dataset_generators import scenarios
from dataset_generators.track_augmentation import modify_rotations
from utils import homogenize_3x4_transformation_matrix

for camera_up_idx in range(3):
    scenario = scenarios.generate_rotations_xyz(5.0)

    sequence_len = scenario.steps

    scenario_t = scenarios.generate_xyz_translation(sequence_len)

    gt_rotation = torch.from_numpy(scenario.rotation_axis_angles)[:sequence_len].cuda().to(torch.float32)  # * 0
    gt_rotation[..., 0] *= -2.
    gt_rotation[..., 1] *= 1.
    gt_rotation[..., 2] *= -0.5

    # gt_translation = torch.zeros(1, 1, sequence_len, 3).cuda()
    gt_translation = torch.from_numpy(np.asarray(scenario_t.translations)).cuda()[:sequence_len].to(torch.float32) * 0
    gt_translation[..., 0] *= 2.
    gt_translation[..., 1] *= 1.
    gt_translation[..., 2] *= 0.5

    source_frame = 5
    target_frame = 10

    if True:
        gt_rotation, gt_translation = modify_rotations(gt_rotation, gt_translation)
        gt_rotation = gt_rotation[:sequence_len]
        gt_translation = gt_translation[:sequence_len]

    Se3_obj_gt = Se3(Quaternion.from_axis_angle(gt_rotation), gt_translation)

    if True:  # Really rough test
        jump = 3
        matrices = []
        for i, j in zip(range(-jump, sequence_len - jump), range(0, sequence_len)):
            matrices.append((Se3_obj_gt[[min(0, i - jump)]].inverse() * Se3_obj_gt[[j]]).matrix())
        Se3_obj_gt = Se3.from_matrix(torch.cat(matrices, dim=0))

    camera_up = torch.tensor([0, 0, 0]).to(torch.float).unsqueeze(0)
    camera_up[:, camera_up_idx] = 1
    camera_translation = torch.tensor([4.0, -2.0, 5.0]).to(torch.float).unsqueeze(0)
    obj_center = torch.tensor([0, 0, 0]).to(torch.float).unsqueeze(0)

    W_4x3 = kaolin.render.camera.generate_transformation_matrix(camera_position=camera_translation,
                                                                camera_up_direction=camera_up,
                                                                look_at=obj_center).cuda()

    T_world_to_cam = homogenize_3x4_transformation_matrix(W_4x3.permute(0, 2, 1)).repeat(gt_rotation.shape[0], 1, 1)
    Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)

    obj_R = axis_angle_to_rotation_matrix(gt_rotation)

    Se3_cam = Se3_epipolar_cam_from_Se3_obj(Se3_obj_gt, Se3_world_to_cam)
    t_cam = Se3_cam.translation
    r_cam = rotation_matrix_to_axis_angle(Se3_cam.quaternion.matrix())

    Se3_obj_prime = Se3_obj_from_epipolar_Se3_cam(Se3_cam, Se3_world_to_cam)
    t_obj_prime = Se3_obj_prime.translation
    r_obj_prime = rotation_matrix_to_axis_angle(Se3_obj_prime.quaternion.matrix())


    def print_tensor(name, tensor: torch.Tensor, max_vals=80):
        tensor = tensor[:max_vals].squeeze()
        tensor_numpy = tensor.cpu().detach().numpy()
        formatted_tensor = "\n".join(" ".join(f"{val:.3f}" for val in row) for row in tensor_numpy)
        print(f"{name}:\n{formatted_tensor}\n")

    r_obj_print = torch.rad2deg(gt_rotation)
    r_cam_print = torch.rad2deg(r_cam)
    r_obj_prime_print = torch.rad2deg(r_obj_prime)

    error_obj: Se3 = Se3_obj_gt * Se3_obj_prime.inverse()
    Se3_error_cam_obj = Se3_obj_gt * Se3_cam.inverse()
    rot_error_angle = torch.rad2deg(2 * error_obj.quaternion.polar_angle)
    rot_error_cam_obj_angle = torch.rad2deg(2 * Se3_error_cam_obj.quaternion.polar_angle)

    try:
        assert torch.max(torch.abs(gt_translation - t_obj_prime)) < 1e-3
        assert torch.max(torch.abs(rot_error_angle)) < 1e-3
        # assert torch.max(torch.abs(rot_error_cam_obj_angle)) < 1e-3
    except:
        print("camera_ip", camera_up)
        print_tensor("t_obj", gt_translation[:5])
        # print_tensor("t_cam", t_cam)
        print_tensor("t_obj_prime", t_obj_prime[:5])
        print_tensor("r_obj", r_obj_print[:5])
        # print_tensor("r_cam", r_cam_print[:5])
        print_tensor("r_obj_prime", r_obj_prime_print[:5])
        breakpoint()

import numpy as np
import torch
from kornia.geometry import Rt_to_matrix4x4, matrix4x4_to_Rt, normalize_quaternion, Se3, Quaternion


def T_obj_from_epipolar_T_cam(T_cam, T_world_to_cam):
    Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)
    Se3_cam = Se3.from_matrix(T_cam)
    Se3_obj = Se3_obj_from_epipolar_Se3_cam(Se3_cam, Se3_world_to_cam)
    T_obj = Se3_obj.matrix()

    return T_obj


def Se3_obj_from_epipolar_Se3_cam(Se3_cam: Se3, Se3_world_to_cam: Se3) -> Se3:
    return Se3_world_to_cam * Se3_cam * Se3_world_to_cam.inverse()


def Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam, T_world_to_cam):
    T_cam = Rt_to_matrix4x4(R_cam, t_cam)

    T_o2_to_o1 = T_obj_from_epipolar_T_cam(T_cam, T_world_to_cam)

    R, t = matrix4x4_to_Rt(T_o2_to_o1)

    return R, t


def Rt_epipolar_cam_from_Rt_obj(R_obj, t_obj, T_world_to_cam):
    T_obj = Rt_to_matrix4x4(R_obj, t_obj)

    T_cam = T_epipolar_cam_from_T_obj(T_obj, T_world_to_cam)

    R_cam, t_cam = matrix4x4_to_Rt(T_cam)

    return R_cam, t_cam


def T_epipolar_cam_from_T_obj(T_obj, T_world_to_cam):
    Se3_obj = Se3.from_matrix(T_obj)
    Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)

    Se3_cam = Se3_epipolar_cam_from_Se3_obj(Se3_obj, Se3_world_to_cam)
    T_cam = Se3_cam.matrix()

    return T_cam


def Se3_epipolar_cam_from_Se3_obj(Se3_obj: Se3, Se3_world_to_cam: Se3) -> Se3:
    return Se3_world_to_cam.inverse() * Se3_obj * Se3_world_to_cam


def camera_Rt_world_from_Rt_obj(R_obj, t_obj, T_world_to_cam):
    T_obj = Rt_to_matrix4x4(R_obj, t_obj)

    T_cam = camera_T_world_from_T_obj(T_obj, T_world_to_cam)

    R_cam, t_cam = matrix4x4_to_Rt(T_cam)

    return R_cam, t_cam


def camera_T_world_from_T_obj(T_obj, T_world_to_cam):
    Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)
    Se3_obj = Se3.from_matrix(T_obj)
    Se3_cam = camera_Se3_world_from_Se3_obj(Se3_obj, Se3_world_to_cam)

    T_cam = Se3_cam.matrix()

    return T_cam


def camera_Se3_world_from_Se3_obj(Se3_obj: Se3, Se3_world_to_cam: Se3) -> Se3:
    Se3_cam = Se3_obj * Se3_world_to_cam.inverse()

    return Se3_cam


def qmult(q1, q0):  # q0, then q1, you get q3
    w0, x0, y0, z0 = q0[0]
    w1, x1, y1, z1 = q1[0]
    q3 = torch.cat(((-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0)[None, None],
                    (x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0)[None, None],
                    (-x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0)[None, None],
                    (x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0)[None, None]), 1)
    return q3


def qdist(q1, q2):
    return 1 - (q1 * q2).sum() ** 2


def qdifference(q1, q2):  # how to get from q1 to q2
    q1conj = -q1
    q1conj[0, 0] = q1[0, 0]
    q1inv = q1conj / q1.norm()
    diff = qmult(q2, q1inv)
    return diff


def quaternion_angular_difference(quaternions1, quaternions2):
    angles = torch.zeros(quaternions1.shape[1])

    for i in range(angles.shape[0]):
        q_ = qdifference(quaternions1[:, i], quaternions2[:, i])
        diff = normalize_quaternion(q_)
        ang = float(2 * torch.atan2(diff[:, 1:].norm(), diff[:, 0])) * 180 / np.pi
        angles[i] = ang
    return angles


def consecutive_quaternions_angular_difference(quaternion):
    angs = []
    for qi in range(quaternion.shape[1] - 1):
        diff = normalize_quaternion(qdifference(quaternion[:, qi], quaternion[:, qi + 1]))
        angs.append(float(2 * torch.atan2(diff[:, 1:].norm(), diff[:, 0])) * 180 / np.pi)
    return np.array(angs)


def get_object_pose_after_in_plane_rot_in_cam_space(obj_rotation_q: torch.Tensor, T_world_to_cam: torch.Tensor,
                                                    in_plane_rotation_degrees: float):
    obj_rotation_q = Quaternion(obj_rotation_q)
    obj_pose_se3 = Se3(obj_rotation_q, torch.zeros(1, 3).cuda())
    obj_pose_R, obj_pose_t = Rt_epipolar_cam_from_Rt_obj(obj_pose_se3.quaternion.matrix(),
                                                         obj_pose_se3.t[..., None], T_world_to_cam)
    obj_pose_q_cam_space = Quaternion.from_matrix(obj_pose_R)
    obj_pose_se3_cam_space = Se3(obj_pose_q_cam_space, obj_pose_t[..., 0])

    cam_space_in_plane_rot_axis_angle = torch.deg2rad(torch.tensor([in_plane_rotation_degrees]).cuda())
    cam_space_in_plane_rot_se3 = Se3.rot_z(cam_space_in_plane_rot_axis_angle, )
    obj_pose_se3_rotated_cam_space = obj_pose_se3_cam_space * cam_space_in_plane_rot_se3
    R_obj_rotated_world, t_obj_rotated_world = Rt_obj_from_epipolar_Rt_cam(
        obj_pose_se3_rotated_cam_space.quaternion.matrix(),
        obj_pose_se3_rotated_cam_space.t[..., None], T_world_to_cam)
    q_obj_rotated_world = Quaternion.from_matrix(R_obj_rotated_world).data

    return q_obj_rotated_world

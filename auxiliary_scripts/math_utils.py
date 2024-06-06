import numpy as np
import torch
from kornia.geometry import inverse_transformation, Rt_to_matrix4x4, compose_transformations, matrix4x4_to_Rt, \
    normalize_quaternion


def T_obj_from_epipolar_T_cam(T_cam, T_world_to_cam):
    T_cam_to_world = inverse_transformation(T_world_to_cam)

    # T_o1_to_o2 = T_w_to_c0 @ T_cam @ (T_w_to_c0)^-1
    T_obj = compose_transformations(compose_transformations(T_world_to_cam, T_cam), T_cam_to_world)

    return T_obj


def Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam, T_world_to_cam):
    T_cam = Rt_to_matrix4x4(R_cam, t_cam)

    T_o2_to_o1 = T_obj_from_epipolar_T_cam(T_cam, T_world_to_cam)

    R, t = matrix4x4_to_Rt(T_o2_to_o1)

    return R, t


def Rt_epipolar_cam_from_Rt_obj(R_obj, t_obj, T_world_to_cam):
    T_obj = Rt_to_matrix4x4(R_obj, t_obj)
    T_cam_to_world = inverse_transformation(T_world_to_cam)

    T_cam = compose_transformations(compose_transformations(T_cam_to_world, T_obj), T_world_to_cam)

    R_cam, t_cam = matrix4x4_to_Rt(T_cam)

    return R_cam, t_cam


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


def consecutive_quaternions_angular_difference2(quaternion):
    angs = []
    for qi in range(quaternion.shape[1] - 1):
        ang = float(torch.acos(torch.dot(quaternion[0, qi], quaternion[0, qi + 1]) /
                               (quaternion[0, qi].norm() * quaternion[0, qi].norm()))) * 180.0 / np.pi
        angs.append(ang)
    return np.array(angs)

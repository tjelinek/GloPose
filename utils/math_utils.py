import numpy as np
import torch
from kornia.geometry import Rt_to_matrix4x4, matrix4x4_to_Rt, Se3, Quaternion, So3


def T_obj_from_epipolar_T_cam(T_cam, T_world_to_cam):
    Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)
    Se3_cam = Se3.from_matrix(T_cam)
    Se3_obj = Se3_obj_from_epipolar_Se3_cam(Se3_cam, Se3_world_to_cam)
    T_obj = Se3_obj.matrix()

    return T_obj


def Se3_obj_from_epipolar_Se3_cam(Se3_cam: Se3, Se3_world_to_cam: Se3) -> Se3:
    return Se3_world_to_cam.inverse() * Se3_cam * Se3_world_to_cam


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
    return Se3_world_to_cam * Se3_obj * Se3_world_to_cam.inverse()


def Se3_last_cam_to_world_from_Se3_obj(Se3_obj: Se3, Se3_world_to_cam: Se3) -> Se3:
    Se3_cam = (Se3_world_to_cam * Se3_obj).inverse()

    return Se3_cam


def pixel_coords_to_unit_coords(image_width: int, image_height: int, pts_yx: torch.Tensor, dtype=torch.float32)\
        -> torch.Tensor:
    return pts_yx.to(dtype) / torch.Tensor([image_height, image_width]).to(pts_yx.device)


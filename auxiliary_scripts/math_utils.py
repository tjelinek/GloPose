from kornia.geometry import inverse_transformation, Rt_to_matrix4x4, compose_transformations, matrix4x4_to_Rt


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

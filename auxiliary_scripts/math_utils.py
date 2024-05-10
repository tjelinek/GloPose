from kornia.geometry import inverse_transformation, Rt_to_matrix4x4, compose_transformations, matrix4x4_to_Rt, \
    rotation_matrix_to_axis_angle


def Rt_obj_from_epipolar_Rt_cam(R_cam, t_cam, T_world_to_cam):

    T_RANSAC = inverse_transformation(Rt_to_matrix4x4(R_cam, t_cam))

    # T_o2_to_o1 = T_w_to_c0 @ T_RANSAC @ (T_w_to_c0)^-1
    T_o2_to_o1 = compose_transformations(compose_transformations(T_world_to_cam, T_RANSAC),
                                         inverse_transformation(T_world_to_cam))

    R, t = matrix4x4_to_Rt(T_o2_to_o1)

    return R, t


def Rt_epipolar_cam_from_Rt_obj(R_obj, t_obj, T_world_to_cam):
    T_o2_to_o1 = Rt_to_matrix4x4(R_obj, t_obj)
    T_w_to_c_inv = inverse_transformation(T_world_to_cam)
    T_cam = compose_transformations(compose_transformations(T_w_to_c_inv, T_o2_to_o1), T_world_to_cam)

    R_cam, t_cam = matrix4x4_to_Rt(T_cam)

    return R_cam, t_cam
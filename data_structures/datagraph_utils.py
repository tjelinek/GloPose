from kornia.geometry import Quaternion, Se3

from auxiliary_scripts.math_utils import Se3_epipolar_cam_from_Se3_obj
from data_structures.data_graph import DataGraph


def get_gt_obj_pose(frame: int, data_graph: DataGraph) -> Se3:
    frame_data = data_graph.get_frame_data(frame)

    quaternion = Quaternion.from_axis_angle(frame_data.gt_rot_axis_angle[None])
    tran = frame_data.gt_translation[None]

    return Se3(quaternion, tran)


def get_gt_cam_pose(frame: int, Se3_world_to_cam, data_graph: DataGraph) -> Se3:
    Se3_obj_gt = get_gt_obj_pose(frame, data_graph)

    return Se3_epipolar_cam_from_Se3_obj(Se3_obj_gt, Se3_world_to_cam)


def get_relative_gt_obj_rotation(source_frame, target_frame, data_graph: DataGraph) -> Se3:
    Se3_obj_ref_frame_gt = get_gt_obj_pose(source_frame, data_graph)
    Se3_obj_target_frame_gt = get_gt_obj_pose(target_frame, data_graph)

    Se3_obj_chained_long_jump = Se3_obj_target_frame_gt * Se3_obj_ref_frame_gt.inverse()
    return Se3_obj_chained_long_jump


def get_relative_gt_cam_rotation(source_frame, target_frame, Se3_obj_to_cam, data_graph: DataGraph) -> Se3:
    Se3_obj = get_relative_gt_obj_rotation(source_frame, target_frame, data_graph)

    Se3_cam = Se3_epipolar_cam_from_Se3_obj(Se3_obj, Se3_obj_to_cam)

    return Se3_cam

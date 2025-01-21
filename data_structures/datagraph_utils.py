from kornia.geometry import Se3

from utils.math_utils import Se3_epipolar_cam_from_Se3_obj
from data_structures.data_graph import DataGraph


def get_gt_cam_pose(frame: int, Se3_world_to_cam, data_graph: DataGraph) -> Se3:
    Se3_obj_gt = data_graph.get_frame_data(frame).gt_obj1_to_obji

    return Se3_epipolar_cam_from_Se3_obj(Se3_obj_gt, Se3_world_to_cam)


def get_relative_gt_obj_rotation(source_frame, target_frame, data_graph: DataGraph) -> Se3:
    Se3_obj_ref_frame_gt = data_graph.get_frame_data(source_frame).gt_obj1_to_obji
    Se3_obj_target_frame_gt = data_graph.get_frame_data(target_frame).gt_obj1_to_obji

    Se3_obj_chained_long_jump = Se3_obj_target_frame_gt * Se3_obj_ref_frame_gt.inverse()
    return Se3_obj_chained_long_jump


def get_relative_gt_cam_rotation(source_frame, target_frame, Se3_obj_to_cam, data_graph: DataGraph) -> Se3:
    Se3_obj = get_relative_gt_obj_rotation(source_frame, target_frame, data_graph)

    Se3_cam = Se3_epipolar_cam_from_Se3_obj(Se3_obj, Se3_obj_to_cam)

    return Se3_cam

from kornia.geometry import Quaternion, Se3

from data_structures.data_graph import DataGraph


def get_relative_gt_rotation(flow_long_jump_source, flow_long_jump_target, data_graph: DataGraph):
    ref_data = data_graph.get_frame_data(flow_long_jump_source)
    target_data = data_graph.get_frame_data(flow_long_jump_target)

    ref_rot = Quaternion.from_axis_angle(ref_data.gt_rot_axis_angle[None])
    ref_trans = ref_data.gt_translation[None]
    target_rot = Quaternion.from_axis_angle(target_data.gt_rot_axis_angle[None])
    target_trans = target_data.gt_translation[None]

    Se3_obj_ref_frame_gt = Se3(ref_rot, ref_trans)
    Se3_obj_target_frame_gt = Se3(target_rot, target_trans)

    Se3_obj_chained_long_jump = Se3_obj_target_frame_gt * Se3_obj_ref_frame_gt.inverse()
    return Se3_obj_chained_long_jump

from typing import List, Tuple, Callable, Optional

from dataclasses import dataclass

import torch
from kornia.geometry import Quaternion, Se3

from auxiliary_scripts.math_utils import quaternion_minimal_angular_difference
from data_structures.keyframe_buffer import FrameObservation


@dataclass
class IcosphereNode:
    quaternion: Quaternion
    translation: torch.Tensor
    observation: FrameObservation
    keyframe_idx_observed: int


class PoseIcosphere:

    def __init__(self):
        self.reference_poses: List[IcosphereNode] = []
        self.glomap_callback: Optional[Callable] = None

    def insert_new_reference(self, template_observation: FrameObservation, pose: Se3, keyframe_idx_observed: int):
        pose_quat_shape = pose.quaternion.q.shape
        assert len(pose_quat_shape) == 2 and pose_quat_shape[1] == 4

        node = IcosphereNode(quaternion=pose.quaternion, translation=pose.translation,
                             observation=template_observation, keyframe_idx_observed=keyframe_idx_observed)

        self.reference_poses.append(node)

        if self.glomap_callback is None:
            raise ValueError('\'glomap_callback\' needs to be initialized.')
        self.glomap_callback(node)

    def get_closest_reference(self, pose_quaternion: Quaternion) -> Tuple[IcosphereNode, float]:
        # Gives the closest reference template image, and angular difference to the closest template image

        min_angle = float('inf')
        min_index = -1

        for i, template in enumerate(self.reference_poses):
            angle_between_poses = float(quaternion_minimal_angular_difference(pose_quaternion, template.quaternion))
            if angle_between_poses < min_angle:
                min_angle = angle_between_poses
                min_index = i

        closest_template = self.reference_poses[min_index]
        return closest_template, min_angle

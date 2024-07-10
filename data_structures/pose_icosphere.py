from typing import List, Tuple

import torch
from dataclasses import dataclass

from auxiliary_scripts.math_utils import quaternion_angular_difference
from data_structures.keyframe_buffer import FrameObservation


@dataclass
class IcosphereNode:
    quaternion: torch.Tensor
    observation: FrameObservation
    keyframe_idx_observed: int


class PoseIcosphere:

    def __init__(self):
        self.reference_poses: List[IcosphereNode] = []

    def insert_new_reference(self, template_observation: FrameObservation, pose_quaternion: torch.Tensor,
                             keyframe_idx_observed: int):
        pose_quat_shape = pose_quaternion.shape
        assert len(pose_quat_shape) == 2 and pose_quat_shape[1] == 4

        node = IcosphereNode(quaternion=pose_quaternion, observation=template_observation,
                             keyframe_idx_observed=keyframe_idx_observed)

        self.reference_poses.append(node)

    def get_closest_reference(self, pose_quaternion) -> Tuple[IcosphereNode, float]:
        # Gives the closest reference template image, and angular difference to the closest template image

        min_angle = float('inf')
        min_index = -1

        for i, template in enumerate(self.reference_poses):
            angle_between_poses = float(quaternion_angular_difference(pose_quaternion, template.quaternion[None]))

            if angle_between_poses < min_angle:
                min_angle = angle_between_poses
                min_index = i

        closest_template = self.reference_poses[min_index]
        return closest_template, min_angle

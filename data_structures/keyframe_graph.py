from typing import List, Tuple

from dataclasses import dataclass

from kornia.geometry import Quaternion

from utils.math_utils import quaternion_minimal_angular_difference
from data_structures.keyframe_buffer import FrameObservation


@dataclass
class IcosphereNode:
    observation: FrameObservation
    keyframe_idx_observed: int


class KeyframeGraph:

    def __init__(self):
        self.reference_poses: List[IcosphereNode] = []

    def insert_new_reference(self, template_observation: FrameObservation, keyframe_idx_observed: int):

        node = IcosphereNode(observation=template_observation, keyframe_idx_observed=keyframe_idx_observed)

        self.reference_poses.append(node)

    def contains_node(self, frame_idx):

        node_indices = [node.keyframe_idx_observed for node in self.reference_poses]
        return frame_idx in node_indices

    def get_keyframe_indices(self) -> List[int]:
        return [node.keyframe_idx_observed for node in self.reference_poses]

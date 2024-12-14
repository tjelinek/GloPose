from enum import Enum
from time import time
from typing import Callable, List, Tuple

import torch
from kornia.geometry import PinholeCamera

from data_providers.flow_provider import FlowProvider
from data_structures.data_graph import DataGraph, CommonFrameData
from data_structures.pose_icosphere import PoseIcosphere
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig


class FrameFilterAlgorithms(Enum):
    BETWEEN_CURRENT_AND_LAST_KF = "Match frames in (last kf, current) once lost"
    ALL_KFS = "Match every kf once lost"

    @classmethod
    def from_value(cls, value):
        return next((algorithm for algorithm in cls if algorithm.value == value), None)


class FrameFilter:

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, pose_icosphere, camera: PinholeCamera,
                 flow_provider: FlowProvider):

        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.pose_icosphere: PoseIcosphere = pose_icosphere

        self.image_width: int = int(camera.width.item())
        self.image_height: int = int(camera.height.item())

        self.flow_provider: FlowProvider = flow_provider

        when_lost_algorithms = {
            FrameFilterAlgorithms.ALL_KFS: self.matching_to_all_kfs_getting_lost_procedure,
            FrameFilterAlgorithms.BETWEEN_CURRENT_AND_LAST_KF: self.matching_to_last_to_newest_getting_lost_procedure
        }

        when_lost_algorithm_enum_val = FrameFilterAlgorithms.from_value(self.config.frame_filter_when_lost_algorithm)
        self.selected_when_lost_algorithm: Callable = when_lost_algorithms[when_lost_algorithm_enum_val]

    @torch.no_grad()
    def filter_frames(self, frame_i: int):

        start_time = time()

        preceding_frame_node = self.data_graph.get_frame_data(frame_i - 1)
        preceding_source = preceding_frame_node.long_jump_source
        edge_data = self.data_graph.get_edge_observations(preceding_source, preceding_frame_node)
        reliable_flows = set()
        if edge_data.is_match_reliable and frame_i > 1:
            source = preceding_source
        elif frame_i > 1:
            reliable_flows, source = self.selected_when_lost_algorithm(frame_i, reliable_flows, preceding_source)
        else:
            source = 0

        flow_arc_long_jump = (source, frame_i)

        self.add_new_flow(source, frame_i)

        long_jump_source, long_jump_target = flow_arc_long_jump

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(frame_i)
        datagraph_node.pose_estimation_time = duration

        datagraph_long_edge = self.data_graph.get_edge_observations(*flow_arc_long_jump)

        flow_reliability = self.flow_reliability(long_jump_source, long_jump_target)
        datagraph_long_edge.reliability_score = flow_reliability
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{flow_reliability}')
        new_node_frame_idx = max(long_jump_source + 1, frame_i - 1)
        nodes_in_icosphere = {pose.keyframe_idx_observed for pose in self.pose_icosphere.reference_poses}

        if flow_reliability < self.config.flow_reliability_threshold and new_node_frame_idx not in nodes_in_icosphere:
            datagraph_node.is_source_reliable = False
            cam_frame_data = self.data_graph.get_frame_data(new_node_frame_idx)

            self.pose_icosphere.insert_new_reference(cam_frame_data.frame_observation, new_node_frame_idx)

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_flows)
        datagraph_node.long_jump_source = source

    def matching_to_all_kfs_getting_lost_procedure(self, frame_i, reliable_flows, preceding_source):
        best_source: int = 0
        best_source_reliability: float = 0.
        for node in self.pose_icosphere.reference_poses:
            source_node_idx = node.keyframe_idx_observed

            self.add_new_flow(source_node_idx, frame_i)
            flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
            flow_reliability = self.flow_reliability(source_node_idx, frame_i)
            flow_edge_data.reliability_score = flow_reliability

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source
        return reliable_flows, source

    def matching_to_last_to_newest_getting_lost_procedure(self, frame_i, reliable_flows, preceding_source):
        best_source: int = 0
        best_source_reliability: float = 0.

        nodes: List[Tuple[int, CommonFrameData]] = [(i, self.data_graph.get_frame_data(i)) for i in range(preceding_source, frame_i)]

        for source_node_idx, node in nodes:

            self.add_new_flow(source_node_idx, frame_i)
            flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
            flow_reliability = self.flow_reliability(source_node_idx, frame_i)
            flow_edge_data.reliability_score = flow_reliability

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source
        return reliable_flows, source

    def flow_reliability(self, source_idx: int, target_idx: int) -> float:

        source_datagraph_node = self.data_graph.get_frame_data(source_idx)
        fg_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze()
        flow_arc_node = self.data_graph.get_edge_observations(source_idx, target_idx)

        H_A, W_A = self.image_height, self.image_width
        src_pts_yx, dst_pts_yx = roma_warp_to_pixel_coordinates(flow_arc_node.flow_warp, H_A, W_A, H_A, W_A)

        src_pts_yx_int = src_pts_yx.int()
        in_segmentation_mask = fg_segmentation_mask[src_pts_yx_int[:, 0], src_pts_yx_int[:, 1]].bool()
        fg_certainties = flow_arc_node.flow_certainty[in_segmentation_mask]
        fg_certainties_above_threshold = fg_certainties > self.config.flow_reliability_threshold

        reliability = fg_certainties_above_threshold.sum() / (fg_certainties.numel() + 1e-5)

        return reliability.item()

    def add_new_flow(self, source_frame, target_frame):
        if (source_frame, target_frame) not in self.data_graph.G.edges:

            self.data_graph.add_new_arc(source_frame, target_frame)
            self.flow_provider.add_flows_into_datagraph(source_frame, target_frame)

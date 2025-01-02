from time import time
from typing import List, Tuple

import torch
from kornia.image import ImageSize

from data_providers.flow_provider import RoMaFlowProviderDirect
from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.data_graph import DataGraph, CommonFrameData
from data_structures.pose_icosphere import PoseIcosphere
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig


class BaseFrameFilter:
    def __init__(self, config: TrackerConfig, data_graph: DataGraph, pose_icosphere, image_shape: ImageSize):
        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.pose_icosphere: PoseIcosphere = pose_icosphere

        self.image_width: int = int(image_shape.width)
        self.image_height: int = int(image_shape.height)


class FrameFilter(BaseFrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, pose_icosphere, image_shape: ImageSize,
                 flow_provider: RoMaFlowProviderDirect):

        super().__init__(config, data_graph, pose_icosphere, image_shape)

        self.flow_provider: RoMaFlowProviderDirect = flow_provider

    @torch.no_grad()
    def filter_frames(self, frame_i: int):

        start_time = time()

        preceding_frame_idx = frame_i - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)
        preceding_source = preceding_frame_node.long_jump_source
        self.add_new_flow(preceding_source, preceding_frame_idx)

        # for preceding_frame in range(frame_i):
        #     self.add_new_flow(preceding_frame, frame_i)

        edge_data = self.data_graph.get_edge_observations(preceding_source, preceding_frame_idx)
        if edge_data.is_match_reliable and frame_i > 1:
            source = preceding_source
            reliable_flows = {source}
        elif frame_i > 1:
            reliable_flows, best_source = self.match_to_all_keyframes(frame_i)
            if best_source is None:
                reliable_flows, best_source = self.match_to_frames_from_last_kf(frame_i, preceding_source)

                if best_source is None:
                    source = preceding_source
                else:
                    source = best_source
                    cam_frame_data = self.data_graph.get_frame_data(best_source)

                    self.pose_icosphere.insert_new_reference(cam_frame_data.frame_observation, best_source)
            else:
                source = best_source
        else:
            source = 0
            reliable_flows = set()
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

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_flows)
        datagraph_node.long_jump_source = source

    def match_to_all_keyframes(self, frame_i):
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_flows = set()

        for node in self.pose_icosphere.reference_poses:
            source_node_idx = node.keyframe_idx_observed

            self.add_new_flow(source_node_idx, frame_i)
            flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
            flow_reliability = flow_edge_data.reliability_score

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source

        if best_source_reliability < self.config.flow_reliability_threshold:
            return None, None
        return reliable_flows, source

    def match_to_frames_from_last_kf(self, frame_i, preceding_source):
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_flows = set()

        nodes: List[Tuple[int, CommonFrameData]] = [(i, self.data_graph.get_frame_data(i)) for i in
                                                    range(preceding_source, frame_i)]

        for source_node_idx, node in nodes:
            self.add_new_flow(source_node_idx, frame_i)
            flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
            flow_reliability = flow_edge_data.reliability_score

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source

        if best_source_reliability < self.config.flow_reliability_threshold:
            return None, None
        return reliable_flows, source

    def flow_reliability(self, source_idx: int, target_idx: int) -> float:

        dev = self.config.device
        source_datagraph_node = self.data_graph.get_frame_data(source_idx)
        fg_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze().to(dev)
        flow_arc_node = self.data_graph.get_edge_observations(source_idx, target_idx)

        H_A, W_A = self.image_height, self.image_width
        src_pts_xy, dst_pts_xy = roma_warp_to_pixel_coordinates(flow_arc_node.flow_warp, H_A, W_A, H_A, W_A)

        src_pts_xy_int = src_pts_xy.int()
        in_segmentation_mask = fg_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
        fg_certainties = flow_arc_node.flow_certainty.to(dev)[in_segmentation_mask]
        fg_certainties_above_threshold = fg_certainties > self.config.min_roma_certainty_threshold

        reliability = fg_certainties_above_threshold.sum() / (fg_certainties.numel() + 1e-5)

        sufficient_reliable_matches = (fg_certainties_above_threshold.numel() >
                                       self.config.min_number_of_reliable_matches)

        reliability *= float(sufficient_reliable_matches)

        return reliability.item()

    def add_new_flow(self, source_frame, target_frame):
        if (source_frame, target_frame) not in self.data_graph.G.edges:
            self.data_graph.add_new_arc(source_frame, target_frame)
        self.flow_provider.add_flows_into_datagraph(source_frame, target_frame)

        reliability = self.flow_reliability(source_frame, target_frame)
        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)
        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability > self.config.flow_reliability_threshold


class FrameFilterSift(BaseFrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, pose_icosphere, image_shape: ImageSize,
                 sift_matcher: SIFTMatchingProvider):

        super().__init__(config, data_graph, pose_icosphere, image_shape)

        self.sift_matcher: SIFTMatchingProvider = sift_matcher

    @torch.no_grad()
    def filter_frames(self, current_frame_idx: int):

        start_time = time()

        preceding_frame_idx = current_frame_idx - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)

        keyframe_idx = preceding_frame_node.long_jump_source

        print("Detection features")

        reliable_sources = set()

        selected_keyframe_idxs = self.pose_icosphere.get_keyframe_indices()

        more_than_enough_matches = self.config.sift_filter_good_to_add_matches
        min_matches = self.config.sift_filter_min_matches

        reliable_keyframe_found = False
        we_stepped_back = False

        while not reliable_keyframe_found:

            num_matches = self.compute_sift_reliability(keyframe_idx, current_frame_idx)

            if num_matches >= more_than_enough_matches:
                if we_stepped_back:
                    print(f"Step back was good, adding keyframe_idx={keyframe_idx}")
                    selected_keyframe_idxs.append(keyframe_idx)

                    if not self.pose_icosphere.contains_node(keyframe_idx):
                        keyframe_observation = self.data_graph.get_frame_data(keyframe_idx).frame_observation
                        self.pose_icosphere.insert_new_reference(keyframe_observation, keyframe_idx)
                    we_stepped_back = False

                reliable_keyframe_found = True

            if (num_matches <= more_than_enough_matches) and (num_matches >= min_matches):
                print("Adding keyframe")

                frame1_observation = self.data_graph.get_frame_data(keyframe_idx).frame_observation
                frame2_observation = self.data_graph.get_frame_data(current_frame_idx).frame_observation

                if not self.pose_icosphere.contains_node(keyframe_idx):
                    self.pose_icosphere.insert_new_reference(frame1_observation, keyframe_idx)
                if not self.pose_icosphere.contains_node(current_frame_idx):
                    self.pose_icosphere.insert_new_reference(frame2_observation, current_frame_idx)

                reliable_keyframe_found = True

            if num_matches < min_matches:  # try going back
                print("Too few matches, going back")
                keyframe_idx = max(0, current_frame_idx - 1)
                we_stepped_back = True
                if (keyframe_idx <= 0):
                    reliable_keyframe_found = True
                elif (keyframe_idx in selected_keyframe_idxs):
                    print(f"We cannot match {current_frame_idx}, skipping it")
                    return

        flow_frames_idxs = (keyframe_idx, current_frame_idx)

        long_jump_source, long_jump_target = flow_frames_idxs

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(current_frame_idx)
        datagraph_node.pose_estimation_time = duration

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_sources)
        datagraph_node.long_jump_source = keyframe_idx

    def compute_sift_reliability(self, frame_idx1: int, frame_idx2: int):

        device = self.config.device

        dists, idxs = self.sift_matcher.match_images_sift(frame_idx1, frame_idx2, device)

        num_matches = len(idxs)

        if not self.data_graph.G.has_edge(frame_idx1, frame_idx2):
            self.data_graph.add_new_arc(frame_idx1, frame_idx2)
        edge_data = self.data_graph.get_edge_observations(frame_idx1, frame_idx2)

        edge_data.num_matches = num_matches
        edge_data.is_match_reliable = num_matches >= self.config.sift_filter_min_matches

        return num_matches

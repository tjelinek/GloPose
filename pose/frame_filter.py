from time import time
from typing import List, Tuple

import networkx as nx
import torch
from kornia.image import ImageSize

from data_providers.flow_provider import RoMaFlowProviderDirect
from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.data_graph import DataGraph, CommonFrameData
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig


class BaseFrameFilter:
    def __init__(self, config: TrackerConfig, data_graph: DataGraph):
        self.config: TrackerConfig = config
        self.data_graph: DataGraph = data_graph
        self.keyframe_graph: nx.DiGraph = nx.DiGraph()

        self.n_frames = self.config.input_frames

    def get_keyframe_graph(self) -> nx.DiGraph:
        if len(self.keyframe_graph.nodes) <= 2:
            nodes_list = sorted(list(self.keyframe_graph.nodes))
            middle_node = (nodes_list[-1] - nodes_list[0]) // 2
            assert nodes_list[0] < middle_node < nodes_list[-1]

            self.keyframe_graph.add_edge(nodes_list[0], middle_node)
            self.keyframe_graph.add_edge(middle_node, nodes_list[-1])

        return self.keyframe_graph


class RoMaFrameFilter(BaseFrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, flow_provider: RoMaFlowProviderDirect):

        super().__init__(config, data_graph)

        self.flow_provider: RoMaFlowProviderDirect = flow_provider

    def filter_frames_new(self, current_frame_idx: int):

        start_time = time()

        if current_frame_idx == 0 or current_frame_idx >= self.n_frames - 1:
            self.keyframe_graph.add_node(current_frame_idx)
            return

        preceding_frame_idx = current_frame_idx - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)

        keyframe_idx = preceding_frame_node.matching_source_keyframe

        print("Detection features")

        reliable_sources = set()

        selected_keyframe_idxs = list(self.keyframe_graph.nodes())

        good_reliability = self.config.flow_reliability_threshold
        min_reliability = 0.5 * self.config.flow_reliability_threshold

        reliable_keyframe_found = False
        we_stepped_back = False

        while not reliable_keyframe_found:

            reliability = self.flow_reliability(keyframe_idx, current_frame_idx)

            if reliability >= min_reliability:
                self.keyframe_graph.add_edge(current_frame_idx, keyframe_idx)

            if reliability >= good_reliability:
                if we_stepped_back:
                    print(f"Step back was good, adding keyframe_idx={keyframe_idx}")
                    selected_keyframe_idxs.append(keyframe_idx)

                    if not self.keyframe_graph.has_node(keyframe_idx):

                        self.keyframe_graph.add_node(keyframe_idx)
                    we_stepped_back = False

                reliable_keyframe_found = True

            if (reliability <= good_reliability) and (reliability >= min_reliability):
                print("Adding keyframe")

                if not self.keyframe_graph.has_node(keyframe_idx):
                    self.keyframe_graph.add_node(keyframe_idx)
                if not self.keyframe_graph.has_node(current_frame_idx):
                    self.keyframe_graph.add_node(current_frame_idx)

                reliable_keyframe_found = True

            if reliability < min_reliability:  # try going back
                print("Too few matches, going back")
                keyframe_idx = max(0, current_frame_idx - 1)
                we_stepped_back = True
                if keyframe_idx <= 0:
                    reliable_keyframe_found = True
                elif keyframe_idx in selected_keyframe_idxs:
                    print(f"We cannot match {current_frame_idx}, skipping it")
                    return

        flow_frames_idxs = (keyframe_idx, current_frame_idx)

        long_jump_source, long_jump_target = flow_frames_idxs

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(current_frame_idx)
        datagraph_node.pose_estimation_time = duration

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_sources)
        datagraph_node.matching_source_keyframe = keyframe_idx

    @torch.no_grad()
    def filter_frames(self, frame_i: int):

        start_time = time()

        if frame_i == 0:
            self.keyframe_graph.add_node(0)
            self.data_graph.get_frame_data(0).reliable_sources = {0}
            self.data_graph.get_frame_data(0).matching_source_keyframe = 0
            return
        if frame_i >= self.n_frames - 1:
            self.keyframe_graph.add_edge(sorted(list(self.keyframe_graph.nodes))[-1], self.n_frames - 1)

        preceding_frame_idx = frame_i - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)
        preceding_source = preceding_frame_node.matching_source_keyframe
        self.add_new_flow(preceding_source, preceding_frame_idx)

        # for preceding_frame in range(frame_i):
        #     self.add_new_flow(preceding_frame, frame_i)

        reliable_flows_sources = set()
        edge_data = self.data_graph.get_edge_observations(preceding_source, preceding_frame_idx)
        if edge_data.is_match_reliable and frame_i > 1:
            source = preceding_source
            reliable_flows_sources |= {source}
        elif frame_i > 1:
            reliable_flows_sources_prime, best_source = self.match_to_all_keyframes(frame_i)
            if best_source is None:
                reliable_flows_sources_prime, best_source = self.match_to_frames_from_last_kf(frame_i, preceding_source)

                if best_source is None:
                    source = preceding_source
                else:
                    source = best_source
                    max_kf = max(self.keyframe_graph.nodes())
                    self.keyframe_graph.add_edge(source, frame_i)
                    self.keyframe_graph.add_edge(max_kf, source)

                    self.keyframe_graph.add_node(best_source)
                reliable_flows_sources |= reliable_flows_sources_prime
            else:
                source = best_source
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

        datagraph_node.reliable_sources = ({long_jump_source} | reliable_flows_sources)

        datagraph_node.matching_source_keyframe = source

    def match_to_all_keyframes(self, frame_i):
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_flows = set()

        current_keyframe_graph_nodes = list(self.keyframe_graph.nodes)
        for source_node_idx in current_keyframe_graph_nodes:

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
            return set(), None
        return reliable_flows, source

    def flow_reliability(self, source_idx: int, target_idx: int) -> float:

        dev = self.config.device
        source_datagraph_node = self.data_graph.get_frame_data(source_idx)
        fg_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze().to(dev)

        if not self.data_graph.G.has_edge(source_idx, target_idx):
            self.add_new_flow(source_idx, target_idx)
        flow_arc_node = self.data_graph.get_edge_observations(source_idx, target_idx)

        H_A, W_A = source_datagraph_node.image_shape.height, source_datagraph_node.image_shape.width
        src_pts_xy, dst_pts_xy = roma_warp_to_pixel_coordinates(flow_arc_node.roma_flow_warp, H_A, W_A, H_A, W_A)

        src_pts_xy_int = torch.ceil(src_pts_xy).int() - 1

        assert ((src_pts_xy_int[:, 0] >= 0) & (src_pts_xy_int[:, 0] < W_A)).all()
        assert ((src_pts_xy_int[:, 1] >= 0) & (src_pts_xy_int[:, 1] < H_A)).all()
        assert fg_segmentation_mask.shape[-2:] == (H_A, W_A)

        in_segmentation_mask_yx = fg_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()

        assert flow_arc_node.roma_flow_certainty.shape == in_segmentation_mask_yx.shape
        fg_certainties = flow_arc_node.roma_flow_certainty.to(dev)[in_segmentation_mask_yx]
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
        edge_data.is_match_reliable = reliability >= self.config.flow_reliability_threshold


class FrameFilterSift(BaseFrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, sift_matcher: SIFTMatchingProvider):

        super().__init__(config, data_graph)

        self.sift_matcher: SIFTMatchingProvider = sift_matcher

    @torch.no_grad()
    def filter_frames(self, current_frame_idx: int):

        start_time = time()

        if current_frame_idx == 0:
            self.keyframe_graph.add_node(current_frame_idx)
            return

        preceding_frame_idx = current_frame_idx - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)

        if current_frame_idx == 1:
            keyframe_idx = 0
        else:
            keyframe_idx = preceding_frame_node.matching_source_keyframe

        print("Detection features")

        reliable_sources = set()

        selected_keyframe_idxs = list(self.keyframe_graph.nodes())

        more_than_enough_matches = self.config.sift_filter_good_to_add_matches
        min_matches = self.config.sift_filter_min_matches

        reliable_keyframe_found = False
        we_stepped_back = False

        while not reliable_keyframe_found:

            num_matches = self.compute_sift_reliability(keyframe_idx, current_frame_idx)
            print(f'{num_matches}, {min_matches}, {more_than_enough_matches}')

            if num_matches >= self.config.sift_filter_min_matches:
                self.keyframe_graph.add_edge(current_frame_idx, keyframe_idx)

            if num_matches >= more_than_enough_matches:
                print(f'{keyframe_idx} has more than enough matches')
                if we_stepped_back:
                    print(f"Step back was good, adding keyframe_idx={keyframe_idx}")
                    selected_keyframe_idxs.append(keyframe_idx)

                    if not self.keyframe_graph.has_node(keyframe_idx):

                        self.keyframe_graph.add_node(keyframe_idx)
                    we_stepped_back = False

                reliable_keyframe_found = True

            if (num_matches <= more_than_enough_matches) and (num_matches >= min_matches):
                print("Adding keyframe")

                if not self.keyframe_graph.has_node(keyframe_idx):
                    self.keyframe_graph.add_node(keyframe_idx)
                if not self.keyframe_graph.has_node(current_frame_idx):
                    self.keyframe_graph.add_node(current_frame_idx)

                reliable_keyframe_found = True

            if num_matches < min_matches:  # try going back
                print("Too few matches, going back")
                keyframe_idx = max(0, keyframe_idx - 1)
                we_stepped_back = True
                if keyframe_idx <= 0:
                    reliable_keyframe_found = True
                elif keyframe_idx in selected_keyframe_idxs:
                    print(f"We cannot match {current_frame_idx}, skipping it")
                    return

        flow_frames_idxs = (keyframe_idx, current_frame_idx)

        long_jump_source, long_jump_target = flow_frames_idxs

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(current_frame_idx)
        datagraph_node.pose_estimation_time = duration

        datagraph_node.reliable_sources |= ({long_jump_source} | reliable_sources)
        datagraph_node.matching_source_keyframe = keyframe_idx

    def compute_sift_reliability(self, frame_idx1: int, frame_idx2: int):

        device = self.config.device

        dists, idxs = self.sift_matcher.match_images_sift(frame_idx1, frame_idx2, device, save_to_datagraph=True)

        num_matches = len(idxs)

        if not self.data_graph.G.has_edge(frame_idx1, frame_idx2):
            self.data_graph.add_new_arc(frame_idx1, frame_idx2)
        edge_data = self.data_graph.get_edge_observations(frame_idx1, frame_idx2)

        edge_data.num_matches = num_matches
        edge_data.is_match_reliable = num_matches >= self.config.sift_filter_min_matches

        return num_matches

from abc import abstractmethod
from time import time
from typing import List, Tuple

import networkx as nx
import numpy as np
import pycolmap
import torch

from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect
from data_providers.matching_provider_sift import SIFTMatchingProvider
from data_structures.data_graph import DataGraph, CommonFrameData
from tracker_config import TrackerConfig
from utils.general import colmap_K_params_vec
from utils.image_utils import otsu_threshold


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
            if nodes_list[0] < middle_node < nodes_list[-1]:
                self.keyframe_graph.add_edge(nodes_list[0], middle_node)
                self.keyframe_graph.add_edge(middle_node, nodes_list[-1])

        return self.keyframe_graph

    @abstractmethod
    def add_keyframe(self, frame_i: int):
        pass

    @abstractmethod
    def filter_frames(self, frame_i: int):
        pass


class RoMaFrameFilter(BaseFrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, flow_provider: PrecomputedRoMaFlowProviderDirect):

        super().__init__(config, data_graph)

        self.flow_provider: PrecomputedRoMaFlowProviderDirect = flow_provider

        self.current_flow_reliability_threshold = self.config.flow_reliability_threshold

    def update_flow_reliability_threshold(self):
        all_frames = self.data_graph.G.nodes
        template_reliabilities = []

        for node in all_frames:
            template_idx = self.data_graph.get_frame_data(node).matching_source_keyframe
            edge = (template_idx, node)
            if self.data_graph.G.has_edge(*edge):
                template_reliabilities.append(self.data_graph.get_edge_observations(*edge).reliability_score)

        template_reliabilities_np = np.array(template_reliabilities)
        # threshold = threshold_otsu(template_reliabilities_np)
        threshold = otsu_threshold(torch.from_numpy(template_reliabilities_np))

        self.current_flow_reliability_threshold = threshold

    def add_keyframe(self, frame_i: int):
        self.keyframe_graph.add_node(frame_i)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(frame_i, frame_i,
                                                                  self.config.roma_sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        kf_data = self.data_graph.get_frame_data(frame_i)
        certainty_threshold = otsu_threshold(certainty)
        if certainty_threshold is None and frame_i > 0:
            prev_kf = kf_data.matching_source_keyframe
            certainty_threshold = self.data_graph.get_frame_data(prev_kf).roma_certainty_threshold
        else:
            certainty_threshold = self.config.min_roma_certainty_threshold
        kf_data.roma_certainty_threshold = certainty_threshold

        if self.config.matchability_based_reliability:
            image_shape = self.data_graph.get_frame_data(frame_i).image_shape
            img_h, img_w = image_shape.height, image_shape.width
            arc_data = self.data_graph.get_edge_observations(frame_i, frame_i)
            roma_shape = arc_data.roma_flow_warp_certainty.shape
            certainty_map = arc_data.roma_flow_warp_certainty[:, :roma_shape[1] // 2]
            certainty_map_img_size = torch.nn.functional.interpolate(certainty_map[None, None], (img_h, img_w),
                                                                     mode='bilinear').squeeze()
            matchability_map = certainty_map_img_size > certainty_threshold
            kf_data.matchability_mask = matchability_map
        kf_data.is_keyframe = True
        print(frame_i)

    @torch.no_grad()
    def filter_frames(self, frame_i: int):

        start_time = time()

        if len(self.keyframe_graph.nodes) >= 10 and False:
            self.update_flow_reliability_threshold()

        if frame_i == 0:
            self.add_keyframe(0)
            first_frame_node = self.data_graph.get_frame_data(0)
            first_frame_node.reliable_sources = {0}
            first_frame_node.matching_source_keyframe = 0
            first_frame_node.current_flow_reliability_threshold = self.current_flow_reliability_threshold
            return
        if frame_i >= self.n_frames - 1 and False:
            self.keyframe_graph.add_edge(sorted(list(self.keyframe_graph.nodes))[-1], self.n_frames - 1)

        preceding_frame_idx = frame_i - 1
        preceding_frame_node = self.data_graph.get_frame_data(preceding_frame_idx)
        preceding_source = preceding_frame_node.matching_source_keyframe
        self.flow_reliability(preceding_source, preceding_frame_idx)

        reliable_flows_sources = set()
        edge_data = self.data_graph.get_edge_observations(preceding_source, preceding_frame_idx)
        if edge_data.is_match_reliable and frame_i > 1:
            source = preceding_source
            reliable_flows_sources |= {source}
        elif frame_i > 1:
            reliable_flows_sources_prime, best_source = self.match_to_all_keyframes(frame_i)
            if best_source is None:
                new_source = frame_i - 1
                self.add_keyframe(new_source)
                self.keyframe_graph.add_edge(preceding_source, new_source)
                source = new_source

                reliable_flows_sources |= {source}
            else:
                source = best_source
                if reliable_flows_sources_prime is not None:
                    reliable_flows_sources |= reliable_flows_sources_prime
        else:
            source = 0
        flow_arc_long_jump = (source, frame_i)
        long_jump_source, long_jump_target = flow_arc_long_jump

        duration = time() - start_time
        datagraph_node = self.data_graph.get_frame_data(frame_i)
        datagraph_node.pose_estimation_time = duration
        datagraph_node.current_flow_reliability_threshold = self.current_flow_reliability_threshold

        flow_reliability = self.flow_reliability(long_jump_source, long_jump_target)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{flow_reliability}')

        datagraph_node.reliable_sources = ({long_jump_source} | reliable_flows_sources)

        datagraph_node.matching_source_keyframe = source

    def match_to_all_keyframes(self, frame_i):
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_flows = set()

        current_keyframe_graph_nodes = list(self.keyframe_graph.nodes)
        for source_node_idx in current_keyframe_graph_nodes:

            flow_reliability = self.flow_reliability(source_node_idx, frame_i)

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source

        if best_source_reliability < self.current_flow_reliability_threshold:
            return None, None
        return reliable_flows, source

    def match_to_frames_from_last_kf(self, frame_i, preceding_source):
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_flows = set()

        nodes: List[Tuple[int, CommonFrameData]] = [(i, self.data_graph.get_frame_data(i)) for i in
                                                    range(preceding_source, frame_i)]

        for source_node_idx, node in nodes:
            self.flow_reliability(source_node_idx, frame_i)
            flow_edge_data = self.data_graph.get_edge_observations(source_node_idx, frame_i)
            flow_reliability = flow_edge_data.reliability_score

            if flow_reliability > best_source_reliability:
                best_source = source_node_idx
                best_source_reliability = flow_reliability
                reliable_flows |= {source_node_idx}
        source = best_source

        if best_source_reliability < self.current_flow_reliability_threshold:
            return set(), None
        return reliable_flows, source

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        dev = self.config.device
        source_datagraph_node = self.data_graph.get_frame_data(source_frame)
        source_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze().to(dev)

        H_A, W_A = source_datagraph_node.image_shape.height, source_datagraph_node.image_shape.width
        assert source_segmentation_mask.shape[-2:] == (H_A, W_A)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.config.roma_sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        assert ((src_pts_xy_int[:, 0] >= 0) & (src_pts_xy_int[:, 0] < W_A)).all()
        assert ((src_pts_xy_int[:, 1] >= 0) & (src_pts_xy_int[:, 1] < H_A)).all()
        assert certainty.shape[0] == src_pts_xy_int.shape[0] and certainty.shape[0] == dst_pts_xy_int.shape[0]

        if self.config.matchability_based_reliability:
            matchability_mask = source_datagraph_node.matchability_mask
            source_segmentation_mask = source_segmentation_mask * matchability_mask
            matchable_fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()

            fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
            in_segmentation_items = float(fg_matches_mask.sum())
            relative_area_matchable = float(fg_matches_mask.sum()) / (in_segmentation_items + 1e-5)

            edge_data.src_pts_xy_roma_matchable = src_pts_xy_int[matchable_fg_matches_mask]
            edge_data.dst_pts_xy_roma_matchable = dst_pts_xy_int[matchable_fg_matches_mask]
            edge_data.src_dst_certainty_roma_matchable = certainty[matchable_fg_matches_mask]
            source_datagraph_node.relative_area_matchable = relative_area_matchable

        min_num_of_certain_matches = self.config.min_number_of_reliable_matches
        certain_matches_share_threshold = self.current_flow_reliability_threshold
        match_certainty_threshold = source_datagraph_node.roma_certainty_threshold

        reliability = compute_matching_reliability(src_pts_xy_int, certainty, source_segmentation_mask,
                                                   match_certainty_threshold, min_num_of_certain_matches)

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= certain_matches_share_threshold

        return reliability


class RoMaFrameFilterRANSAC(RoMaFrameFilter):

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        src_pts_xy, dst_pts_xy, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.config.roma_sample_size, as_int=False,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))
        src_pts_xy_np = src_pts_xy.numpy(force=True)
        dst_pts_xy_np = dst_pts_xy.numpy(force=True)

        frame_data_source = self.data_graph.get_frame_data(source_frame)
        frame_data_target = self.data_graph.get_frame_data(target_frame)
        camera_K1 = self.data_graph.get_frame_data(source_frame).gt_pinhole_K
        camera_K2 = self.data_graph.get_frame_data(target_frame).gt_pinhole_K

        source_shape = frame_data_source.image_shape
        target_shape = frame_data_target.image_shape

        ransac_opts = pycolmap.RANSACOptions()
        ransac_opts.max_error = 0.5
        # ransac_opts.confidence = 0.99999999

        if camera_K1 is not None and camera_K2 is not None:
            K_params1 = colmap_K_params_vec(camera_K1.numpy(force=True))
            K_params2 = colmap_K_params_vec(camera_K2.numpy(force=True))

            camera1 = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.PINHOLE, width=source_shape.width,
                                      height=source_shape.height, params=K_params1)
            camera2 = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.PINHOLE, width=target_shape.width,
                                      height=target_shape.height, params=K_params2)

            ransac_res = pycolmap.estimate_essential_matrix(src_pts_xy_np, dst_pts_xy_np, camera1, camera2, ransac_opts)
        else:
            ransac_res = pycolmap.estimate_fundamental_matrix(src_pts_xy_np, dst_pts_xy_np, ransac_opts)

        if ransac_res is not None:
            inlier_mask = ransac_res.get('inlier_mask')
            reliability = inlier_mask.sum() / len(inlier_mask)
        else:
            reliability = 0.

        return reliability


class FrameFilterPassThrough(BaseFrameFilter):

    def filter_frames(self, frame_i: int):
        if frame_i % self.config.passthrough_frame_filter_skip == 0:
            self.add_keyframe(frame_i)

    def add_keyframe(self, frame_i: int):

        forward_edges = [(i, frame_i) for i in range(frame_i)]
        backward_edges = [(frame_i, i) for i in range(frame_i)]

        self.keyframe_graph.add_edges_from(forward_edges)
        self.keyframe_graph.add_edges_from(backward_edges)

        frame_data = self.data_graph.get_frame_data(frame_i)
        frame_data.matching_source_keyframe = frame_i - 1 if frame_i > 0 else 0


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


def compute_matching_reliability(src_pts_xy_int: torch.Tensor, certainty: torch.Tensor,
                                 source_segmentation_mask: torch.Tensor, match_certainty_threshold: float,
                                 min_num_of_certain_matches: int = 0) -> float:
    fg_matches_mask = source_segmentation_mask[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
    fg_certainties = certainty[fg_matches_mask]
    fg_certainties_above_threshold = fg_certainties > match_certainty_threshold
    reliability = fg_certainties_above_threshold.sum() / (fg_certainties.numel() + 1e-5)
    enough_certain_matches = (fg_certainties_above_threshold.numel() > min_num_of_certain_matches)
    reliability *= float(enough_certain_matches)
    reliability = reliability.item()
    return reliability

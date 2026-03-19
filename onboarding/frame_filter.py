from abc import abstractmethod
from time import time
from typing import Tuple

import networkx as nx
import torch

from data_providers.flow_provider import MatchingProvider
from data_structures.data_graph import DataGraph
from configs.glopose_config import OnboardingConfig
from onboarding.ransac import estimate_inlier_mask
from utils.image_utils import otsu_threshold


class BaseFrameFilter:
    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, device: str = 'cuda'):
        self.onboarding: OnboardingConfig = onboarding
        self.n_frames: int = n_frames
        self.device: str = device
        self.data_graph: DataGraph = data_graph
        self.keyframe_graph: nx.DiGraph = nx.DiGraph()

    def get_keyframe_graph(self) -> nx.DiGraph:
        # if len(self.keyframe_graph.nodes) <= 2:
        #     nodes_list = sorted(list(self.keyframe_graph.nodes))
        #     middle_node = (nodes_list[-1] - nodes_list[0]) // 2
        #     if nodes_list[0] < middle_node < nodes_list[-1]:
        #         self.keyframe_graph.add_edge(nodes_list[0], middle_node)
        #         self.keyframe_graph.add_edge(middle_node, nodes_list[-1])

        if self.onboarding.view_graph_strategy == 'dense':
            nodes_list = list(self.keyframe_graph.nodes)
            for i in range(len(nodes_list)):
                for j in range(len(nodes_list)):
                    if i != j:  # Don't add self-loops
                        self.keyframe_graph.add_edge(nodes_list[i], nodes_list[j])
        else:
            assert self.onboarding.view_graph_strategy == 'from_matching'

        return self.keyframe_graph

    @abstractmethod
    def add_keyframe(self, frame_i: int):
        pass

    @abstractmethod
    def filter_frames(self, frame_i: int):
        pass


class RoMaFrameFilter(BaseFrameFilter):

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, flow_provider: MatchingProvider, device: str = 'cuda'):

        super().__init__(onboarding, n_frames, data_graph, device)

        self.flow_provider: MatchingProvider = flow_provider

        self.matching_reliability_threshold = self.onboarding.flow_reliability_threshold

    def add_keyframe(self, frame_i: int):
        self.keyframe_graph.add_node(frame_i)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(frame_i, frame_i,
                                                                  self.onboarding.sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        kf_data = self.data_graph.get_frame_data(frame_i)
        if self.onboarding.certainty_threshold_strategy == 'otsu':
            certainty_threshold = otsu_threshold(certainty)
            if certainty_threshold is None and frame_i > 0:
                prev_kf = kf_data.matching_source_keyframe
                certainty_threshold = self.data_graph.get_frame_data(prev_kf).roma_certainty_threshold
            else:
                certainty_threshold = self.onboarding.min_certainty_threshold
        else:
            certainty_threshold = self.onboarding.min_certainty_threshold
        kf_data.roma_certainty_threshold = certainty_threshold

        if self.onboarding.matchability_based_reliability:
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

    def _init_first_frame(self):
        self.add_keyframe(0)
        first_frame_node = self.data_graph.get_frame_data(0)
        first_frame_node.reliable_sources = {0}
        first_frame_node.matching_source_keyframe = 0
        first_frame_node.current_flow_reliability_threshold = self.matching_reliability_threshold

    def _find_source(self, frame_i: int, preceding_source: int, preceding_reliable: bool
                     ) -> Tuple[int, set]:
        need_all_keyframes = (
            self.onboarding.edge_strategy == 'always' or not preceding_reliable
        )

        if need_all_keyframes:
            reliable_kfs, best_source = self._match_to_all_keyframes(frame_i)
            if best_source is not None:
                return best_source, reliable_kfs
            # No reliable match anywhere — add frame_i-1 as new keyframe
            new_source = frame_i - 1
            self.add_keyframe(new_source)
            self.keyframe_graph.add_edge(preceding_source, new_source)
            return new_source, {new_source}

        # Preceding match is reliable and strategy is 'on_unreliable'
        return preceding_source, {preceding_source}

    def _ensure_last_frame_keyframe(self, frame_i: int, source: int, reliable_kfs: set
                                    ) -> Tuple[int, set]:
        if frame_i in self.keyframe_graph.nodes:
            return source, reliable_kfs

        self.add_keyframe(frame_i)

        if reliable_kfs:
            for kf in reliable_kfs:
                self.keyframe_graph.add_edge(kf, frame_i)
            return source, reliable_kfs

        # No reliable match — try matching against preceding frame
        prev = frame_i - 1
        if prev not in self.keyframe_graph.nodes:
            self.add_keyframe(prev)
            # Connect prev to its own source
            prev_source = self.data_graph.get_frame_data(prev).matching_source_keyframe
            if prev_source is not None and prev_source in self.keyframe_graph.nodes:
                self.keyframe_graph.add_edge(prev_source, prev)

        self.flow_reliability(prev, frame_i)
        edge = self.data_graph.get_edge_observations(prev, frame_i)
        if edge.is_match_reliable:
            self.keyframe_graph.add_edge(prev, frame_i)
            return prev, {prev}

        return source, reliable_kfs

    @torch.no_grad()
    def filter_frames(self, frame_i: int):
        start_time = time()

        if frame_i == 0:
            self._init_first_frame()
            return

        # Step 1: Check preceding frame's match reliability
        preceding_source = self.data_graph.get_frame_data(frame_i - 1).matching_source_keyframe
        self.flow_reliability(preceding_source, frame_i - 1)
        preceding_reliable = self.data_graph.get_edge_observations(
            preceding_source, frame_i - 1).is_match_reliable

        # Step 2: Determine source and collect reliable keyframe matches
        if frame_i == 1:
            source, reliable_kfs = 0, {0}
        else:
            source, reliable_kfs = self._find_source(frame_i, preceding_source, preceding_reliable)

        # Step 3: Handle last frame
        is_last = (frame_i == self.n_frames - 1)
        if is_last and self.onboarding.always_add_last_frame:
            source, reliable_kfs = self._ensure_last_frame_keyframe(
                frame_i, source, reliable_kfs)

        # Step 4: Final reliability + metadata
        flow_reliability = self.flow_reliability(source, frame_i)
        print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~{flow_reliability}')
        node = self.data_graph.get_frame_data(frame_i)
        node.pose_estimation_time = time() - start_time
        node.current_flow_reliability_threshold = self.matching_reliability_threshold
        node.reliable_sources = {source} | reliable_kfs
        node.matching_source_keyframe = source

    def _match_to_all_keyframes(self, frame_i: int) -> Tuple[set, int | None]:
        best_source: int = 0
        best_source_reliability: float = 0.
        reliable_kfs = set()

        for kf in list(self.keyframe_graph.nodes):
            reliability = self.flow_reliability(kf, frame_i)
            if reliability >= self.matching_reliability_threshold:
                reliable_kfs.add(kf)
            if reliability > best_source_reliability:
                best_source = kf
                best_source_reliability = reliability

        if best_source_reliability < self.matching_reliability_threshold:
            return set(), None
        return reliable_kfs, best_source

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        dev = self.device
        source_datagraph_node = self.data_graph.get_frame_data(source_frame)
        source_segmentation_mask = source_datagraph_node.frame_observation.observed_segmentation.squeeze().to(dev)

        H_A, W_A = source_datagraph_node.image_shape.height, source_datagraph_node.image_shape.width
        assert source_segmentation_mask.shape[-2:] == (H_A, W_A)

        src_pts_xy_int, dst_pts_xy_int, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.onboarding.sample_size, as_int=True,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        assert ((src_pts_xy_int[:, 0] >= 0) & (src_pts_xy_int[:, 0] < W_A)).all()
        assert ((src_pts_xy_int[:, 1] >= 0) & (src_pts_xy_int[:, 1] < H_A)).all()
        assert certainty.shape[0] == src_pts_xy_int.shape[0] and certainty.shape[0] == dst_pts_xy_int.shape[0]

        if self.onboarding.matchability_based_reliability:
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

        min_num_of_certain_matches = self.onboarding.min_number_of_reliable_matches
        certain_matches_share_threshold = self.matching_reliability_threshold
        match_certainty_threshold = source_datagraph_node.roma_certainty_threshold

        reliability = compute_matching_reliability(src_pts_xy_int, certainty, source_segmentation_mask,
                                                   match_certainty_threshold, min_num_of_certain_matches)

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= certain_matches_share_threshold

        return reliability


class FrameFilterRANSAC(RoMaFrameFilter):

    def flow_reliability(self, source_frame: int, target_frame: int) -> float:
        src_pts_xy, dst_pts_xy, certainty = (
            self.flow_provider.get_source_target_points_datagraph(source_frame, target_frame,
                                                                  self.onboarding.sample_size, as_int=False,
                                                                  zero_certainty_outside_segmentation=True,
                                                                  only_foreground_matches=True))
        src_pts_xy_np = src_pts_xy.numpy(force=True)
        dst_pts_xy_np = dst_pts_xy.numpy(force=True)
        certainty_np = certainty.numpy(force=True)

        frame_data_source = self.data_graph.get_frame_data(source_frame)
        frame_data_target = self.data_graph.get_frame_data(target_frame)
        camera_K1 = frame_data_source.gt_pinhole_K
        camera_K2 = frame_data_target.gt_pinhole_K

        K1_np = camera_K1.numpy(force=True) if camera_K1 is not None else None
        K2_np = camera_K2.numpy(force=True) if camera_K2 is not None else None

        ransac_config = self.onboarding.ransac
        inlier_mask = estimate_inlier_mask(
            src_pts_xy_np, dst_pts_xy_np, ransac_config,
            K1=K1_np, K2=K2_np,
            source_shape=frame_data_source.image_shape,
            target_shape=frame_data_target.image_shape,
            confidences=certainty_np)

        edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)

        if inlier_mask is not None:
            reliability = float(inlier_mask.sum() / len(inlier_mask))
            edge_data.ransac_inliers = torch.from_numpy(src_pts_xy_np[inlier_mask])
            edge_data.ransac_outliers = torch.from_numpy(src_pts_xy_np[~inlier_mask])
        else:
            reliability = 0.

        edge_data.reliability_score = reliability
        edge_data.is_match_reliable = reliability >= self.matching_reliability_threshold

        return reliability


class FrameFilterPassThrough(BaseFrameFilter):

    def filter_frames(self, frame_i: int):
        if frame_i % self.onboarding.passthrough_skip == 0:
            self.add_keyframe(frame_i)

        # Set matching_source_keyframe for every frame (nearest preceding keyframe)
        frame_data = self.data_graph.get_frame_data(frame_i)
        nearest_keyframe = frame_i - (frame_i % self.onboarding.passthrough_skip)
        frame_data.matching_source_keyframe = nearest_keyframe if nearest_keyframe >= 0 else 0

    def add_keyframe(self, frame_i: int):
        forward_edges = [(i, frame_i) for i in range(frame_i)]
        backward_edges = [(frame_i, i) for i in range(frame_i)]

        self.keyframe_graph.add_edges_from(forward_edges)
        self.keyframe_graph.add_edges_from(backward_edges)


class FrameFilterSift(BaseFrameFilter):

    def __init__(self, onboarding: OnboardingConfig, n_frames: int, data_graph: DataGraph, sift_matcher: MatchingProvider):

        super().__init__(onboarding, n_frames, data_graph)

        self.sift_matcher: MatchingProvider = sift_matcher

    @torch.no_grad()
    def filter_frames(self, current_frame_idx: int):

        start_time = time()

        if current_frame_idx == 0:
            self.keyframe_graph.add_node(current_frame_idx)
            self.data_graph.get_frame_data(current_frame_idx).matching_source_keyframe = current_frame_idx
            # Create self-edge so visualization code can access it
            if not self.data_graph.G.has_edge(current_frame_idx, current_frame_idx):
                self.data_graph.add_new_arc(current_frame_idx, current_frame_idx)
            edge_data = self.data_graph.get_edge_observations(current_frame_idx, current_frame_idx)
            edge_data.num_matches = 0
            edge_data.is_match_reliable = True
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

        more_than_enough_matches = self.onboarding.sift_filter_good_to_add_matches
        min_matches = self.onboarding.sift_filter_min_matches

        reliable_keyframe_found = False
        we_stepped_back = False

        while not reliable_keyframe_found:

            num_matches = self.compute_sift_reliability(keyframe_idx, current_frame_idx)
            print(f'{num_matches}, {min_matches}, {more_than_enough_matches}')

            if num_matches >= self.onboarding.sift_filter_min_matches:
                self.keyframe_graph.add_edge(keyframe_idx, current_frame_idx)

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

        source_frame_observation = self.data_graph.get_frame_data(frame_idx1)
        target_frame_observation = self.data_graph.get_frame_data(frame_idx2)

        source_img = source_frame_observation.frame_observation.observed_image.squeeze()
        target_img = target_frame_observation.frame_observation.observed_image.squeeze()

        source_seg = source_frame_observation.frame_observation.observed_segmentation.squeeze()
        target_seg = target_frame_observation.frame_observation.observed_segmentation.squeeze()

        src_pts, dst_pts, certainty = self.sift_matcher.get_source_target_points(
            source_img, target_img, source_image_segmentation=source_seg,
            target_image_segmentation=target_seg)

        num_matches = len(src_pts)

        if not self.data_graph.G.has_edge(frame_idx1, frame_idx2):
            self.data_graph.add_new_arc(frame_idx1, frame_idx2)
        edge_data = self.data_graph.get_edge_observations(frame_idx1, frame_idx2)

        edge_data.num_matches = num_matches
        edge_data.is_match_reliable = num_matches >= self.onboarding.sift_filter_min_matches
        edge_data.src_pts_xy_roma = src_pts
        edge_data.dst_pts_xy_roma = dst_pts
        edge_data.src_dst_certainty_roma = certainty

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


def create_frame_filter(onboarding: OnboardingConfig, device: str, n_frames: int,
                        data_graph: DataGraph,
                        flow_provider: MatchingProvider = None) -> BaseFrameFilter:
    """Factory that maps a config string to a BaseFrameFilter instance.

    Args:
        onboarding: OnboardingConfig with frame_filter, sift sub-configs.
        device: PyTorch device string (e.g. 'cuda').
        n_frames: Total number of input frames.
        data_graph: The shared DataGraph.
        flow_provider: Flow provider for dense-matching-based filters.
    """

    def _dense_matching():
        return RoMaFrameFilter(onboarding, n_frames, data_graph, flow_provider, device)

    def _ransac():
        return FrameFilterRANSAC(onboarding, n_frames, data_graph, flow_provider, device)

    def _passthrough():
        return FrameFilterPassThrough(onboarding, n_frames, data_graph)

    def _sift():
        from data_providers.matching_provider_sift import (
            SparseMatchingProvider, SIFTKeypointDetector, LightGlueKeypointMatcher)
        detector = SIFTKeypointDetector(device)
        matcher = LightGlueKeypointMatcher(device)
        sift_provider = SparseMatchingProvider(detector, matcher,
                                               num_features=onboarding.sift.sift_filter_num_feats,
                                               device=device)
        return FrameFilterSift(onboarding, n_frames, data_graph, sift_provider)

    filters = {
        'dense_matching': _dense_matching,
        'RANSAC': _ransac,
        'passthrough': _passthrough,
        'SIFT': _sift,
    }
    if onboarding.frame_filter not in filters:
        raise ValueError(f"Unknown frame filter '{onboarding.frame_filter}'. Options: {list(filters.keys())}")
    return filters[onboarding.frame_filter]()

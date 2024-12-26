from time import time
from typing import List, Tuple, Optional

import cv2
import torch
from kornia.feature import get_laf_center, match_adalam
from kornia.geometry import Se3
from kornia.image import ImageSize
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from torchvision.transforms.functional import to_pil_image

from data_providers.flow_provider import RoMaFlowProviderDirect
from data_structures.data_graph import DataGraph, CommonFrameData
from data_structures.pose_icosphere import PoseIcosphere
from flow import roma_warp_to_pixel_coordinates
from tracker_config import TrackerConfig
from utils.sift import sift_to_rootsift


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

                    mock_pose = Se3.identity(1, device=self.config.device)
                    self.pose_icosphere.insert_new_reference(cam_frame_data.frame_observation, mock_pose, best_source)
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


class FrameFilterSift(FrameFilter):

    def __init__(self, config: TrackerConfig, data_graph: DataGraph, pose_icosphere, image_shape: ImageSize,
                 flow_provider: RoMaFlowProviderDirect):

        super().__init__(config, data_graph, pose_icosphere, image_shape, flow_provider)

    @torch.no_grad()
    def filter_frames(self, frame_i: int):

        start_time = time()

        idx = frame_i - 1
        preceding_frame_node = self.data_graph.get_frame_data(idx)
        source1 = preceding_frame_node.long_jump_source
        result = idx, source1
        preceding_frame_idx, preceding_source = result

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

                    mock_pose = Se3.identity(1, device=self.config.device)
                    self.pose_icosphere.insert_new_reference(cam_frame_data.frame_observation, mock_pose, best_source)
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

    def filter_frames(self, frame_i: int):
        print("Detection features")
        keyframes_single_dir = []

        device = self.config.device
        selected_keyframe_idxs = [0]
        matching_pairs_original_idx = []
        print("Now matching to add keyframes")
        max_matches = self.config.sift_filter_good_to_add_matches
        min_matches = self.config.sift_filter_min_matches

        idx1 = selected_keyframe_idxs[-1]
        fname1 = keyframes_single_dir[idx1]
        key1 = fname1.split('/')[-1]

        lafs_img1, keypoints_img1, descriptors_img1 = self.detect_sift(image1, segmentation1)

        img1 = cv2.imread(fname1)
        hw1 = img1.shape[:2]
        done = False
        idx2 = idx1
        we_stepped_back = False
        while not done:
            idx2 = idx2 + 1
            is_last_frame = idx2 == len(keyframes_single_dir) - 1
            if idx2 >= len(keyframes_single_dir):
                break
            fname2 = keyframes_single_dir[idx2]
            key2 = fname2.split('/')[-1]

            lafs_img2, keypoints_img2, descriptors_img2 = self.detect_sift(image2, segmentation2)
            img2 = cv2.imread(fname2)
            hw2 = img2.shape[:2]
            with torch.inference_mode():
                dists, idxs = match_adalam(descriptors_img1, descriptors_img2,
                                           lafs_img1, lafs_img2,  # Adalam takes into account also geometric information
                                           hw1=hw1, hw2=hw2)
            num_matches = len(idxs)

            if num_matches >= max_matches:
                if (not we_stepped_back):
                    print("Too many matches, skipping")
                    if (len(selected_keyframe_idxs) == 1) and is_last_frame:
                        # We need at least two keyframes
                        selected_keyframe_idxs.append(idx1)
                        matching_pairs_original_idx.append((idx1, idx2))
                        selected_keyframe_idxs.append(idx2)
                        break
                elif is_last_frame:
                    print("Last frame, adding")
                    selected_keyframe_idxs.append(idx1)
                    matching_pairs_original_idx.append((idx1, idx2))
                    selected_keyframe_idxs.append(idx2)
                    break
                else:
                    print(f"Step back was good, adding idx1={idx1}")
                    selected_keyframe_idxs.append(idx1)
                    we_stepped_back = False
                continue
            if (len(idxs) <= max_matches) and (len(idxs) >= min_matches):
                print("Adding keyframe")
                selected_keyframe_idxs.append(idx2)
                selected_keyframe_idxs.append(idx1)
                matching_pairs_original_idx.append((idx1, idx2))
                idx1 = idx2
                key1, lafs_img1, descriptors_img1, hw1 = key2, lafs_img2, descriptors_img2, hw2
            if len(idxs) < min_matches:  # try going back
                print("Too few matches, going back")
                idx1 = idx2 - 1
                we_stepped_back = True
                if (idx1 <= 0):
                    done = True
                elif (idx1 in selected_keyframe_idxs):
                    print(f"We cannot match {idx2}, skipping it")
                    idx2 += 1
                else:
                    fname1 = keyframes_single_dir[idx1]
                    key1 = fname1.split('/')[-1]
                    lafs_img1 = torch.from_numpy(f_laf[key1][...]).to(device)
                    descriptors_img1 = torch.from_numpy(f_desc[key1][...]).to(device)
                    img1 = cv2.imread(fname1)
                    hw1 = img1.shape[:2]
        matching_pairs_new_idxs = []
        selected_keyframe_idxs = sorted(list(set(selected_keyframe_idxs)))
        print(f'{selected_keyframe_idxs=}')
        print(f'{matching_pairs_original_idx=}')

        for idx1, idx2 in matching_pairs_original_idx:
            matching_pairs_new_idxs.append((selected_keyframe_idxs.index(idx1), selected_keyframe_idxs.index(idx2)))
        keyframes = [keyframes_single_dir[i] for i in selected_keyframe_idxs]
        keysegs = [segmentations[i] for i in selected_keyframe_idxs]
        return keyframes, keysegs, matching_pairs_new_idxs

    def detect_sift(self, image: torch.Tensor, segmentation: Optional[torch.Tensor] = None):

        device = self.config.device

        sift = cv2.SIFT_create(self.config.sift_filter_num_feats, edgeThreshold=-1000, contrastThreshold=-1000)

        if segmentation is not None:
            segmentation_np = to_pil_image(image)
        else:
            segmentation_np = None

        image_np = to_pil_image(image)
        keypoints, descriptors = sift.detectAndCompute(image_np, segmentation_np)
        lafs = laf_from_opencv_SIFT_kpts(keypoints)
        descriptors = sift_to_rootsift(torch.from_numpy(descriptors)).to(device)
        desc_dim = descriptors.shape[-1]
        keypoints = get_laf_center(lafs).reshape(-1, 2).detach().cpu().numpy()
        descriptors = descriptors.reshape(-1, desc_dim).detach().cpu().numpy()

        return lafs, keypoints, descriptors

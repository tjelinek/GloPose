from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from kornia.feature import get_laf_center, match_adalam
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model
from torchvision.transforms.functional import to_pil_image

from data_structures.data_graph import DataGraph
from utils.sift import sift_to_rootsift


class SIFTMatchingProvider:

    def __init__(self, data_graph: DataGraph, num_sift_features: int, device: str):
        self.device = device
        self.data_graph = data_graph

        self.num_sift_features: int = num_sift_features

    def detect_sift_features(self, source_image_idx: int, device: Optional[str] = 'cpu',
                             save_to_datagraph: bool = False):

        frame_data = self.data_graph.get_frame_data(source_image_idx)

        frame1_image = frame_data.frame_observation.observed_image.squeeze()
        frame1_segmentation = frame_data.frame_observation.observed_segmentation.squeeze()

        lafs, keypoints, descriptors = detect_sift_features(frame1_image, self.num_sift_features,
                                                            frame1_segmentation, device)

        if save_to_datagraph:
            frame_data = self.data_graph.get_frame_data(source_image_idx)
            frame_data.sift_lafs = lafs
            frame_data.sift_keypoints = keypoints
            frame_data.sift_descriptors = descriptors

        return lafs, keypoints, descriptors

    def match_images_sift(self, source_image_idx: int, target_image_idx: int, device: Optional[str] = 'cpu',
                          save_to_datagraph: bool = False):

        frame1_data = self.data_graph.get_frame_data(source_image_idx)
        frame2_data = self.data_graph.get_frame_data(target_image_idx)

        image1 = frame1_data.frame_observation.observed_image.squeeze()

        image2 = frame2_data.frame_observation.observed_image.squeeze()

        lafs1, keypoints1, descriptors1 = self.detect_sift_features(source_image_idx, device, save_to_datagraph)
        lafs2, keypoints2, descriptors2 = self.detect_sift_features(target_image_idx, device, save_to_datagraph)

        hw1 = tuple(image1.shape[-2:])
        hw2 = tuple(image2.shape[-2:])

        dists, idxs = match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2)

        if save_to_datagraph:
            if not self.data_graph.G.has_edge(source_image_idx, target_image_idx):
                self.data_graph.add_new_arc(source_image_idx, target_image_idx)
            edge_data = self.data_graph.get_edge_observations(source_image_idx, target_image_idx)
            edge_data.sift_keypoint_indices = idxs
            edge_data.sift_dists = dists

        return dists, idxs


class PrecomputedSIFTMatchingProvider(SIFTMatchingProvider):

    def __init__(self, data_graph: DataGraph, num_sift_features: int, cache_dir: Path, allow_missing: bool = True,
                 device: Optional[str] = 'cpu'):
        super().__init__(data_graph, num_sift_features, device)
        self.flow_model: roma_model = roma_outdoor(device=device)

        self.saved_flow_paths = cache_dir

        # Features
        self.lafs_path = cache_dir / 'sift_lafs'
        self.keypoints_path = cache_dir / 'sift_keypoints'
        self.descriptors_path = cache_dir / 'sift_descriptors'

        self.lafs_path.mkdir(exist_ok=True, parents=True)
        self.keypoints_path.mkdir(exist_ok=True, parents=True)
        self.descriptors_path.mkdir(exist_ok=True, parents=True)

        # Matching
        self.matching_dists_paths = cache_dir / 'sift_matching_dists'
        self.matching_indices_paths = cache_dir / 'sift_matching_indices'

        self.matching_dists_paths.mkdir(exist_ok=True, parents=True)
        self.matching_indices_paths.mkdir(exist_ok=True, parents=True)

        self.allow_missing: bool = allow_missing

    def detect_sift_features(self, source_image_idx: int, device: Optional[str] = 'cpu',
                             save_to_datagraph: bool = False):

        image_name = self.data_graph.get_frame_data(source_image_idx).image_filename

        lafs_path = self.lafs_path / f'{image_name.stem}.pt'
        keypoints_path = self.keypoints_path / f'{image_name.stem}.pt'
        descriptors_path = self.descriptors_path / f'{image_name.stem}.pt'

        if lafs_path.exists() and keypoints_path.exists() and descriptors_path.exists():
            lafs = torch.load(lafs_path).to(device)
            keypoints = torch.load(keypoints_path)
            descriptors = torch.load(descriptors_path)
        else:
            lafs, keypoints, descriptors = super().detect_sift_features(source_image_idx, device)

            torch.save(lafs, lafs_path)
            torch.save(keypoints, keypoints_path)
            torch.save(descriptors, descriptors_path)

        if save_to_datagraph:
            frame_data = self.data_graph.get_frame_data(source_image_idx)
            frame_data.sift_lafs = lafs
            frame_data.sift_keypoints = keypoints
            frame_data.sift_descriptors = descriptors

        return lafs, keypoints, descriptors


def detect_sift_features(image: torch.Tensor, num_features: int, segmentation: Optional[torch.Tensor] = None,
                         device: str = 'cpu'):
    sift = cv2.SIFT_create(num_features, edgeThreshold=-1000, contrastThreshold=-1000)

    if segmentation is not None:
        segmentation_np = np.array(to_pil_image(segmentation))
    else:
        segmentation_np = None

    image_rgb = np.array(to_pil_image(image))
    image_cv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    keypoints, descriptors = sift.detectAndCompute(image_cv, segmentation_np)
    lafs = laf_from_opencv_SIFT_kpts(keypoints)
    descriptors = sift_to_rootsift(torch.from_numpy(descriptors)).to(device)
    desc_dim = descriptors.shape[-1]
    keypoints = get_laf_center(lafs).reshape(-1, 2).to(device)
    descriptors = descriptors.reshape(-1, desc_dim).to(device)

    return lafs, keypoints, descriptors


def match_images_sift(image1: torch.Tensor, image2, num_features: int, segmentation1: Optional[torch.Tensor] = None,
                      segmentation2=None, device: str = 'cpu'):

    lafs1, keypoints1, descriptors1 = detect_sift_features(image1, num_features, segmentation1, device)
    lafs2, keypoints2, descriptors2 = detect_sift_features(image2, num_features, segmentation2, device)

    hw1 = tuple(image1.shape[-2:])
    hw2 = tuple(image2.shape[-2:])

    dists, idxs = match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2)

    return dists, idxs


def match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2):
    with torch.inference_mode():
        dists, idxs = match_adalam(descriptors1, descriptors2,
                                   lafs1, lafs2,  # Adalam takes into account also geometric information
                                   hw1=hw1, hw2=hw2)
    return dists, idxs

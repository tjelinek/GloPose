from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from kornia.feature import get_laf_center, match_adalam, LightGlueMatcher
from kornia_moons.feature import laf_from_opencv_SIFT_kpts
from torchvision.transforms.functional import to_pil_image

from configs.matching_configs.sift_configs.base_sift_config import BaseSiftConfig
from data_providers.flow_provider import FlowProviderDirect
from data_structures.data_graph import DataGraph
from utils.sift import sift_to_rootsift


class SIFTMatchingProviderDirect(FlowProviderDirect):

    def __init__(self, sift_config: BaseSiftConfig, device: str):
        super().__init__(device)
        self.num_sift_features: int = sift_config.sift_filter_num_feats
        self.device: str = device

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None, zero_certainty_outside_segmentation: bool = False):

        raise NotImplementedError("This awaits implementation.\n"
                                  "The source target points muse be converted to RoMa format.")

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False):
        lafs1, kpts1_xy, descriptors1 = detect_sift_features(source_image, self.num_sift_features,
                                                             source_image_segmentation, self.device)
        lafs2, kpts2_xy, descriptors2 = detect_sift_features(target_image, self.num_sift_features,
                                                             source_image_segmentation, self.device)

        kpts1_xy_int = kpts1_xy.to(torch.long)  # XY format
        kpts2_xy_int = kpts2_xy.to(torch.long)  # XY format

        in_segment_mask1 = source_image_segmentation[kpts1_xy_int[:, 1], kpts1_xy_int[:, 0]].to(torch.bool)
        in_segment_mask2 = source_image_segmentation[kpts2_xy_int[:, 1], kpts2_xy_int[:, 0]].to(torch.bool)

        kpts1_xy = kpts1_xy[in_segment_mask1]
        lafs1 = lafs1[:, in_segment_mask1]
        descriptors1 = descriptors1[in_segment_mask1]

        kpts2_xy = kpts2_xy[in_segment_mask2]
        lafs2 = lafs2[:, in_segment_mask2]
        descriptors2 = descriptors2[in_segment_mask2]

        hw1 = tuple(source_image.shape[-2:])
        hw2 = tuple(target_image.shape[-2:])

        dists, idxs = match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2)

        src_pts_xy = kpts1_xy[idxs[:, 0]]
        dst_pts_xy = kpts2_xy[idxs[:, 1]]

        certainty = torch.ones(src_pts_xy.shape[0], dtype=torch.float, device=self.device)

        if as_int:
            src_pts_xy, dst_pts_xy = self.keypoints_to_int(src_pts_xy, dst_pts_xy, source_image, target_image)

        return src_pts_xy, dst_pts_xy, certainty

    def sample(self, warp: torch.Tensor, certainty: torch.Tensor, sample: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class PrecomputedUFMFlowProviderDirect(SIFTMatchingProviderDirect):

    def __init__(self, num_sift_features: int, data_graph: DataGraph, device: str):

        SIFTMatchingProviderDirect.__init__(self, num_sift_features, device)
        self.data_graph: DataGraph = data_graph

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None,
                     zero_certainty_outside_segmentation=False) -> Tuple[torch.Tensor, torch.Tensor]:

        pass


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
    lafs = laf_from_opencv_SIFT_kpts(keypoints).to(device)
    descriptors = sift_to_rootsift(torch.from_numpy(descriptors)).to(device)
    desc_dim = descriptors.shape[-1]
    keypoints = get_laf_center(lafs).reshape(-1, 2).to(device)
    descriptors = descriptors.reshape(-1, desc_dim).to(device)

    return lafs, keypoints, descriptors


def match_features_sift(descriptors1, descriptors2, lafs1, lafs2, hw1, hw2, alg='lightglue'):
    if alg == 'lightglue':
        matcher = LightGlueMatcher('sift').eval().to(descriptors1.device)
    elif alg == 'adalam':
        matcher = match_adalam
    else:
        raise ValueError('unknown algorithm')

    with torch.inference_mode():
        dists, idxs = matcher(descriptors1, descriptors2, lafs1, lafs2, hw1=hw1, hw2=hw2)
        # Adalam takes into account also geometric information and image sizes

    return dists, idxs

from pathlib import Path
from typing import List, Union, Optional

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


class RoMaFlowProviderDirect:

    def __init__(self, data_graph: DataGraph, device):
        self.device = device
        self.data_graph = data_graph

    def detect_sift_features_datagraph(self, source_image_idx, num_features: int, device: Optional[str] = 'cpu'):

        frame_data = self.data_graph.get_frame_data(source_image_idx)

        frame1_image = frame_data.frame_observation.observed_image.squeeze()
        frame1_segmentation = frame_data.frame_observation.observed_segmentation.squeeze()

        return detect_sift_features(frame1_image, num_features, frame1_segmentation, device)

    def add_flows_into_datagraph(self, flow_source_frame, flow_target_frame):
        edge_data = self.data_graph.get_edge_observations(flow_source_frame, flow_target_frame)
        if edge_data.flow_warp is None or edge_data.flow_certainty is None:
            warp, certainty = self.detect_sift_features_datagraph(flow_source_frame, flow_target_frame, sample=10000)
        else:
            warp, certainty = edge_data.flow_warp, edge_data.flow_certainty
            # TODO handle sampling when reading from datagraph

        edge_data.flow_warp = warp
        edge_data.flow_certainty = certainty


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect):

    def __init__(self, data_graph: DataGraph, device, cache_dir: Path, image_files_paths: List,
                 allow_missing: bool = True):
        super().__init__(data_graph, device)
        self.flow_model: roma_model = roma_outdoor(device=device)

        self.saved_flow_paths = cache_dir
        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.image_names = [Path(p) for p in image_files_paths]

        self.allow_missing: bool = allow_missing

    def detect_sift_features_datagraph(self, source_image_idx: int, target_image_idx: int, sample=None):

        assert source_image_idx < len(self.image_names)
        assert target_image_idx < len(self.image_names)

        src_image_name = Path(self.image_names[source_image_idx])
        target_image_name = Path(self.image_names[target_image_idx])
        saved_filename = f'{src_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if (not warp_filename.exists() or not certainty_filename.exists()) and self.allow_missing:
            warp, certainty = super().detect_sift_features_datagraph(source_image_idx, target_image_idx)

            torch.save(warp, warp_filename)
            torch.save(certainty, certainty_filename)
        else:
            warp = torch.load(warp_filename).to(self.device)
            certainty = torch.load(certainty_filename).to(self.device)

        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def cached_flow_from_filenames(self, src_image_name: Union[str, Path], target_image_name: Union[str, Path]):

        src_image_name = Path(src_image_name)
        target_image_name = Path(target_image_name)
        saved_filename = f'{src_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if warp_filename.exists() and certainty_filename.exists():
            warp = torch.load(warp_filename).to(self.device)
            certainty = torch.load(certainty_filename).to(self.device)
            return warp, certainty

        return None


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

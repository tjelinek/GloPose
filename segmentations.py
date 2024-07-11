from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable, List

import cv2
import torch
import imageio
import numpy as np

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from auxiliary_scripts.image_utils import resize_and_filter_image
from data_structures.keyframe_buffer import FrameObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from tracker_config import TrackerConfig


class BaseTracker(ABC):
    def __init__(self, perc, max_width, feature_extractor, device=torch.device('cuda')):
        self.downsample_factor = perc
        self.max_width = max_width
        self.feature_extractor = feature_extractor
        self.shape = None
        self.device = device

    @abstractmethod
    def next(self, file) -> FrameObservation:
        pass

    def process_segm(self, img):
        segment = cv2.resize(img, self.shape[1::-1]).astype(np.float64)
        width = int(self.shape[1] * self.downsample_factor)
        height = int(self.shape[0] * self.downsample_factor)
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        segm = transforms.ToTensor()(segment)
        return segm

    def standardize_image_and_segment(self, image, segment):
        new_width = int(image.shape[1] * self.downsample_factor)
        new_height = int(image.shape[0] * self.downsample_factor)
        image = resize_and_filter_image(image, new_width, new_height)

        segment = (segment > 0.5).astype(np.float64)
        segment_resized = cv2.resize(segment, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
        segment = torch.from_numpy(segment_resized).cuda()[None, None]
        return image, segment


class SyntheticDataGeneratingTracker(BaseTracker):

    def __init__(self, tracker_config: TrackerConfig, renderer: RenderingKaolin, gt_encoder: Encoder, gt_texture,
                 feature_extractor):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width, feature_extractor)
        self.gt_encoder: Encoder = gt_encoder
        self.renderer = renderer
        self.gt_texture = gt_texture
        self.shape = (tracker_config.max_width, tracker_config.max_width)

    @staticmethod
    def binary_segmentation_from_rendered_segmentation(rendered_segmentations: torch.Tensor):
        rendered_segment_discrete: torch.Tensor = torch.logical_not(torch.lt(rendered_segmentations, 1.0))
        rendered_segment_discrete = rendered_segment_discrete.to(rendered_segmentations.dtype)
        return rendered_segment_discrete

    def next(self, frame_id):
        keyframes = [frame_id]
        flow_frames = [frame_id]

        encoder_result, _ = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        rendering_result = self.renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                                 encoder_result.vertices, self.gt_encoder.face_features,
                                                 self.gt_texture, encoder_result.lights)

        image = rendering_result.rendered_image
        segment = rendering_result.rendered_image_segmentation

        image = image.detach().to(self.device)
        image_feat = self.feature_extractor(image).detach()
        segment = segment.detach().to(self.device)

        frame_observation = FrameObservation(observed_image=image, observed_image_features=image_feat,
                                             observed_segmentation=segment)

        return frame_observation


class PrecomputedTracker(BaseTracker, ABC):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width, feature_extractor)

        image = imageio.v3.imread(images_paths[0])
        self.shape = (image.shape[0], image.shape[1])

        self.images_paths: List[Path] = images_paths
        self.segmentations_paths: List[Path] = segmentations_paths


class PrecomputedTrackerHO3D(PrecomputedTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        self.resize_transform = transforms.Resize((self.shape[0], self.shape[1]),
                                                  interpolation=InterpolationMode.NEAREST)

    def next(self, frame_id):

        image = imageio.v3.imread(self.images_paths[frame_id])
        image = torch.from_numpy(image).cuda().permute(2, 0, 1)[None, None].to(torch.float32) / 255.0

        segmentation = imageio.v3.imread(self.segmentations_paths[frame_id])
        segmentation = torch.from_numpy(segmentation).cuda().permute(2, 0, 1)
        segmentation = self.resize_transform(segmentation)[None, None, [1]].to(torch.bool)

        image_feat = self.feature_extractor(image).detach()

        frame_observation = FrameObservation(observed_image=image, observed_image_features=image_feat,
                                             observed_segmentation=segmentation)

        return frame_observation


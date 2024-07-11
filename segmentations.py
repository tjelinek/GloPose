from abc import abstractmethod, ABC

import sys

import cv2
import torch
import numpy as np

from scipy.ndimage import uniform_filter
from torchvision import transforms
import torch.nn.functional as F

from data_structures.keyframe_buffer import FrameObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from tracker_config import TrackerConfig
from utils import imread

sys.path.insert(0, 'repositories/OSTrack')
from repositories.OSTrack.lib.test.tracker.ostrack import OSTrack
from repositories.OSTrack.lib.test.parameter.ostrack import parameters


def pad_image(image):
    W, H = image.shape[-2:]
    max_size = max(H, W)
    pad_h = max_size - H
    pad_w = max_size - W
    padding = [(pad_h + 1) // 2, pad_h // 2, pad_w // 2, (pad_w + 1) // 2]  # (top, bottom, left, right)

    image = F.pad(image, padding, mode='constant', value=0)

    return image


def resize_and_filter_image(image, new_width, new_height):
    image = cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)
    image = uniform_filter(image, size=(3, 3, 1))

    image_tensor = transforms.ToTensor()(image / 255.0)
    image = image_tensor.unsqueeze(0).float()

    return image


class BaseTracker(ABC):
    def __init__(self, perc, max_width, feature_extractor, device=torch.device('cuda')):
        self.perc = perc
        self.max_width = max_width
        self.feature_extractor = feature_extractor
        self.shape = None
        self.device = device

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments

    @abstractmethod
    def next(self, file) -> FrameObservation:
        pass

    def process_segm(self, img):
        segment = cv2.resize(img, self.shape[1::-1]).astype(np.float64)
        width = int(self.shape[1] * self.perc)
        height = int(self.shape[0] * self.perc)
        segment = cv2.resize(segment, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

        segm = transforms.ToTensor()(segment)
        return segm

    def standardize_image_and_segment(self, image, segment):
        new_width = int(image.shape[1] * self.perc)
        new_height = int(image.shape[0] * self.perc)
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

    def init_bbox(self, file0, bbox0, init_mask=None):
        frame_observation = self.next(file0)

        image = frame_observation.observed_image
        segments = frame_observation.observed_segmentation

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments

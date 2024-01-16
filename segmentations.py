from typing import Tuple

from abc import abstractmethod, ABC

import os
import sys

import cv2
import torch
import numpy as np

from scipy.ndimage import uniform_filter
from torchvision import transforms
import torch.nn.functional as F

from keyframe_buffer import FrameObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from tracker_config import TrackerConfig
from utils import imread

sys.path.insert(0, 'OSTrack')
from OSTrack.lib.test.tracker.ostrack import OSTrack
from OSTrack.lib.test.parameter.ostrack import parameters


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
    def __init__(self, perc, max_width, feature_extractor, device = torch.device('cuda')):
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


class PrecomputedTracker(BaseTracker):
    def __init__(self, perc, max_width, baseline_dict, feature_extractor):
        super().__init__(perc, max_width, feature_extractor)
        self.perc = perc
        self.max_width = max_width
        self.baseline_dict = baseline_dict
        self.shape = None

        self.background_mdl = np.zeros((1, 65), np.float64)
        self.foreground_mdl = np.zeros((1, 65), np.float64)

    def process(self, image, ind):
        self.shape = image.shape
        if self.max_width / image.shape[1] < self.perc:
            self.perc = self.max_width / image.shape[1]
        segment = cv2.resize(imread(self.baseline_dict[ind]), self.shape[1::-1]).astype(np.float64)
        if len(segment.shape) > 2:
            segment = segment[:, :, :1]

        image, segments = self.standardize_image_and_segment(image, segment)

        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments

    def next(self, file):
        ind = os.path.splitext(os.path.basename(file))[0]
        image = imread(file) * 255
        image, segments = self.process(image, ind)

        segments = pad_image(segments)[None]
        image = pad_image(image)[None]

        return image, segments


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
        rendered_segment_discrete: torch.Tensor = ~(rendered_segmentations < 1)
        rendered_segment_discrete = rendered_segment_discrete.to(rendered_segmentations.dtype)
        return rendered_segment_discrete

    def next(self, frame_id):
        keyframes = [frame_id]
        flow_frames = [frame_id]

        encoder_result, _ = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        rendering_result = self.renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                                 encoder_result.vertices, self.gt_encoder.face_features,
                                                 self.gt_texture, encoder_result.lights)

        image, segment = rendering_result
        image = image.detach().to(self.device)
        image_feat = self.feature_extractor(image)
        segment = segment.detach().to(self.device)

        frame_observation = FrameObservation(observed_image=image, observed_image_features=image_feat,
                                             observed_segmentation=segment)

        return frame_observation

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments


class MyTracker(BaseTracker):
    def __init__(self, perc, max_width):
        feature_extractor = None
        super().__init__(perc, max_width, feature_extractor)
        sys.path.insert(0, './d3s')
        from pytracking.tracker.segm import Segm
        from pytracking.parameter.segm import default_params as vot_params
        params = vot_params.parameters()
        self.tracker = Segm(params)
        self.perc = perc
        self.shape = None
        self.max_width = max_width
        self.background_mdl = np.zeros((1, 65), np.float64)
        self.foreground_mdl = np.zeros((1, 65), np.float64)

    def process(self, image):
        self.shape = image.shape
        if self.max_width / image.shape[1] < self.perc:
            self.perc = self.max_width / image.shape[1]
        segment = cv2.resize(self.tracker.mask, image.shape[1::-1]).astype(np.float64)

        image, segments = self.standardize_image_and_segment(image, segment)
        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        image = imread(file0) * 255
        self.tracker.initialize(image, bbox0, init_mask=init_mask)
        image, segments = self.process(image)
        return image, segments

    def next(self, file):
        image = imread(file) * 255
        image, segments = self.process(image)
        return image, segments


class CSRTrack(BaseTracker):
    def __init__(self, perc, max_width):
        feature_extractor = None
        super().__init__(perc, max_width, feature_extractor)
        self.tracker = cv2.TrackerCSRT_create()
        self.perc = perc
        self.max_width = max_width
        self.background_mdl = np.zeros((1, 65), np.float64)
        self.foreground_mdl = np.zeros((1, 65), np.float64)
        self.shape = None

    def process(self, image, bbox0):
        self.shape = image.shape
        bbox = (bbox0 + np.array([0, 0, bbox0[0], bbox0[1]]))
        if self.max_width / image.shape[1] < self.perc:
            self.perc = self.max_width / image.shape[1]
        segment = np.zeros((image.shape[0], image.shape[1])).astype(np.float64)
        segment[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        image, segments = self.standardize_image_and_segment(image, segment)
        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        image = (imread(file0) * 255).astype(np.uint8)
        bbox = bbox0.astype(int)
        self.tracker.init(image, bbox)
        image, segments = self.process(image, bbox)
        return image, segments

    def next(self, file):
        image = (imread(file) * 255).astype(np.uint8)
        ok, bbox0 = self.tracker.update(image)
        image, segments = self.process(image, bbox0)
        return image, segments


def get_ar(img, init_box, ar_path):
    """ set up Alpha-Refine """
    sys.path.insert(0, './AlphaRefine')
    from pytracking.refine_modules.refine_module import RefineModule
    selector_path = 0
    sr = 2.0
    input_sz = int(128 * sr)  # 2.0 by default
    RF_module = RefineModule(ar_path, selector_path, search_factor=sr, input_sz=input_sz)
    RF_module.initialize(img, np.array(init_box))
    return RF_module


class OSTracker(BaseTracker):
    def __init__(self, perc, max_width):
        feature_extractor = None
        super().__init__(perc, max_width, feature_extractor)
        params = parameters("vitb_384_mae_ce_32x4_ep300")
        params.debug = 0
        params.tracker_name = "ostrack"
        params.param_name = "vitb_384_mae_ce_32x4_ep300"
        self.tracker = OSTrack(params, "video")
        self.perc = perc
        self.max_width = max_width
        self.RF_module = None
        self.shape = None

    def process(self, image, bbox0, segment):
        self.shape = image.shape
        bbox = (bbox0 + np.array([0, 0, bbox0[0], bbox0[1]]))
        if self.max_width / image.shape[1] < self.perc:
            self.perc = self.max_width / image.shape[1]

        image, segments = self.standardize_image_and_segment(image, segment)
        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        image = (imread(file0) * 255).astype(np.uint8)
        bbox = bbox0.astype(int)
        self.tracker.initialize(image, {'init_bbox': bbox})
        self.RF_module = get_ar(image, bbox, "/cluster/home/denysr/scratch/dataset/ostrack/SEcmnet_ep0040-c.pth.tar")
        segment = self.RF_module.get_mask(image, np.array(bbox0))
        image, segments = self.process(image, bbox, segment)
        return image, segments

    def next(self, file):
        image = (imread(file) * 255).astype(np.uint8)
        out = self.tracker.track(image)
        bbox0 = [int(s) for s in out['target_bbox']]
        segment = self.RF_module.get_mask(image, np.array(bbox0))
        image, segments = self.process(image, bbox0, segment)
        return image, segments


def get_bbox(segments):
    all_segments = segments[0, :, 1].sum(0) > 0
    nzeros = torch.nonzero(all_segments, as_tuple=True)
    pix_offset = 20
    x0 = max(0, nzeros[0].min().item() - pix_offset)
    x1 = min(all_segments.shape[0] - 1, nzeros[0].max().item() + pix_offset)
    y0 = max(0, nzeros[1].min().item() - pix_offset)
    y1 = min(all_segments.shape[1] - 1, nzeros[1].max().item() + pix_offset)
    if x1 - x0 > y1 - y0:
        addall = (x1 - x0) - (y1 - y0)
        add0 = int(addall / 2)
        add1 = addall - add0
        y0 = max(0, y0 - add0)
        y1 = min(all_segments.shape[1] - 1, y1 + add1)
    else:
        addall = (y1 - y0) - (x1 - x0)
        add0 = int(addall / 2)
        add1 = addall - add0
        x0 = max(0, x0 - add0)
        x1 = min(all_segments.shape[0] - 1, x1 + add1)
    bounds = [x0, x1, y0, y1]
    return bounds

from typing import Tuple

from abc import abstractmethod, ABC

import os
import sys

import cv2
import torch
import numpy as np

from scipy import ndimage
from scipy.ndimage import uniform_filter
from torchvision import transforms
import torch.nn.functional as F

from tracker_config import TrackerConfig
from utils import imread, normalize_rendered_flows

sys.path.insert(0, 'OSTrack')
from OSTrack.lib.test.tracker.ostrack import OSTrack
from OSTrack.lib.test.parameter.ostrack import parameters


def compute_segments_dist(segment_resized):
    """

    :param segment_resized: np.ndarray of shape [1, H, W]
    :param segment_orig_torch: torch.Tensor of shape [1, H, W]
    :return: Segmentation of shape [1, 2, H, W], where the 1st dimension contains distance to the background in pixels
             and 2nd the segmentation mask as a floating point number in [0.0, 1.0].
    """
    euclid_distance_to_background = ndimage.distance_transform_edt(1 - segment_resized)
    distance_tensor = torch.from_numpy(euclid_distance_to_background)[None]
    segment_resized_torch = torch.from_numpy(segment_resized)[None]
    segments = torch.cat((distance_tensor, segment_resized_torch), 1)
    return segments


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
    def __init__(self, perc, max_width):
        self.perc = perc
        self.max_width = max_width
        self.shape = None

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments, self.perc

    @abstractmethod
    def next(self, file) -> Tuple[torch.Tensor, torch.Tensor]:
        # Implement this method or replace with the actual logic
        # Returns tensors of shape (1, 1, 3, H, W) for image and (1, 1, 1, H, W) for segmentation
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
        segment_orig_torch = transforms.ToTensor()(segment)
        segments = compute_segments_dist(segment_resized[None], segment_orig_torch)
        return image, segments


class PrecomputedTracker(BaseTracker):
    def __init__(self, perc, max_width, baseline_dict):
        super().__init__(perc, max_width)
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

        return image, segments, self.perc

    def next(self, file):
        ind = os.path.splitext(os.path.basename(file))[0]
        image = imread(file) * 255
        image, segments = self.process(image, ind)

        segments = pad_image(segments)[None]
        image = pad_image(image)[None]

        return image, segments


class SyntheticDataGeneratingTracker(BaseTracker):

    def __init__(self, perc, max_width, tracking6d, gt_rotations, gt_translations):
        super().__init__(perc, max_width)
        self.tracking6d = tracking6d
        self.gt_rotations = gt_rotations
        self.gt_translations = gt_translations
        self.encoder_face_features = self.tracking6d.gt_encoder.face_features
        self.gt_texture = self.tracking6d.gt_texture
        self.shape = (self.tracking6d.rendering.width, self.tracking6d.rendering.height, 3)

    def next(self, frame_id):
        keyframes = [frame_id]
        flow_frames = [frame_id]

        encoder_result, _ = self.tracking6d.frames_and_flow_frames_inference(keyframes, flow_frames,
                                                                             encoder_type='gt_encoder')

        rendering_result = self.tracking6d.rendering(encoder_result.translations, encoder_result.quaternions,
                                                     encoder_result.vertices, self.encoder_face_features,
                                                     self.gt_texture, encoder_result.lights)

        image, segment = rendering_result
        image = image.detach()
        segment = segment.detach()
        segment_np = segment.cpu().numpy()

        segments = compute_segments_dist(segment_np[0, 0], segment)[None]
        segments = segments.cuda()
        return image, segments

    def init_bbox(self, file0, bbox0, init_mask=None):
        image, segments = self.next(file0)

        segments = pad_image(segments)
        image = pad_image(image)

        return image, segments, self.perc


class MyTracker(BaseTracker):
    def __init__(self, perc, max_width):
        super().__init__(perc, max_width, None)
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
        return image, segments, self.perc

    def next(self, file):
        image = imread(file) * 255
        image, segments = self.process(image)
        return image, segments


class CSRTrack(BaseTracker):
    def __init__(self, perc, max_width):
        super().__init__(perc, max_width)
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
        return image, segments, self.perc

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
        super().__init__(perc, max_width)
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
        return image, segments, self.perc

    def next(self, file):
        image = (imread(file) * 255).astype(np.uint8)
        out = self.tracker.track(image)
        bbox0 = [int(s) for s in out['target_bbox']]
        segment = self.RF_module.get_mask(image, np.array(bbox0))
        image, segments = self.process(image, bbox0, segment)
        return image, segments


def rle_to_mask(rle, width, height):
    """
    rle: input rle mask encoding
    each evenly-indexed element represents number of consecutive 0s
    each oddly indexed element represents number of consecutive 1s
    width and height are dimensions of the mask
    output: 2-D binary mask
    """
    # allocate list of zeros
    v = [0] * (width * height)

    # set id of the last different element to the beginning of the vector
    idx_ = 0
    for i in range(len(rle)):
        if i % 2 != 0:
            # write as many 1s as RLE says (zeros are already in the vector)
            for j in range(rle[i]):
                v[idx_ + j] = 1
        idx_ += rle[i]

    # reshape vector into 2-D mask
    # return np.reshape(np.array(v, dtype=np.uint8), (height, width)) # numba bug / not supporting np.reshape
    return np.array(v, dtype=np.uint8).reshape((height, width))


def create_mask_from_string(mask_encoding):
    """
    mask_encoding: a string in the following format: x0, y0, w, h, RLE
    output: mask, offset
    mask: 2-D binary mask, size defined in the mask encoding
    offset: (x, y) offset of the mask in the image coordinates
    """
    elements = [int(el) for el in mask_encoding]
    tl_x, tl_y, region_w, region_h = elements[:4]
    rle = np.array([el for el in elements[4:]], dtype=np.int32)

    # create mask from RLE within target region
    mask = rle_to_mask(rle, region_w, region_h)

    return mask, (tl_x, tl_y)


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

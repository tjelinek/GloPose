import sys
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable, List

import cv2
import torch
import imageio
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from auxiliary_scripts.image_utils import resize_and_filter_image, get_shape
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
    def next(self, frame) -> FrameObservation:
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

        self.shape = get_shape(images_paths[0], self.downsample_factor)
        self.images_paths: List[Path] = images_paths
        self.segmentations_paths: List[Path] = segmentations_paths

        self.resize_transform = transforms.Resize((self.shape.height, self.shape.width),
                                                  interpolation=InterpolationMode.NEAREST)

    def next_image(self, frame_i):
        image = imageio.v3.imread(self.images_paths[frame_i])
        image_perm = torch.from_numpy(image).cuda().permute(2, 0, 1)[None].to(torch.float32) / 255.0

        image_downsampled = F.interpolate(image_perm, scale_factor=self.downsample_factor, mode='bilinear',
                                          align_corners=False)[None]
        return image_downsampled

    def next_segmentation(self, frame_i):
        segmentation = imageio.v3.imread(self.segmentations_paths[frame_i])
        if len(segmentation.shape) == 2:
            segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)
        segmentation_p = torch.from_numpy(segmentation).cuda().permute(2, 0, 1)
        segmentation_resized = self.resize_transform(segmentation_p)[None, None, [1]].to(torch.bool).to(torch.float32)

        return segmentation_resized

    def next(self, frame_i):
        image = self.next_image(frame_i)
        image_feat = self.feature_extractor(image).detach()
        segmentation = self.next_segmentation(frame_i)

        frame_observation = FrameObservation(observed_image=image, observed_image_features=image_feat,
                                             observed_segmentation=segmentation)

        return frame_observation


class PrecomputedTrackerSegmentAnything(PrecomputedTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        weights_path = "/mnt/personal/jelint19/weights/SegmentAnything/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=weights_path).cuda()
        self.predictor = SamPredictor(sam)
        # self.mask_generator = SamAutomaticMaskGenerator(sam)

    def next_segmentation(self, frame_i):
        image = self.next_image(frame_i)
        image_np = image.squeeze().permute(1, 2, 0).numpy(force=True)

        from time import time
        start = time()
        self.predictor.set_image(image_np)
        gt_mask = super().next_segmentation(frame_i)
        prompts = torch.nonzero(gt_mask.squeeze()).numpy(force=True)
        point_labels = np.ones_like(prompts[:, 0])
        masks, _, _ = self.predictor.predict(prompts, point_labels, multimask_output=False)
        masks = torch.from_numpy(masks).cuda().to(torch.float32)
        print(f"SAM inference took {time() - start} seconds")

        # masks = self.mask_generator.generate(image_np)
        return masks[None, None]


class PrecomputedTrackerXMem(PrecomputedTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        config = {
            'generic_path': None,  # Set this to the appropriate path if needed
            'split': 'val',
            'output': None,
            'save_all': False,
            'benchmark': False,
            'disable_long_term': False,
            'max_mid_term_frames': 10,
            'min_mid_term_frames': 5,
            'max_long_term_elements': 10000,
            'num_prototypes': 128,
            'top_k': 30,
            'mem_every': 5,
            'deep_update_every': -1,
            'save_scores': False,
            'flip': False,
            'size': 480,
            'enable_long_term': True,
        }

        # Example usage
        vid_length = len(images_paths)

        config['enable_long_term_count_usage'] = (
                not config['disable_long_term'] and
                (vid_length / (config['max_mid_term_frames'] - config['min_mid_term_frames']) * config[
                    'num_prototypes'])
                >= config['max_long_term_elements']
        )

        sys.path.append('repositories/XMem')

        model_path = Path('/mnt/personal/jelint19/weights/XMem/XMem.pth')

        sys.path.append('repositories/XMem')

        from repositories.XMem.inference.inference_core import InferenceCore
        from repositories.XMem.model.network import XMem
        from repositories.XMem.inference.data.mask_mapper import MaskMapper

        network = XMem(config, str(model_path)).cuda().eval()

        self.processor = InferenceCore(network, config=config)
        self.mapper = MaskMapper()

    def next_segmentation(self, frame_i):
        from time import time
        start_time = time()
        image = self.next_image(frame_i).squeeze()

        msk = None
        labels = None
        if frame_i == 0:

            segmentation_init = super().next_segmentation(0)[0, 0]
            image_init = self.next_image(0).squeeze()

            msk, labels = self.mapper.convert_mask(segmentation_init[0].numpy(force=True))
            msk = torch.Tensor(msk).cuda()
            self.processor.set_all_labels(list(self.mapper.remappings.values()))
            self.processor.step(image_init, msk, valid_labels=labels)

        segmentation, pad = self.processor.step(image, msk, valid_labels=labels)
        segmentation = segmentation[None, None, None]

        return segmentation

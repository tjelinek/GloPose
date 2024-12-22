from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import kaolin
import torch
import imageio
import numpy as np
from segment_anything import sam_model_registry, SamPredictor

from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

from auxiliary_scripts.image_utils import resize_and_filter_image, get_shape, ImageShape, get_intrinsics_from_exif
from data_structures.keyframe_buffer import FrameObservation
from models.encoder import Encoder
from models.rendering import RenderingKaolin
from tracker_config import TrackerConfig
from utils import homogenize_3x3_camera_intrinsics


class BaseTracker(ABC):
    def __init__(self, perc, max_width, feature_extractor, device=torch.device('cuda')):
        self.downsample_factor = perc
        self.max_width = max_width
        self.feature_extractor = feature_extractor
        self.image_shape: Optional[ImageShape] = None
        self.device = device

    @abstractmethod
    def next(self, frame) -> FrameObservation:
        pass

    @abstractmethod
    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        pass

    def process_segm(self, img):
        segment = cv2.resize(img, self.image_shape[1::-1]).astype(np.float64)
        width = int(self.image_shape[1] * self.downsample_factor)
        height = int(self.image_shape[0] * self.downsample_factor)
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

    def __init__(self, tracker_config: TrackerConfig, gt_encoder: Encoder, gt_texture,
                 feature_extractor, gt_mesh: kaolin.rep.SurfaceMesh):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width, feature_extractor)
        self.gt_encoder: Encoder = gt_encoder
        self.gt_texture = gt_texture
        self.image_shape = ImageShape(tracker_config.max_width, tracker_config.max_width)
        self.device = tracker_config.device

        faces = gt_mesh.faces
        self.renderer = RenderingKaolin(tracker_config, faces, self.image_shape.width,
                                        self.image_shape.height).to(self.device)

    @staticmethod
    def binary_segmentation_from_rendered_segmentation(rendered_segmentations: torch.Tensor):
        rendered_segment_discrete: torch.Tensor = torch.logical_not(torch.lt(rendered_segmentations, 1.0))
        rendered_segment_discrete = rendered_segment_discrete.to(rendered_segmentations.dtype)
        return rendered_segment_discrete

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.renderer.camera_intrinsics.to(self.device)

    def next(self, frame_id, **kwargs):
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


class PrecomputedTracker(BaseTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width, feature_extractor)

        self.image_shape = get_shape(images_paths[0], self.downsample_factor)
        self.images_paths: List[Path] = images_paths
        self.segmentations_paths: List[Path] = segmentations_paths

        self.resize_transform = transforms.Resize((self.image_shape.height, self.image_shape.width),
                                                  interpolation=InterpolationMode.NEAREST)

    def next_image(self, frame_i):
        image = imageio.v3.imread(self.images_paths[frame_i])
        image_perm = torch.from_numpy(image).cuda().permute(2, 0, 1)[None].to(torch.float32) / 255.0

        image_downsampled = F.interpolate(image_perm, scale_factor=self.downsample_factor, mode='bilinear',
                                          align_corners=False)[None]
        return image_downsampled

    def get_intrinsics_for_frame(self, frame_i):
        return get_intrinsics_from_exif(self.images_paths[frame_i]).to(self.device)

    def next_segmentation(self, frame_i, **kwargs):
        segmentation = imageio.v3.imread(self.segmentations_paths[frame_i])
        if len(segmentation.shape) == 2:
            segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)
        segmentation_p = torch.from_numpy(segmentation).cuda().permute(2, 0, 1)
        segmentation_resized = self.resize_transform(segmentation_p)[None, None, [1]].to(torch.bool).to(torch.float32)

        return segmentation_resized

    def next(self, frame_i, **kwargs):
        image = self.next_image(frame_i)
        image_feat = self.feature_extractor(image).detach()
        segmentation = self.next_segmentation(frame_i)

        frame_observation = FrameObservation(observed_image=image, observed_image_features=image_feat,
                                             observed_segmentation=segmentation)

        return frame_observation


class PrecomputedTrackerSegmentAnythingAbstract(PrecomputedTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        self.predictor: Optional[SamPredictor] = None

    def next_segmentation(self, frame_i, **kwargs):
        image = self.next_image(frame_i)
        image_np = image.squeeze().permute(1, 2, 0).numpy(force=True)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.predictor.set_image(image_np)
            gt_mask = super().next_segmentation(frame_i)
            prompts = torch.nonzero(gt_mask.squeeze()).numpy(force=True)
            point_labels = np.ones_like(prompts[:, 0])
            masks, _, _ = self.predictor.predict(prompts, point_labels, multimask_output=False)
            masks = torch.from_numpy(masks).cuda().to(torch.float32)

        return masks[None, None]


class PrecomputedTrackerSegmentAnything(PrecomputedTrackerSegmentAnythingAbstract):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        weights_path = "/mnt/personal/jelint19/weights/SegmentAnything/sam_vit_h_4b8939.pth"
        sam = sam_model_registry["vit_h"](checkpoint=weights_path).cuda()
        self.predictor = SamPredictor(sam)


class PrecomputedTrackerSegmentAnything2(PrecomputedTrackerSegmentAnything):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2_hiera_large.pt")
        model_cfg = Path("repositories/SAM2/sam2_configs/sam2_hiera_l.yaml")
        model_cfg = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2_configs/sam2_hiera_l.yaml")
        breakpoint()
        self.predictor = SAM2ImagePredictor(build_sam2(str(model_cfg), str(checkpoint)))

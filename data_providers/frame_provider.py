from abc import abstractmethod, ABC
from pathlib import Path
from typing import Callable, List, Optional

import imageio
import kaolin
import numpy as np
import torch
import torch.nn.functional as F
from kornia.image import ImageSize
from segment_anything import SamPredictor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data_structures.keyframe_buffer import FrameObservation
from models.encoder import Encoder, init_gt_encoder
from models.rendering import RenderingKaolin
from tracker_config import TrackerConfig
from utils.general import normalize_vertices
from utils.image_utils import get_shape, get_intrinsics_from_exif


class SyntheticDataProvider:

    def __init__(self, tracker_config: TrackerConfig, gt_texture,
                 gt_mesh: kaolin.rep.SurfaceMesh, gt_rotations: torch.Tensor,
                 gt_translations: torch.Tensor):
        super().__init__(tracker_config.image_downsample)
        self.image_shape = ImageSize(tracker_config.max_width, tracker_config.max_width)
        self.device = tracker_config.device

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self.gt_encoder = init_gt_encoder(gt_mesh, self.gt_texture, self.image_shape, gt_rotations,
                                          gt_translations, tracker_config, self.device)

        self.renderer = RenderingKaolin(tracker_config, faces, self.image_shape.width,
                                        self.image_shape.height).to(self.device)

    def next(self, frame_id) -> FrameObservation:
        keyframes = [frame_id]
        flow_frames = [frame_id]

        encoder_result, _ = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        rendering_result = self.renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                                 encoder_result.vertices, self.gt_encoder.face_features,
                                                 self.gt_texture, encoder_result.lights)

        image = rendering_result.rendered_image
        segment = rendering_result.rendered_image_segmentation

        image = image.detach().to(self.device)
        segment = segment.detach().to(self.device)

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segment)

        return frame_observation


class FrameProvider(ABC):
    def __init__(self, perc, max_width, device=torch.device('cuda')):
        self.downsample_factor = perc
        self.max_width = max_width
        self.image_shape: Optional[ImageSize] = None
        self.device = device

    @abstractmethod
    def next_image(self, frame) -> torch.Tensor:
        pass

    @abstractmethod
    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        pass


class SyntheticFrameProvider(FrameProvider):

    def __init__(self, tracker_config: TrackerConfig, gt_texture,
                 gt_mesh: kaolin.rep.SurfaceMesh, gt_rotations: torch.Tensor,
                 gt_translations: torch.Tensor):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width)
        self.image_shape = ImageSize(tracker_config.max_width, tracker_config.max_width)
        self.device = tracker_config.device

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self.gt_encoder = init_gt_encoder(gt_mesh, self.gt_texture, self.image_shape, gt_rotations,
                                          gt_translations, tracker_config, self.device)

        self.renderer = RenderingKaolin(tracker_config, faces, self.image_shape.width,
                                        self.image_shape.height).to(self.device)

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.renderer.camera_intrinsics.to(self.device)

    def next_image(self, frame_id):
        keyframes = [frame_id]
        flow_frames = [frame_id]

        encoder_result, _ = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        rendering_result = self.renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                                 encoder_result.vertices, self.gt_encoder.face_features,
                                                 self.gt_texture, encoder_result.lights)

        image = rendering_result.rendered_image.squeeze()

        image = image.detach().to(self.device)

        return image


class PrecomputedTracker(FrameProvider):

    def __init__(self, tracker_config: TrackerConfig, images_paths: List[Path]):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width)

        self.image_shape = get_shape(images_paths[0], self.downsample_factor)
        self.images_paths: List[Path] = images_paths

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


class BaseTracker(ABC):
    def __init__(self, perc, max_width, device=torch.device('cuda')):
        self.downsample_factor = perc
        self.max_width = max_width
        self.image_shape: Optional[ImageSize] = None
        self.device = device

        self.frame_provider = FrameProvider(perc, max_width, device)

    @abstractmethod
    def next(self, frame) -> FrameObservation:
        pass

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.frame_provider.get_intrinsics_for_frame(frame_i)


class SyntheticDataGeneratingTracker(BaseTracker):

    def __init__(self, tracker_config: TrackerConfig, gt_texture,
                 gt_mesh: kaolin.rep.SurfaceMesh, gt_rotations: torch.Tensor,
                 gt_translations: torch.Tensor):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width)
        self.image_shape = ImageSize(tracker_config.max_width, tracker_config.max_width)
        self.device = tracker_config.device

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self._init_gt_encoder(gt_mesh, gt_rotations, gt_translations, tracker_config)

        self.renderer = RenderingKaolin(tracker_config, faces, self.image_shape.width,
                                        self.image_shape.height).to(self.device)

    def _init_gt_encoder(self, gt_mesh, gt_rotations, gt_translations, tracker_config):
        ivertices = normalize_vertices(gt_mesh.vertices).numpy()
        iface_features = gt_mesh.uvs[gt_mesh.face_uvs_idx].numpy()
        self.gt_encoder = Encoder(tracker_config, ivertices, iface_features,
                                  self.image_shape.width, self.image_shape.height, 3).to(self.device)
        for name, param in self.gt_encoder.named_parameters():
            if isinstance(param, torch.Tensor):
                param.detach_()
        self.gt_encoder.set_encoder_poses(gt_rotations, gt_translations)
        self.gt_encoder.gt_texture = self.gt_texture

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
        segment = segment.detach().to(self.device)

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segment)

        return frame_observation


class PrecomputedTracker(BaseTracker):

    def __init__(self, tracker_config: TrackerConfig, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config.image_downsample, tracker_config.max_width)

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
        segmentation = self.next_segmentation(frame_i)

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segmentation)

        return frame_observation


class PrecomputedTrackerSegmentAnything2(PrecomputedTracker):

    def __init__(self, tracker_config: TrackerConfig, feature_extractor: Callable, images_paths: List[Path],
                 segmentations_paths: List[Path]):
        super().__init__(tracker_config, feature_extractor, images_paths, segmentations_paths)

        self.predictor: Optional[SamPredictor] = None

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2_hiera_large.pt")
        model_cfg = Path("repositories/SAM2/sam2_configs/sam2_hiera_l.yaml")
        model_cfg = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2_configs/sam2_hiera_l.yaml")
        self.predictor = SAM2ImagePredictor(build_sam2(str(model_cfg), str(checkpoint)))

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

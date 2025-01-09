from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from kornia.image import ImageSize
from segment_anything import SamPredictor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data_structures.keyframe_buffer import FrameObservation
from models.encoder import init_gt_encoder
from tracker_config import TrackerConfig
from utils.image_utils import get_shape, get_intrinsics_from_exif


class SyntheticDataProvider:

    def __init__(self, config: TrackerConfig, gt_texture, gt_mesh, gt_rotations: torch.Tensor,
                 gt_translations: torch.Tensor):
        from models.rendering import RenderingKaolin

        self.image_shape = ImageSize(config.max_width, config.max_width)
        self.device = config.device

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self.gt_encoder = init_gt_encoder(gt_mesh, self.gt_texture, self.image_shape, gt_rotations,
                                          gt_translations, config, self.device)

        self.renderer = RenderingKaolin(config, faces, self.image_shape.width,
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

    def get_intrinsics(self) -> torch.Tensor:
        return self.renderer.camera_intrinsics.to(self.device)


class FrameProvider(ABC):
    def __init__(self, downsample_factor, config: TrackerConfig, device=torch.device('cuda')):
        self.downsample_factor = downsample_factor
        self.image_shape: Optional[ImageSize] = None
        self.device = device
        self.sequence_length: int = config.input_frames

    @abstractmethod
    def next_image(self, frame) -> torch.Tensor:
        pass

    @abstractmethod
    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        pass


class SyntheticFrameProvider(FrameProvider, SyntheticDataProvider):

    def __init__(self, config: TrackerConfig, gt_texture,
                 gt_mesh, gt_rotations: torch.Tensor,
                 gt_translations: torch.Tensor, **kwargs):
        FrameProvider.__init__(self, config.image_downsample, config, config.device)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_rotations, gt_translations)

    def next_image(self, frame_id):
        image = super().next(frame_id).observed_image

        return image

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return super().get_intrinsics()


class PrecomputedFrameProvider(FrameProvider):

    def __init__(self, config: TrackerConfig, images_paths: List[Path], **kwargs):
        super().__init__(config.image_downsample, config)

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


##############################
class SegmentationProvider(ABC):
    def __init__(self, downsample_factor, device='cuda'):
        self.downsample_factor = downsample_factor
        self.image_shape: Optional[ImageSize] = None
        self.device = device

    @abstractmethod
    def next_segmentation(self, frame: int, input_image: torch.Tensor) -> torch.Tensor:
        pass


class SyntheticSegmentationProvider(SegmentationProvider, SyntheticDataProvider):

    def __init__(self, config: TrackerConfig, gt_texture, gt_mesh,
                 gt_rotations: torch.Tensor, gt_translations: torch.Tensor):
        SegmentationProvider.__init__(self, config.image_downsample, config.device)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_rotations, gt_translations)

    def next_segmentation(self, frame_id, **kwargs):
        segmentation = super().next(frame_id).observed_segmentation
        return segmentation


class PrecomputedSegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, images_paths: List[Path], segmentations_paths: List[Path]):
        super().__init__(config.image_downsample, config.device)

        self.image_shape = get_shape(images_paths[0], self.downsample_factor)
        self.segmentations_paths: List[Path] = segmentations_paths

        self.resize_transform = transforms.Resize((self.image_shape.height, self.image_shape.width),
                                                  interpolation=InterpolationMode.NEAREST)

    def next_segmentation(self, frame_i, **kwargs):
        segmentation = imageio.v3.imread(self.segmentations_paths[frame_i])
        if len(segmentation.shape) == 2:
            segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)
        segmentation_p = torch.from_numpy(segmentation).cuda().permute(2, 0, 1)
        segmentation_resized = self.resize_transform(segmentation_p)[None, None, [1]].to(torch.bool).to(torch.float32)

        return segmentation_resized


class SAM2OnlineSegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, initial_segmentation: torch.Tensor,
                 initial_image: torch.Tensor, **kwargs):
        super().__init__(config.image_downsample, config.device)

        assert initial_segmentation is not None

        self.image_shape = ImageSize(width=initial_segmentation.shape[-1], height=initial_segmentation.shape[-2])
        self.predictor: Optional[SamPredictor] = None

        import sys
        sys.path.append('repositories/segment-anything-2-real-time')
        from sam2.build_sam import build_sam2, build_sam2_video_predictor, build_sam2_camera_predictor

        checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2.1_hiera_large.pt")
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        # self.predictor = SAM2ImagePredictor(build_sam2(str(model_cfg), str(checkpoint)))
        self.predictor = build_sam2_camera_predictor(model_cfg, str(checkpoint), device=self.device)

        initial_image_sam_format = self._image_to_sam(initial_image)
        self.predictor.load_first_frame(initial_image_sam_format)

        initial_mask_sam_format = self._mask_to_sam_prompt(initial_segmentation)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(0, 0, initial_mask_sam_format)

    @staticmethod
    def _image_to_sam(image: torch.Tensor) -> np.ndarray:
        return image.squeeze().permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def _mask_to_sam_prompt(mask: torch.Tensor) -> np.ndarray:
        return mask.squeeze().to(torch.bool).numpy(force=True)

    def next_segmentation(self, frame_i, image) -> torch.Tensor:
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            image_sam_format = self._image_to_sam(image)
            out_obj_ids, out_mask_logits = self.predictor.track(image_sam_format)

        obj_seg_mask = out_mask_logits[0, 0] > 0
        obj_seg_mask_formatted = obj_seg_mask[None, None, None].to(torch.float32)
        return obj_seg_mask_formatted


class SAM2SegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, initial_segmentation: torch.Tensor,
                 initial_image: torch.Tensor, **kwargs):
        super().__init__(config.image_downsample, config.device)

        assert initial_segmentation is not None

        self.image_shape = ImageSize(width=initial_image.shape[-1], height=initial_image.shape[-2])
        self.predictor: Optional[SamPredictor] = None

        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        images_paths = kwargs['images_paths']
        saved_imgs_dir = self.save_images_as_jpeg(images_paths, tracker_config.write_folder / 'sam2_imgs')

        checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2.1_hiera_large.pt")
        model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
        self.predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=self.device)

        state = self.predictor.init_state(str(saved_imgs_dir[0].parent))

        initial_mask_sam_format = self._mask_to_sam_prompt(initial_segmentation)
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(state, 0, 0, initial_mask_sam_format)

        self.past_predictions = {0: (out_obj_ids, out_mask_logits)}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
            self.past_predictions[out_frame_idx] = (out_obj_ids, out_mask_logits)

    def save_images_as_jpeg(self, image_paths, output_path: Path) -> List[Path]:
        output_path.mkdir(exist_ok=True)

        saved_img_dirs = []
        for image_path in image_paths:
            with Image.open(image_path) as img:
                img = img.resize((self.image_shape.width, self.image_shape.height), Image.NEAREST)

                output_file = output_path / f"{image_path.stem}.JPEG"
                img.convert("RGB").save(output_file, format="JPEG")
                saved_img_dirs.append(output_file)

        return saved_img_dirs

    @staticmethod
    def _image_to_sam(image: torch.Tensor) -> np.ndarray:
        return image.squeeze().permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def _mask_to_sam_prompt(mask: torch.Tensor) -> np.ndarray:
        return mask.squeeze().to(torch.bool).numpy(force=True)

    def next_segmentation(self, frame_i, image) -> torch.Tensor:

        if frame_i in self.past_predictions.keys():
            out_obj_ids, out_mask_logits = self.past_predictions[frame_i]
        else:
            raise ValueError("Not predicted")

        obj_seg_mask = out_mask_logits[0, 0] > 0
        obj_seg_mask_formatted = obj_seg_mask[None, None, None].to(torch.float32)

        assert obj_seg_mask_formatted.shape[-2] == self.image_shape.height
        assert obj_seg_mask_formatted.shape[-1] == self.image_shape.width

        return obj_seg_mask_formatted

class BaseTracker:
    def __init__(self, config: TrackerConfig, **kwargs):
        self.downsample_factor = config.image_downsample
        self.image_shape: Optional[ImageSize] = None
        self.device = config.device

        self.frame_provider: FrameProvider
        self.segmentation_provider: SegmentationProvider

        if config.frame_provider == 'synthetic':
            self.frame_provider = SyntheticFrameProvider(config, **kwargs)
        elif config.frame_provider == 'precomputed':
            self.frame_provider = PrecomputedFrameProvider(config, **kwargs)
        else:
            raise ValueError(f"Unknown value of 'frame_provider': {config.frame_provider}")

        if config.segmentation_provider == 'synthetic':
            self.segmentation_provider = SyntheticSegmentationProvider(config, **kwargs)
        elif config.segmentation_provider == 'precomputed':
            self.segmentation_provider = PrecomputedSegmentationProvider(config, **kwargs)
        elif config.segmentation_provider == 'SAM2':
            assert 'initial_segmentation' in kwargs
            if config.frame_provider == 'synthetic' and kwargs['initial_segmentation'] is not None:
                synthetic_segment_provider = SyntheticDataProvider(config, **kwargs)
                next_observation = synthetic_segment_provider.next(0)
                initial_segmentation = next_observation.observed_segmentation.squeeze()
                kwargs['initial_segmentation'] = initial_segmentation
            self.segmentation_provider = SAM2SegmentationProvider(config, **kwargs)
        else:
            raise ValueError(f"Unknown value of 'segmentation_provider': {config.segmentation_provider}")

    def next(self, frame_i) -> FrameObservation:
        image = self.frame_provider.next_image(frame_i)

        image_squeezed = image.squeeze()
        segmentation = self.segmentation_provider.next_segmentation(frame_i, image=image_squeezed)

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segmentation)

        return frame_observation

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.frame_provider.get_intrinsics_for_frame(frame_i)

    def get_image_size(self) -> ImageSize:
        return ImageSize(*self.next(0).observed_image.shape[-2:])

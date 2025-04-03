import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from kornia.image import ImageSize
from torchvision import transforms

from data_structures.keyframe_buffer import FrameObservation
from models.encoder import init_gt_encoder
from tracker_config import TrackerConfig
from utils.image_utils import get_target_shape, get_intrinsics_from_exif, get_nth_video_frame, get_video_length_in_frames


class SyntheticDataProvider:

    def __init__(self, config: TrackerConfig, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i, **kwargs):
        from models.rendering import RenderingKaolin

        self.image_shape: ImageSize = config.rendered_image_shape
        self.device = config.device

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self.gt_encoder = init_gt_encoder(gt_mesh, self.gt_texture, gt_Se3_obj1_to_obj_i, self.image_shape, config,
                                          self.device)

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
    def __init__(self, downsample_factor, config: TrackerConfig):
        self.downsample_factor = downsample_factor
        self.image_shape: Optional[ImageSize] = None
        self.device = config.device
        self.sequence_length: int = config.input_frames

    @abstractmethod
    def next_image(self, frame) -> torch.Tensor:
        pass

    @abstractmethod
    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        pass


class SyntheticFrameProvider(FrameProvider, SyntheticDataProvider):

    def __init__(self, config: TrackerConfig, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i, **kwargs):
        FrameProvider.__init__(self, config.image_downsample, config)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i)

    def next_image(self, frame_id):
        image = super().next(frame_id).observed_image

        return image

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return super().get_intrinsics()


class PrecomputedFrameProvider(FrameProvider):

    def __init__(self, config: TrackerConfig, images_paths: Optional[List[Path]], video_path: Optional[Path] = None,
                 **kwargs):
        super().__init__(config.image_downsample, config)

        assert images_paths is not None or video_path is not None
        ref_path = images_paths[0] if images_paths is not None else video_path
        self.image_shape = get_target_shape(ref_path, self.downsample_factor)

        self.images_paths: Optional[List[Path]] = images_paths
        self.video_path: Optional[Path] = video_path

    @staticmethod
    def load_and_downsample_image(image_path: Path, downsample_factor: float, device: str = 'cpu') -> torch.Tensor:
        loaded_image = imageio.v3.imread(image_path)
        image_tensor = transforms.ToTensor()(loaded_image)[None].to(device)

        image_downsampled = PrecomputedFrameProvider.downsample_image(image_tensor, downsample_factor)

        return image_downsampled

    @staticmethod
    def downsample_image(image_tensor, downsample_factor: float) -> torch.Tensor:
        image_downsampled = F.interpolate(image_tensor, scale_factor=downsample_factor, mode='bilinear',
                                          align_corners=False)
        return image_downsampled

    def next_image(self, frame_i) -> torch.Tensor:
        if self.images_paths is not None:
            frame = imageio.v3.imread(self.images_paths[frame_i])
        else:
            frame = get_nth_video_frame(self.video_path, frame_i)

        image_tensor = transforms.ToTensor()(frame)[None].to(self.device)

        image_downsampled = self.downsample_image(image_tensor, self.downsample_factor)

        return image_downsampled

    def get_intrinsics_for_frame(self, frame_i):
        if self.images_paths is not None:
            return get_intrinsics_from_exif(self.images_paths[frame_i]).to(self.device)
        else:  # We can not read it from a video
            raise ValueError("Can not gen cam intrinsics from a video")


##############################
class SegmentationProvider(ABC):
    def __init__(self, image_shape: ImageSize, config: TrackerConfig):
        self.image_shape: ImageSize = image_shape
        self.device = config.device
        self.config = config

    @abstractmethod
    def next_segmentation(self, frame: int, input_image: torch.Tensor) -> torch.Tensor:
        pass

    def get_sequence_length(self):
        return self.config.input_frames


class WhiteSegmentationProvider(SegmentationProvider):
    def __init__(self, config: TrackerConfig, image_shape: ImageSize, **kwargs):
        super().__init__(image_shape, config)

    def next_segmentation(self, frame_i, **kwargs) -> torch.Tensor:
        return torch.ones((1, 1, 1, self.image_shape.height, self.image_shape.width), dtype=torch.float).to(self.device)


class SyntheticSegmentationProvider(SegmentationProvider, SyntheticDataProvider):

    def __init__(self, config: TrackerConfig, image_shape, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i):
        SegmentationProvider.__init__(self, image_shape, config)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i)

    def next_segmentation(self, frame_id, **kwargs):
        segmentation = super().next(frame_id).observed_segmentation
        return segmentation

    def get_sequence_length(self) -> int:
        return self.config.input_frames


class PrecomputedSegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, image_shape: ImageSize, segmentation_paths: List[Path], **kwargs):
        super().__init__(image_shape, config)

        self.image_shape: ImageSize = image_shape
        self.segmentations_paths: List[Path] = segmentation_paths

    def get_sequence_length(self):
        return len(self.segmentations_paths)

    def next_segmentation(self, frame_i, **kwargs):
        return self.load_and_downsample_segmentation(self.segmentations_paths[frame_i], self.image_shape, self.device)

    @staticmethod
    def load_and_downsample_segmentation(segmentation_path: Path, image_size: ImageSize, device: str = 'cpu') \
            -> torch.Tensor:
        # Load segmentation
        segmentation = imageio.v3.imread(segmentation_path)

        if len(segmentation.shape) == 2:
            segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)

        segmentation_p = torch.from_numpy(segmentation).to(device).permute(2, 0, 1)[None]
        segmentation_resized = F.interpolate(segmentation_p, size=[image_size.height, image_size.width], mode='nearest')
        segmentation_mask = segmentation_resized[:, [1]].to(torch.bool).to(torch.float32)

        return segmentation_mask


class SAM2OnlineSegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, image_shape, initial_segmentation: torch.Tensor,
                 initial_image: torch.Tensor, **kwargs):
        super().__init__(image_shape, config)

        assert initial_segmentation is not None

        self.image_shape: ImageSize = image_shape
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
        obj_seg_mask_formatted = obj_seg_mask[None, None].to(torch.float32)
        return obj_seg_mask_formatted


class SAM2SegmentationProvider(SegmentationProvider):

    def __init__(self, config: TrackerConfig, image_shape, initial_segmentation: torch.Tensor,
                 sam2_images_paths: Optional[List[Path]] = None, video_path: Optional[Path] = None,
                 sam2_cache_folder: Optional[Path] = None, **kwargs):
        super().__init__(image_shape, config)

        assert initial_segmentation is not None
        assert sam2_images_paths is not None or video_path is not None

        if sam2_images_paths is not None:
            self.sequence_length = len(sam2_images_paths)
        else:
            self.sequence_length = get_video_length_in_frames(video_path)

        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        self.predictor: Optional[SamPredictor] = None
        self.cache_folder: Optional[Path] = sam2_cache_folder
        if self.cache_folder is not None:
            if self.cache_folder.exists() and config.purge_cache:
                shutil.rmtree(self.cache_folder)
            self.cache_folder.mkdir(exist_ok=True, parents=True)

        if self.cache_folder is not None:
            if sam2_images_paths is not None:
                self.cache_paths: List[Path] = [self.cache_folder / (img_path.stem + '.pt')
                                                for img_path in sam2_images_paths]
            else:
                self.cache_paths: List[Path] = [self.cache_folder / f'{video_path.stem}_{i}.pt'
                                                for i in range(self.sequence_length)]
            self.cache_exists: bool = all(x.exists() for x in self.cache_paths)
        else:
            self.cache_exists = False

        self.past_predictions = {}
        if not self.cache_exists:

            checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2.1_hiera_large.pt")
            model_cfg = 'configs/sam2.1/sam2.1_hiera_l.yaml'
            self.predictor = build_sam2_video_predictor(model_cfg, str(checkpoint), device=self.device)

            if sam2_images_paths is not None:
                state = self.predictor.init_state(str(sam2_images_paths[0].parent),
                                                  offload_video_to_cpu=True,
                                                  offload_state_to_cpu=True,
                                                  )
            else:
                state = self.predictor.init_state(str(video_path), offload_video_to_cpu=True,
                                                  offload_state_to_cpu=True)

            initial_mask_sam_format = self._mask_to_sam_prompt(initial_segmentation)
            out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(state, 0, 0,
                                                                                      initial_mask_sam_format)

            self.past_predictions = {0: (out_obj_ids, out_mask_logits)}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(state):
                self.past_predictions[out_frame_idx] = (out_frame_idx, out_obj_ids, out_mask_logits)
                if self.cache_folder is not None:
                    torch.save((out_frame_idx, out_obj_ids, out_mask_logits), self.cache_paths[out_frame_idx])
        else:
            for cache_path in self.cache_paths:
                out_frame_idx, out_obj_ids, out_mask_logits = torch.load(cache_path, weights_only=True,
                                                                         map_location=self.device)
                self.past_predictions[out_frame_idx] = (out_frame_idx, out_obj_ids, out_mask_logits)

    def get_sequence_length(self):
        return self.sequence_length

    @staticmethod
    def _image_to_sam(image: torch.Tensor) -> np.ndarray:
        return image.squeeze().permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def _mask_to_sam_prompt(mask: torch.Tensor) -> np.ndarray:
        return mask.squeeze().to(torch.bool).numpy(force=True)

    def next_segmentation(self, frame_i, image) -> torch.Tensor:

        if frame_i in self.past_predictions.keys():
            out_frame_idx, out_obj_ids, out_mask_logits = self.past_predictions[frame_i]
        else:
            raise ValueError("Not predicted")

        obj_seg_mask = out_mask_logits[0, 0] > 0
        obj_seg_mask_formatted = obj_seg_mask[None, None].to(torch.float32)

        assert obj_seg_mask_formatted.shape[-2] == self.image_shape.height
        assert obj_seg_mask_formatted.shape[-1] == self.image_shape.width

        return obj_seg_mask_formatted


class BaseTracker:
    def __init__(self, config: TrackerConfig, **kwargs):
        self.downsample_factor = config.image_downsample
        self.image_shape: Optional[ImageSize] = None
        self.device = config.device
        self.black_background: bool = config.black_background

        self.frame_provider: FrameProvider
        self.segmentation_provider: SegmentationProvider

        if config.frame_provider == 'synthetic':
            self.frame_provider = SyntheticFrameProvider(config, **kwargs)
        elif config.frame_provider == 'precomputed':
            self.frame_provider = PrecomputedFrameProvider(config, **kwargs)
        else:
            raise ValueError(f"Unknown value of 'frame_provider': {config.frame_provider}")

        self.image_shape: ImageSize = self.frame_provider.image_shape

        if config.segmentation_provider == 'synthetic':
            self.segmentation_provider = SyntheticSegmentationProvider(config, self.image_shape, **kwargs)
        elif config.segmentation_provider == 'precomputed':
            self.segmentation_provider = PrecomputedSegmentationProvider(config, self.image_shape, **kwargs)
        elif config.segmentation_provider == 'whites':
            self.segmentation_provider = WhiteSegmentationProvider(config, self.image_shape, **kwargs)
        elif config.segmentation_provider == 'SAM2':
            sam2_tmp_path = config.write_folder / 'sam2_imgs'

            need_to_delete_cache = False
            if kwargs.get('images_paths'):
                images_paths = kwargs['images_paths']
                images_paths_for_sam = self.save_images_as_jpeg(sam2_tmp_path, self.frame_provider, images_paths)
                need_to_delete_cache = True
            else:
                images_paths_for_sam = None
                assert kwargs.get('video_path')

            if config.frame_provider == 'synthetic':  # and kwargs['initial_segmentation'] is not None:
                synthetic_segment_provider = SyntheticDataProvider(config, **kwargs)
                next_observation = synthetic_segment_provider.next(0)
                initial_segmentation = next_observation.observed_segmentation.squeeze()
                kwargs['initial_segmentation'] = initial_segmentation
            else:
                assert 'initial_segmentation' in kwargs and kwargs['initial_segmentation'] is not None
            initial_segmentation = kwargs['initial_segmentation']
            del kwargs['initial_segmentation']
            self.segmentation_provider = SAM2SegmentationProvider(config, self.image_shape, initial_segmentation,
                                                                  sam2_images_paths=images_paths_for_sam, **kwargs)
            if need_to_delete_cache:
                shutil.rmtree(sam2_tmp_path)
        else:
            raise ValueError(f"Unknown value of 'segmentation_provider': {config.segmentation_provider}")

    def save_images_as_jpeg(self, output_path: Path, frame_provider: FrameProvider,
                            images_paths: Optional[List[Path]] = None) -> List[Path]:
        output_path.mkdir(exist_ok=True)
        transform_to_pil = transforms.ToPILImage()

        saved_img_paths = []
        for frame_i in range(frame_provider.sequence_length):
            img = frame_provider.next_image(frame_i).squeeze()

            img = transform_to_pil(img)
            img = img.resize((self.image_shape.width, self.image_shape.height), Image.NEAREST)

            # Define the output file name
            if images_paths is not None:
                output_file = output_path / f"{frame_i:05d}_{Path(images_paths[frame_i]).stem}.JPEG"
            else:
                output_file = output_path / f"{frame_i:05d}.JPEG"

            # Save the image in JPEG format
            img.convert("RGB").save(output_file, format="JPEG")
            saved_img_paths.append(output_file)

        return saved_img_paths

    def next(self, frame_i) -> FrameObservation:
        image = self.frame_provider.next_image(frame_i)

        image_squeezed = image.squeeze()
        segmentation = self.segmentation_provider.next_segmentation(frame_i, image=image_squeezed)

        if self.black_background:
            image = image * segmentation

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segmentation)

        return frame_observation

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.frame_provider.get_intrinsics_for_frame(frame_i)

    def get_image_size(self) -> ImageSize:
        return ImageSize(*self.next(0).observed_image.shape[-2:])

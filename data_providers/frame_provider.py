import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import List, Optional, Union

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from kornia.image import ImageSize
from torchvision import transforms
from torchvision.io import decode_image, ImageReadMode

from configs.glopose_config import GloPoseConfig
from data_structures.keyframe_buffer import FrameObservation
from utils.data_utils import get_scale_from_meter, is_video_input
from utils.general import erode_segment_mask2
from utils.image_utils import get_target_shape, get_intrinsics_from_exif, get_nth_video_frame, get_video_length


class SyntheticDataProvider:

    def __init__(self, config: GloPoseConfig, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i, **kwargs):
        from models.rendering import RenderingKaolin
        from models.encoder import init_gt_encoder

        self.image_shape: ImageSize = config.renderer.rendered_image_shape
        self.device = config.run.device
        self.num_frames = config.input.input_frames

        faces = gt_mesh.faces
        self.gt_texture = gt_texture

        self.gt_encoder = init_gt_encoder(gt_mesh, self.gt_texture, gt_Se3_obj1_to_obj_i, self.image_shape, config,
                                          self.device)

        self.renderer = RenderingKaolin(config, faces, self.image_shape.width,
                                        self.image_shape.height).to(self.device)

    def next_synthetic_observation(self, frame_id) -> FrameObservation:
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
    def __init__(self, downsample_factor, sequence_length: int = None, skip_indices: int = 1, device: str = 'cpu'):
        self.downsample_factor = downsample_factor
        self.image_shape: Optional[ImageSize] = None
        self.device = device
        self.sequence_length: int = sequence_length if sequence_length is not None else self.get_input_length()
        self.skip_indices: int = skip_indices

    @abstractmethod
    def get_input_length(self):
        pass

    @abstractmethod
    def next_image(self, frame) -> torch.Tensor:
        pass

    @abstractmethod
    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        pass

    @abstractmethod
    def get_n_th_image_name(self, frame_i: int) -> Path:
        pass

    def save_images(self, output_path: Path, only_frame_index: bool = False, progress=None) -> \
            List[Path]:
        output_path.mkdir(exist_ok=True)
        transform_to_pil = transforms.ToPILImage()

        saved_img_paths = []
        for frame_i in range(0, self.sequence_length):
            if progress is not None:
                progress(frame_i / float(self.sequence_length), desc="Caching SAM2 images...")
            img = self.next_image(frame_i).squeeze()

            img = transform_to_pil(img)
            img = img.resize((self.image_shape.width, self.image_shape.height), Image.NEAREST)

            # if images_paths is not None:
            #     output_file = output_path / (f"{frame_i * self.skip_indices:05d}_"
            #                                  f"{Path(images_paths[frame_i * self.skip_indices]).stem}.jpg")
            # else:
            #     output_file = output_path / f"{frame_i * self.skip_indices:05d}.jpg"

            # Define the output file name
            if only_frame_index:
                output_file = output_path / f"{frame_i * self.skip_indices:05d}.jpg"
            else:
                output_file = output_path / (f"{frame_i * self.skip_indices:05d}_"
                                             f"{self.get_n_th_image_name(frame_i * self.skip_indices).stem}.jpg")

            print(f'Cached SAM2 file {output_file}')

            # Save the image in JPG format
            img.convert("RGB").save(output_file, format="JPEG")
            saved_img_paths.append(output_file)

        return saved_img_paths


class SyntheticFrameProvider(FrameProvider, SyntheticDataProvider):

    def __init__(self, config: GloPoseConfig, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i, **kwargs):
        FrameProvider.__init__(self, config.input.image_downsample, config.input.input_frames,
                               config.input.skip_indices, config.run.device)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i)

    def next_image(self, frame_id):
        image = super().next_synthetic_observation(frame_id * self.skip_indices).observed_image

        return image

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return super().get_intrinsics()

    def get_n_th_image_name(self, frame_i: int) -> Path:
        return Path(f"{frame_i * self.skip_indices}.png")

    def get_input_length(self):
        assert super(SyntheticDataProvider).num_frames


class PrecomputedFrameProvider(FrameProvider):

    def __init__(self, input_images: Union[List[Path], Path], num_frames: Optional[int] = None,
                 image_downsample: float = 1.0, skip_indices: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(image_downsample, num_frames, skip_indices, device)

        assert type(input_images) is list or isinstance(input_images, Path)

        self.input_are_images: bool = type(input_images) is list
        if self.sequence_length is None:
            if self.input_are_images:
                self.sequence_length = len(input_images)
            else:
                assert is_video_input(input_images)
                self.sequence_length = get_video_length(input_images)

        ref_path = input_images[0] if self.input_are_images else input_images
        self.image_shape = get_target_shape(ref_path, self.downsample_factor)

        self.input_images: Union[List[Path], Path] = input_images

    def get_n_th_image_name(self, frame_i: int) -> Path:
        if self.input_are_images:
            return Path(self.input_images[frame_i * self.skip_indices].name)
        else:
            return Path(f"{self.input_images.stem}_{frame_i * self.skip_indices}.png")

    @staticmethod
    def load_and_downsample_image(img_path: Path, downsample_factor: float = 1.0, device: str = 'cpu') -> torch.Tensor:
        loaded_image = imageio.v3.imread(img_path)
        image_tensor = transforms.ToTensor()(loaded_image)[None, :3].to(device)  # RGBA -> RGB

        image_downsampled = PrecomputedFrameProvider.downsample_image(image_tensor, downsample_factor)

        return image_downsampled

    @staticmethod
    def downsample_image(image_tensor, downsample_factor: float) -> torch.Tensor:
        image_downsampled = F.interpolate(image_tensor, scale_factor=downsample_factor, mode='bilinear',
                                          align_corners=False)
        return image_downsampled

    def get_input_length(self):
        if self.input_are_images:
            # For image list, return the number of images
            return len(self.input_images) // self.skip_indices
        else:
            # For video, return the total frame count
            return get_video_length(self.input_images) // self.skip_indices

    def next_image(self, frame_i) -> torch.Tensor:
        if self.input_are_images:
            frame = imageio.v3.imread(self.input_images[frame_i * self.skip_indices])
        else:
            frame = get_nth_video_frame(self.input_images, frame_i * self.skip_indices)

        image_tensor = transforms.ToTensor()(frame)[None, :3].to(self.device)  # RGBA -> RGB
        image_downsampled = self.downsample_image(image_tensor, self.downsample_factor)

        return image_downsampled

    def next_image_255(self, frame_i) -> torch.Tensor:
        if self.input_are_images is not None:
            frame = decode_image(str(self.input_images[frame_i * self.skip_indices]), mode=ImageReadMode.UNCHANGED)
        else:
            raise NotImplementedError()

        image_tensor = frame[None].to(self.device)

        image_downsampled = self.downsample_image(image_tensor, self.downsample_factor)

        return image_downsampled

    def get_intrinsics_for_frame(self, frame_i):
        if self.input_are_images is not None:
            return get_intrinsics_from_exif(self.input_images[frame_i]).to(self.device)
        else:  # We can not read it from a video
            raise ValueError("Can not gen cam intrinsics from a video")


##############################
class SegmentationProvider(ABC):
    def __init__(self, image_shape: ImageSize, device: str = 'cpu'):
        self.image_shape: ImageSize = image_shape
        self.device = device

    @abstractmethod
    def next_segmentation(self, frame: int, input_image: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        pass


class WhiteSegmentationProvider(SegmentationProvider):

    def __init__(self, image_shape: ImageSize, skip_indices: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(image_shape, device)
        self.skip_indices = skip_indices

    def next_segmentation(self, frame_i, **kwargs) -> torch.Tensor:
        return torch.ones((1, 1, self.image_shape.height, self.image_shape.width), dtype=torch.float).to(self.device)

    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        return Path(f"{frame_i * self.skip_indices}.png")


class SyntheticSegmentationProvider(SegmentationProvider, SyntheticDataProvider):

    def __init__(self, config: GloPoseConfig, image_shape, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i):
        SegmentationProvider.__init__(self, image_shape, config.run.device)
        SyntheticDataProvider.__init__(self, config, gt_texture, gt_mesh, gt_Se3_obj1_to_obj_i)
        self.skip_indices = config.input.skip_indices

    def next_segmentation(self, frame_id, **kwargs):
        segmentation = super().next_synthetic_observation(frame_id * self.skip_indices).observed_segmentation
        return segmentation

    def get_sequence_length(self) -> int:
        return super(SyntheticDataProvider).num_frames

    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        return Path(f"{frame_i * self.skip_indices}.png")


class PrecomputedSegmentationProvider(SegmentationProvider):

    def __init__(self, image_shape: ImageSize, input_segmentations: List[Path] | Path, segmentation_channel: int = 0,
                 skip_indices: int = 1, device: str = 'cpu', **kwargs):
        super().__init__(image_shape, device)

        self.image_shape: ImageSize = image_shape
        if type(input_segmentations) is not list:
            raise NotImplementedError("Can't work with segmentation video yet")
        self.input_segmentations: List[Path] = input_segmentations
        self.segmentation_channel = segmentation_channel
        self.skip_indices = skip_indices

    def get_sequence_length(self):
        return len(self.input_segmentations)

    def next_segmentation(self, frame_i, **kwargs):
        return self.load_and_downsample_segmentation(self.input_segmentations[frame_i * self.skip_indices],
                                                     self.image_shape, self.segmentation_channel, device=self.device)

    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        return self.input_segmentations[frame_i * self.skip_indices]

    @staticmethod
    def get_initial_segmentation(input_images: List[Path] | Path = None, input_segmentations: List[Path] | Path = None,
                                 segmentation_channel=0, image_downsample: float = 1.,
                                 device: str = 'cpu') -> torch.Tensor:

        image_shape = get_target_shape(input_images[0], image_downsample)

        segmentation_provider = PrecomputedSegmentationProvider(image_shape, input_segmentations, device=device,
                                                                segmentation_channel=segmentation_channel)

        first_segment_tensor = segmentation_provider.next_segmentation(0).squeeze()

        return first_segment_tensor

    @staticmethod
    def load_and_downsample_segmentation(segmentation_path: Path, image_size: ImageSize, segmentation_channel: int = 0,
                                         device: str = 'cpu') \
            -> torch.Tensor:
        # Load segmentation
        segmentation = imageio.v3.imread(segmentation_path)

        if len(segmentation.shape) == 2:
            segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)

        segmentation_p = torch.from_numpy(segmentation).to(device).permute(2, 0, 1)[None]
        segmentation_resized = F.interpolate(segmentation_p, size=[image_size.height, image_size.width], mode='nearest')
        segmentation_mask = segmentation_resized[:, [segmentation_channel]].to(torch.bool).to(torch.float32)

        return segmentation_mask


class SAM2SegmentationProvider(SegmentationProvider):

    def __init__(self, config: GloPoseConfig, image_shape, initial_segmentation: torch.Tensor,
                 image_provider: FrameProvider, write_folder: Path,
                 sam2_cache_folder: Path, progress=None, **kwargs):
        super().__init__(image_shape, config.run.device)

        assert initial_segmentation is not None

        self.sequence_length = image_provider.sequence_length
        self.skip_indices = config.input.skip_indices

        from sam2.build_sam import build_sam2_video_predictor
        import sam2

        self.predictor: Optional[SamPredictor] = None
        self.cache_folder: Optional[Path] = sam2_cache_folder

        if self.cache_folder is not None:

            if self.cache_folder.exists() and config.paths.purge_cache:
                shutil.rmtree(self.cache_folder)
            self.cache_folder.mkdir(exist_ok=True, parents=True)

            self.cache_paths: List[Path] = [self.cache_folder / f'{i * self.skip_indices:05d}.pt'
                                            for i in range(self.sequence_length)]

            self.cache_exists: bool = all(x.exists() for x in self.cache_paths)
        else:
            self.cache_exists = False

        self.past_predictions = {}

        if not self.cache_exists:

            checkpoint = Path("/mnt/personal/jelint19/weights/SegmentAnything2/sam2.1_hiera_large.pt")
            cfg_dir = Path(sam2.__file__).parent / "configs"
            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_dir(config_dir=str(cfg_dir.resolve()), version_base=None, job_name="sam2"):
                self.predictor = build_sam2_video_predictor("sam2.1/sam2.1_hiera_l.yaml", checkpoint,
                                                            device=self.device)

            sam2_tmp_path = write_folder / 'sam2_imgs'
            sam2_tmp_path.mkdir(exist_ok=True, parents=True)

            image_provider.save_images(sam2_tmp_path, True, progress)

            state = self.predictor.init_state(str(sam2_tmp_path),
                                              offload_video_to_cpu=True,
                                              offload_state_to_cpu=True,
                                              )

            initial_mask_sam_format = self._mask_to_sam_prompt(initial_segmentation)
            out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(state, 0, 0,
                                                                                      initial_mask_sam_format)

            self.past_predictions = {0: (0, out_obj_ids, out_mask_logits)}

            for i, (_, out_obj_ids, out_mask_logits) in enumerate(self.predictor.propagate_in_video(state)):

                if progress is not None:
                    progress(i / float(image_provider.sequence_length), desc="SAM2 image tracking...")

                out_frame_idx = i * self.skip_indices
                self.past_predictions[out_frame_idx] = (out_frame_idx, out_obj_ids, out_mask_logits)
                if self.cache_folder is not None:
                    torch.save((out_frame_idx, out_obj_ids, out_mask_logits), self.cache_paths[i])

            shutil.rmtree(sam2_tmp_path)
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

        if frame_i * self.skip_indices in self.past_predictions.keys():
            out_frame_idx, out_obj_ids, out_mask_logits = self.past_predictions[frame_i * self.skip_indices]
        else:
            raise ValueError("Not predicted")

        obj_seg_mask = out_mask_logits[0, 0] > 0
        obj_seg_mask_formatted = obj_seg_mask[None, None].to(torch.float32)

        segmentation_resized = F.interpolate(obj_seg_mask_formatted, size=[self.image_shape.height,
                                                                           self.image_shape.width], mode='nearest')

        assert segmentation_resized.shape[-2] == self.image_shape.height
        assert segmentation_resized.shape[-1] == self.image_shape.width

        return segmentation_resized

    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        return Path(f"{frame_i * self.skip_indices}.png")


class FrameProviderAll:
    def __init__(self, config: GloPoseConfig, **kwargs):
        self.downsample_factor = config.input.image_downsample
        self.image_shape: Optional[ImageSize] = None
        self.device = config.run.device
        self.config = config.input.frame_provider_config

        self.frame_provider: FrameProvider
        self.segmentation_provider: SegmentationProvider

        if config.input.frame_provider == 'synthetic':
            self.frame_provider = SyntheticFrameProvider(config, **kwargs)
        elif config.input.frame_provider == 'precomputed':
            self.frame_provider = PrecomputedFrameProvider(num_frames=config.input.input_frames,
                                                           image_downsample=config.input.image_downsample,
                                                           skip_indices=config.input.skip_indices,
                                                           device=config.run.device, **kwargs)
        else:
            raise ValueError(f"Unknown value of 'frame_provider': {config.input.frame_provider}")

        self.image_shape: ImageSize = self.frame_provider.image_shape

        if kwargs.get('depth_paths') is not None:
            if config.run.dataset == 'HO3D':
                self.depth_provider = PrecomputedDepthProvider_HO3D(config, self.image_shape, **kwargs)
            else:
                self.depth_provider = PrecomputedDepthProvider(config, self.image_shape,
                                                               depth_scale_to_meter=config.input.depth_scale_to_meter,
                                                               **kwargs)
        else:
            self.depth_provider = None

        if config.input.segmentation_provider == 'synthetic':
            self.segmentation_provider = SyntheticSegmentationProvider(config, self.image_shape, **kwargs)
        elif config.input.segmentation_provider == 'precomputed':
            self.segmentation_provider = PrecomputedSegmentationProvider(device=config.run.device,
                                                                         image_shpape=self.image_shape, **kwargs)
        elif config.input.segmentation_provider == 'whites':
            self.segmentation_provider = WhiteSegmentationProvider(self.image_shape, config.input.skip_indices,
                                                                   config.run.device,
                                                                   **kwargs)
        elif config.input.segmentation_provider == 'SAM2':

            if config.input.frame_provider == 'synthetic':  # and kwargs['initial_segmentation'] is not None:
                synthetic_segment_provider = SyntheticDataProvider(config, **kwargs)
                next_observation = synthetic_segment_provider.next_synthetic_observation(0)
                initial_segmentation = next_observation.observed_segmentation.squeeze()
                kwargs['initial_segmentation'] = initial_segmentation
            else:
                assert 'initial_segmentation' in kwargs and kwargs['initial_segmentation'] is not None
            initial_segmentation = kwargs['initial_segmentation']
            del kwargs['initial_segmentation']
            self.segmentation_provider = SAM2SegmentationProvider(config, self.image_shape, initial_segmentation,
                                                                  self.frame_provider,
                                                                  **kwargs)
        else:
            raise ValueError(f"Unknown value of 'segmentation_provider': {config.input.segmentation_provider}")

    def get_n_th_image_name(self, frame_i: int) -> Path:
        return self.frame_provider.get_n_th_image_name(frame_i)

    def get_n_th_segmentation_name(self, frame_i: int) -> Path:
        return self.segmentation_provider.get_n_th_segmentation_name(frame_i)

    def next(self, frame_i) -> FrameObservation:
        image = self.frame_provider.next_image(frame_i)

        image_squeezed = image.squeeze()
        segmentation = self.segmentation_provider.next_segmentation(frame_i, image=image_squeezed)

        if self.config.erode_segmentation:
            segmentation = erode_segment_mask2(self.config.erode_segmentation_iters, segmentation)

        if self.config.black_background:
            image = image * segmentation

        depth = None
        if self.depth_provider is not None:
            depth = self.depth_provider.next_depth(frame_i, input_image=image)

        assert image.shape[-2:] == segmentation.shape[-2:]
        if depth is not None:
            assert image.shape[-2:] == depth.shape[-2:]

        frame_observation = FrameObservation(observed_image=image, observed_segmentation=segmentation,
                                             depth=depth)

        return frame_observation

    def get_intrinsics_for_frame(self, frame_i: int) -> torch.Tensor:
        return self.frame_provider.get_intrinsics_for_frame(frame_i)

    def get_image_size(self) -> ImageSize:
        return ImageSize(*self.next(0).observed_image.shape[-2:])


##############################
class DepthProvider(ABC):
    def __init__(self, image_shape: ImageSize, config: GloPoseConfig):
        self.image_shape: ImageSize = image_shape
        self.device = config.run.device
        self.config = config

    @abstractmethod
    def next_depth(self, frame: int, input_image: torch.Tensor) -> torch.Tensor:
        pass

    def get_sequence_length(self):
        return self.config.input.input_frames


class PrecomputedDepthProvider(DepthProvider):

    def __init__(self, config: GloPoseConfig, image_shape: ImageSize, depth_paths: List[Path],
                 depth_scale_to_meter: float = 1.0, output_unit: str = None, **kwargs):
        super().__init__(image_shape, config)

        self.image_shape: ImageSize = image_shape
        self.depth_paths: List[Path] = depth_paths

        self.depth_scale_to_meter: float = depth_scale_to_meter
        if output_unit is not None:
            self.conversion_scale = get_scale_from_meter(output_unit)
        else:
            self.conversion_scale = 1.0

        self.skip_indices = config.input.skip_indices

    def get_sequence_length(self):
        return len(self.depth_paths)

    def next_depth(self, frame_i, **kwargs):
        depth_path = self.depth_paths[frame_i * self.skip_indices]
        depth_tensor = self.load_and_downsample_depth(depth_path, self.image_shape, self.device)
        return self.depth_scale_to_meter * self.conversion_scale * depth_tensor

    @staticmethod
    def load_and_downsample_depth(depth_path: Path, image_size: ImageSize, device: str = 'cpu') -> torch.Tensor:
        # Load depth
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

        depth_p = torch.from_numpy(depth).to(device)[None, None].to(torch.float32)
        depth_resized = F.interpolate(depth_p, size=[image_size.height, image_size.width], mode='bilinear')

        return depth_resized


class PrecomputedDepthProvider_HO3D(PrecomputedDepthProvider):

    @staticmethod
    def load_and_downsample_depth(depth_path: Path, image_size: ImageSize, device: str = 'cpu') -> torch.Tensor:
        # Taken from
        # https://github.com/shreyashampali/ho3d/blob/c1c8923f2f90fc2ec7c502f491b3851e2c58c388/vis_pcl_all_cameras.py#L60

        depth_scale = 0.00012498664727900177
        depth_img = cv2.imread(str(depth_path))

        dpt = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
        dpt = dpt * depth_scale

        depth_p = torch.from_numpy(dpt).to(device)[None, None].to(torch.float32)
        depth_resized = F.interpolate(depth_p, size=[image_size.height, image_size.width], mode='bilinear')

        return depth_resized

import math
from collections import namedtuple
from dataclasses import dataclass

import copy
import imageio
import kaolin
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from kornia.geometry.conversions import QuaternionCoeffOrder, angle_axis_to_quaternion
from pathlib import Path
from torch.optim import lr_scheduler
from torchvision.utils import save_image
from typing import List

from OSTrack.S2DNet.s2dnet import S2DNet
from auxiliary_scripts.logging import visualize_flow, WriteResults, visualize_theoretical_flow, load_gt_annotations_file
from flow import get_flow_from_images, get_flow_from_images_mft
from flow_gma import get_flow_model_gma
from flow_mft import get_flow_model_mft
from flow_raft import get_flow_model_raft
from helpers.torch_helpers import write_renders
from main_settings import g_ext_folder
from models.encoder import Encoder, EncoderResult
from models.initial_mesh import generate_face_features
from models.kaolin_wrapper import load_obj
from models.loss import FMOLoss
from models.rendering import RenderingKaolin
from optimization import lsq_gna_custom, lsq_lma_custom
from segmentations import PrecomputedTracker, CSRTrack, OSTracker, MyTracker, get_bbox, SyntheticDataGeneratingTracker
from utils import consecutive_quaternions_angular_difference, normalize_vertices


@dataclass
class TrackerConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # General settings
    tracker_type: str = 'ostrack'
    features: str = 'deep'
    verbose: bool = True
    write_results: bool = True
    write_intermediate: bool = True
    visualize_loss_landscape: bool = True
    render_just_bounding_box: bool = False
    training_print_status_frequency = 1

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 0
    keyframes: int = None
    all_frames_keyframes: bool = False
    fmo_steps: int = 1
    stochastically_add_keyframes: bool = False

    # Shape settings
    shapes: List = list
    init_shape: str = 'sphere'
    predict_vertices: bool = None

    # Mesh settings
    mesh_size: int = None
    mesh_normalize: bool = None
    texture_size: int = None
    use_lights: bool = None

    # Camera settings
    camera_distance: float = None
    max_width: int = None
    image_downsample: float = None

    # Learning rates
    learning_rate: float = None
    quaternion_learning_rate_coef: float = 1.0
    translation_learning_rate_coef: float = 1.0

    # Tracking settings
    tran_init: float = None
    rot_init: List[float] = None
    inc_step: float = None
    iterations: int = None
    stop_value: float = None
    rgb_iters: int = None
    project_coin: bool = None
    connect_frames: bool = None
    accumulate: bool = None
    weight_by_gradient: bool = None
    mot_opt_all: bool = None
    motion_only_last: bool = None

    # Loss function coefficients
    loss_laplacian_weight: float = None
    loss_tv_weight: float = None
    loss_iou_weight: float = None
    loss_dist_weight: float = None
    loss_q_weight: float = None
    loss_texture_change_weight: float = None
    loss_t_weight: float = None
    loss_rgb_weight: float = None
    loss_flow_weight: float = None
    loss_fl_obs_and_rend_weight: float = None
    loss_fl_not_obs_rend_weight: float = None
    loss_fl_obs_not_rend_weight: float = None

    # Additional settings
    sigmainv: float = None
    factor: float = None
    mask_iou_th: float = None
    erode_renderer_mask: int = None
    rotation_divide: int = None
    sequence: str = None

    # Ground truths
    gt_texture: str = None
    gt_mesh_prototype: str = None
    gt_tracking_log: str = None
    use_gt: bool = False

    # Optimization
    allow_break_sgd_after = 30
    break_sgd_after_iters_with_no_change = 20
    optimize_non_positional_params_after = 70
    use_lr_scheduler = False
    lr_scheduler_patience = 5

    # Optical flow loss
    flow_model: str = 'RAFT'  # 'RAFT' 'GMA' and 'MFT'
    segmentation_mask_erosion_iters: int = 0
    coordinate_descent: bool = True
    flow_sgd: bool = False
    flow_sgd_n_samples: int = 100


@dataclass
class KeyframeBuffer:
    keyframes: list = None
    flow_keyframes: list = None
    images: torch.Tensor = None
    prev_images: torch.Tensor = None
    images_feat: torch.Tensor = None
    segments: torch.Tensor = None
    observed_flows: torch.Tensor = None
    flow_segment_masks: torch.Tensor = None

    @staticmethod
    def merge(buffer1, buffer2):
        """
        Concatenates and merges two KeyframeBuffer instances while sorting the keyframes. Assumes buffer1.keyframes
        and buffer2.keyframes are disjoint.

        Args:
            buffer1 (KeyframeBuffer): The first KeyframeBuffer instance.
            buffer2 (KeyframeBuffer): The second KeyframeBuffer instance.

        Returns:
            Tuple[KeyframeBuffer, List[int], List[int]]: A tuple containing the merged KeyframeBuffer instance,
            indices of buffer1 keyframes in the merged buffer, and indices of buffer2 keyframes in the merged buffer.

        """
        if buffer1.keyframes is None and buffer2.keyframes is None:
            return KeyframeBuffer(), [], []
        elif buffer1.keyframes is None or (buffer1.keyframes is not None and len(buffer1.keyframes) == 0):
            return copy.deepcopy(buffer2), [], [k for k in
                                                range(len(buffer2.keyframes))] if buffer2.keyframes is not None else []
        elif buffer2.keyframes is None or (buffer2.keyframes is not None and len(buffer2.keyframes) == 0):
            return copy.deepcopy(buffer1), [k for k in
                                            range(len(buffer1.keyframes))] if buffer1.keyframes is not None else [], []

        all_keyframes = sorted(set(buffer1.keyframes + buffer2.keyframes))

        merged_buffer = KeyframeBuffer()
        merged_buffer.keyframes = all_keyframes

        indices_buffer1 = []
        indices_buffer2 = []

        for attr_name, attr_type in merged_buffer.__annotations__.items():
            merged_attr = None
            if attr_type is list:
                merged_attr = [getattr(buffer1, attr_name)[buffer1.keyframes.index(k)] if k in buffer1.keyframes else
                               getattr(buffer2, attr_name)[buffer2.keyframes.index(k)]
                               for k in all_keyframes]
            elif attr_type is torch.Tensor:
                attr1 = getattr(buffer1, attr_name)
                attr2 = getattr(buffer2, attr_name)
                merged_attr = torch.cat(
                    [attr1[:, buffer1.keyframes.index(k)].unsqueeze(1) if k in buffer1.keyframes else
                     attr2[:, buffer2.keyframes.index(k)].unsqueeze(1)
                     for k in all_keyframes], dim=1)
            setattr(merged_buffer, attr_name, merged_attr)

        # Track indices of keyframes in the merged buffer
        indices_buffer1.extend([buffer1.keyframes.index(k) for k in all_keyframes if k in buffer1.keyframes])
        indices_buffer2.extend([buffer2.keyframes.index(k) for k in all_keyframes if k in buffer2.keyframes])

        return merged_buffer, indices_buffer1, indices_buffer2

    def trim_keyframes(self, max_keyframes):
        if len(self.keyframes) > max_keyframes:
            # Keep only those last ones
            keep_keyframes = np.zeros(len(self.keyframes), dtype=bool)
            keep_keyframes[-max_keyframes:] = True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()

    def keep_selected_keyframes(self, keep_keyframes):
        not_keep_keyframes = ~ keep_keyframes

        # Get the deleted keyframes
        deleted_buffer = KeyframeBuffer()
        for attr_name, attr_type in deleted_buffer.__annotations__.items():
            if attr_type is list:
                modified_attr = (np.array(getattr(self, attr_name))[not_keep_keyframes]).tolist()
                setattr(deleted_buffer, attr_name, modified_attr)
            elif attr_type is torch.Tensor:
                modified_attr = getattr(self, attr_name)[:, not_keep_keyframes]
                setattr(deleted_buffer, attr_name, modified_attr)

        self.keyframes = (np.array(self.keyframes)[keep_keyframes]).tolist()
        self.flow_keyframes = (np.array(self.flow_keyframes)[keep_keyframes]).tolist()
        self.images = self.images[:, keep_keyframes]
        self.prev_images = self.prev_images[:, keep_keyframes]
        self.images_feat = self.images_feat[:, keep_keyframes]
        self.segments = self.segments[:, keep_keyframes]
        self.observed_flows = self.observed_flows[:, keep_keyframes]
        self.flow_segment_masks = self.flow_segment_masks[:, keep_keyframes]

        return deleted_buffer

    def stochastic_update(self, max_keyframes):
        N = len(self.keyframes)
        if len(self.keyframes) > max_keyframes:
            keep_keyframes = np.full(N, False)  # Create an array of length N initialized with False
            indices = np.random.choice(N, max_keyframes, replace=False)  # Randomly select keep_keyframes indices
            keep_keyframes[indices] = True  # Set the selected indices to True

            return self.keep_selected_keyframes(keep_keyframes)
        else:
            return KeyframeBuffer()  # No items removed, return an empty buffer


class Tracking6D:
    FrameResult = namedtuple('FrameResult', ['theoretical_flow', 'encoder_result', 'renders', 'frame_losses',
                                             'per_pixel_flow_error'])

    def __init__(self, config, device, write_folder, file0, bbox0, init_mask=None):
        # Encoders and related components
        self.encoder = None
        self.gt_encoder = None
        self.rgb_encoder = None
        self.last_encoder_result = None
        self.last_encoder_result_rgb = None

        # Rendering and mesh related
        self.rendering = None
        self.faces = None
        self.gt_mesh_prototype = None
        self.gt_texture = None

        # Features
        self.feat = None
        self.feat_rgb = None

        # Loss functions and optimizers
        self.all_parameters = None
        self.translational_params = None
        self.rotational_params = None
        self.positional_params = None
        self.non_positional_params = None
        self.loss_function = None
        self.rgb_loss_function = None
        self.optimizer_translational_parameters = None
        self.optimizer_rotational_parameters = None
        self.optimizer_positional_parameters = None
        self.optimizer_non_positional_parameters = None
        self.rgb_optimizer = None

        # Network related
        self.net = None
        self.model_flow = None

        # Ground truth related
        self.gt_rotations = None
        self.gt_translations = None

        # Keyframes
        self.active_keyframes = None
        self.all_keyframes = None
        self.recently_flushed_keyframes = None

        # Other utilities and flags
        self.write_results = None
        self.logged_sgd_translations = []
        self.logged_sgd_quaternions = []

        self.shape = tuple()
        self.write_folder = Path(write_folder)
        self.config = TrackerConfig(**config)
        self.config.fmo_steps = 1
        self.config_copy = copy.deepcopy(self.config)
        self.device = device

        iface_features, ivertices = self.initialize_mesh()
        self.initialize_flow_model()
        self.initialize_feature_extractor()

        if self.config.gt_tracking_log is not None:
            _, gt_rotations, gt_translations = load_gt_annotations_file(self.config.gt_tracking_log)
            self.gt_rotations = gt_rotations.to(self.device)
            self.gt_translations = gt_translations.to(self.device)

        torch.backends.cudnn.benchmark = True
        if type(bbox0) is dict:
            self.tracker = PrecomputedTracker(self.config.image_downsample, self.config.max_width, bbox0)
        else:
            if self.config.tracker_type == 'csrt':
                self.tracker = CSRTrack(self.config.image_downsample, self.config.max_width)
            elif self.config.tracker_type == 'ostrack':
                self.tracker = OSTracker(self.config.image_downsample, self.config.max_width)
            else:  # d3s
                self.tracker = MyTracker(self.config.image_downsample, self.config.max_width)

        images, images_feat, observed_flows, segments = self.get_initial_images(file0, bbox0, init_mask)

        num_channels = images_feat.shape[2]
        self.initialize_renderer_and_encoder(iface_features, ivertices, num_channels)

        if self.config.use_gt and self.gt_translations is not None and self.gt_rotations is not None:
            self.tracker = SyntheticDataGeneratingTracker(self.config.image_downsample, self.config.max_width, self,
                                                          self.gt_rotations, self.gt_translations)
            # Re-render the images using the synthetic tracker
            images, images_feat, observed_flows, segments = self.get_initial_images(file0, bbox0, init_mask)

        self.initialize_optimizer_and_loss(ivertices)

        if self.config.features == 'deep':
            self.initialize_rgb_encoder(self.faces, iface_features, ivertices, self.shape)

        self.best_model = {"value": 100,
                           "face_features": self.encoder.face_features.detach().clone(),
                           "faces": self.faces,
                           "encoder": copy.deepcopy(self.encoder.state_dict())}
        self.initialize_keyframes(flow_segment_masks=segments[:, :, -1:], images=images, images_feat=images_feat,
                                  observed_flows=observed_flows, prev_images=images.clone(), segments=segments)

        if self.config.verbose:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))

    def initialize_renderer_and_encoder(self, iface_features, ivertices, num_channels):
        self.rendering = RenderingKaolin(self.config, self.faces, self.shape[-1], self.shape[-2]).to(self.device)
        self.encoder = Encoder(self.config, ivertices, self.faces, iface_features, self.shape[-1], self.shape[-2],
                               num_channels).to(self.device)
        # Encoder for inferring the GT flow, and so on
        self.gt_encoder = Encoder(self.config, ivertices, self.faces, iface_features,
                                  self.shape[-1], self.shape[-2], 3).to(self.device)
        for name, param in self.gt_encoder.named_parameters():
            if isinstance(param, torch.Tensor):
                param.detach_()
        if self.gt_rotations is not None:
            rotation_quaternion = angle_axis_to_quaternion(self.gt_rotations, order=QuaternionCoeffOrder.WXYZ)
            self.gt_encoder.quaternion_w[...] = rotation_quaternion[..., 0, None]
            self.gt_encoder.quaternion_x[...] = rotation_quaternion[..., 1, None]
            self.gt_encoder.quaternion_y[...] = rotation_quaternion[..., 2, None]
            self.gt_encoder.quaternion_z[...] = rotation_quaternion[..., 3, None]
            self.gt_encoder.axis_angle_x[...] = self.gt_rotations[..., 0, None]
            self.gt_encoder.axis_angle_y[...] = self.gt_rotations[..., 1, None]
            self.gt_encoder.axis_angle_z[...] = self.gt_rotations[..., 2, None]
        if self.gt_translations is not None:
            self.gt_encoder.translation[...] = self.gt_translations
        self.encoder.train()

    def get_initial_images(self, file0, bbox0, init_mask):
        if type(self.tracker) is SyntheticDataGeneratingTracker:
            file0 = 0
        images, segments, self.config.image_downsample = self.tracker.init_bbox(file0, bbox0, init_mask)
        images, segments = images[None].to(self.device), segments[None].to(self.device)
        images_feat = self.feat(images).detach()
        self.shape = segments.shape
        observed_flows = segments * 0
        return images, images_feat, observed_flows, segments

    def initialize_optimizer_and_loss(self, ivertices):
        all_parameters = set(list(self.encoder.parameters()))
        translational_params = {self.encoder.translation}
        rotational_params = {self.encoder.quaternion_w, self.encoder.quaternion_x,
                             self.encoder.quaternion_y, self.encoder.quaternion_z}
        positional_params = translational_params | rotational_params
        rotational_params = [
            {'params': [self.encoder.quaternion_x, self.encoder.quaternion_y, self.encoder.quaternion_z],
             'lr': self.config.learning_rate,
             'name': 'axes_quat'},
            {'params': [self.encoder.axis_angle_x, self.encoder.axis_angle_y, self.encoder.axis_angle_z],
             'lr': self.config.learning_rate * 1e-0,
             'name': 'axis_angle'},
            {'params': [self.encoder.quaternion_w],
             'lr': self.config.learning_rate * 1e-0,
             'name': 'half_cosine'}
        ]
        non_positional_params = all_parameters - positional_params
        positional_params = [
            {'params': [self.encoder.quaternion_x, self.encoder.quaternion_y, self.encoder.quaternion_z],
             'lr': self.config.learning_rate,
             'name': 'axes_quat'},
            {'params': [self.encoder.axis_angle_x, self.encoder.axis_angle_y, self.encoder.axis_angle_z],
             'lr': self.config.learning_rate * 1e-0,
             'name': 'axis_angle'},
            {'params': [self.encoder.quaternion_w],
             'lr': self.config.learning_rate * 1e-0,
             'name': 'half_cosine'},
            {'params': list(translational_params),
             'lr': self.config.learning_rate * self.config.translation_learning_rate_coef,
             'name': 'trans'},
        ]
        self.optimizer_non_positional_parameters = torch.optim.Adam(non_positional_params, lr=self.config.learning_rate)
        self.optimizer_positional_parameters = torch.optim.SGD(positional_params, lr=self.config.learning_rate)
        self.optimizer_translational_parameters = torch.optim.SGD(translational_params, lr=self.config.learning_rate)
        self.optimizer_rotational_parameters = torch.optim.SGD(rotational_params, lr=self.config.learning_rate)
        self.loss_function = FMOLoss(self.config, ivertices, self.faces).to(self.device)

    def initialize_feature_extractor(self):
        if self.config.features == 'deep':
            self.net = S2DNet(device=self.device, checkpoint_path=g_ext_folder).to(self.device)
            self.feat = lambda x: self.net(x[0])[0][None][:, :, :64]
            self.feat_rgb = lambda x: x
        else:
            self.feat = lambda x: x

    def initialize_mesh(self):
        if self.config.gt_texture is not None and self.config.use_gt:
            texture_np = torch.from_numpy(imageio.imread(Path(self.config.gt_texture)))
            self.gt_texture = texture_np.permute(2, 0, 1)[None].to(self.device) / 255.0

        if self.config.gt_mesh_prototype is not None:
            self.gt_mesh_prototype = kaolin.io.obj.import_mesh(str(self.config.gt_mesh_prototype), with_materials=True)

        prot = self.config.shapes[0]
        if self.config.use_gt:
            ivertices = normalize_vertices(self.gt_mesh_prototype.vertices).numpy()
            self.faces = self.gt_mesh_prototype.faces
            iface_features = self.gt_mesh_prototype.uvs[self.gt_mesh_prototype.face_uvs_idx].numpy()
        elif self.config.init_shape:
            mesh = load_obj(self.config.init_shape)
            ivertices = mesh.vertices.numpy()
            ivertices = ivertices - ivertices.mean(0)
            ivertices = ivertices / ivertices.max()
            self.faces = mesh.faces.numpy().copy()
            iface_features = generate_face_features(ivertices, self.faces)
        else:
            mesh = load_obj(os.path.join('./prototypes', prot + '.obj'))
            ivertices = mesh.vertices.numpy()
            self.faces = mesh.faces.numpy().copy()
            iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()
        return iface_features, ivertices

    def initialize_rgb_encoder(self, faces, iface_features, ivertices, shape):
        config = copy.deepcopy(self.config)
        config.features = 'rgb'
        self.rgb_encoder = Encoder(config, ivertices, faces, iface_features, shape[-1], shape[-2], 3).to(
            self.device)
        rgb_parameters = list(self.rgb_encoder.parameters())[-1:]
        self.rgb_optimizer = torch.optim.Adam(rgb_parameters, lr=self.config.learning_rate)
        self.rgb_encoder.train()
        config.loss_laplacian_weight = 0
        config.loss_tv_weight = 1.0
        config.loss_iou_weight = 0
        config.loss_dist_weight = 0
        config.loss_q_weight = 0
        config.loss_t_weight = 0
        config.loss_flow_weight = 0
        config.loss_texture_change_weight = 0
        self.rgb_loss_function = FMOLoss(config, ivertices, faces).to(self.device)

    def initialize_keyframes(self, flow_segment_masks, images, images_feat, observed_flows, prev_images, segments):
        keyframes = [0]
        flow_keyframes = [0]
        self.active_keyframes = KeyframeBuffer(keyframes=keyframes,
                                               flow_keyframes=flow_keyframes,
                                               images=images,
                                               prev_images=prev_images,
                                               images_feat=images_feat,
                                               segments=segments,
                                               observed_flows=observed_flows,
                                               flow_segment_masks=flow_segment_masks)
        self.recently_flushed_keyframes = KeyframeBuffer()
        self.all_keyframes = self.active_keyframes

    def initialize_flow_model(self):
        if self.config.flow_model == 'RAFT':
            self.model_flow = get_flow_model_raft()
        elif self.config.flow_model == 'GMA':
            self.model_flow = get_flow_model_gma()
        elif self.config.flow_model == 'MFT':
            self.model_flow = get_flow_model_mft()

    def run_tracking(self, files, bboxes, gt_flows=None):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1
        if type(bboxes) is dict:
            sorted_bb_keys = sorted(list(bboxes.keys()))
            bboxes = {i: bboxes[sorted_bb_keys[i]] for i, key in zip(range(len(bboxes)), sorted_bb_keys)}

        our_losses = -np.ones((files.shape[0] - 1, 1))
        self.write_results = WriteResults(self.write_folder, self.active_keyframes.images, files.shape[0])

        prev_image = self.active_keyframes.images[:, -1]
        prev_segment = self.active_keyframes.segments[:, -1]
        self.last_encoder_result_rgb = self.rgb_encoder(self.all_keyframes.keyframes)
        self.last_encoder_result = self.encoder(self.all_keyframes.keyframes)

        resize_transform = transforms.Resize(self.all_keyframes.images.shape[-2:],
                                             interpolation=transforms.InterpolationMode.BICUBIC)

        b0 = None
        for stepi in range(1, self.config.input_frames):

            image_raw, segment = self.tracker.next(stepi if self.config.use_gt else files[stepi])

            image, segment = image_raw[None].to(self.device), segment[None].to(self.device)
            if b0 is not None:
                segment_clean = segment * 0
                segment_clean[:, :, :, b0[0]:b0[1], b0[2]:b0[3]] = segment[:, :, :, b0[0]:b0[1], b0[2]:b0[3]]
                segment_clean[:, :, 0] = segment[:, :, 0]
                segment = segment_clean

            image_feat = self.feat(image).detach()

            self.active_keyframes.images = torch.cat((self.active_keyframes.images, image), 1)
            self.active_keyframes.images_feat = torch.cat((self.active_keyframes.images_feat, image_feat), 1)
            self.active_keyframes.segments = torch.cat((self.active_keyframes.segments, segment), 1)
            self.active_keyframes.keyframes += [stepi]
            self.active_keyframes.flow_keyframes += [stepi - 1]
            self.active_keyframes.prev_images = torch.cat((self.active_keyframes.prev_images, prev_image[None]), dim=1)

            image_prev_x255 = (self.active_keyframes.prev_images[:, -1, :, :, :]).float() * 255
            image_new_x255 = (self.active_keyframes.images[:, -1, :, :, :]).float() * 255
            if self.active_keyframes.images.shape[1] > 2:
                image_preprev_x255 = (self.active_keyframes.images[:, -1, :, :, :]).float() * 255
            else:
                image_preprev_x255 = None

            start = time.time()
            if self.config.render_just_bounding_box:
                b0 = get_bbox(self.all_keyframes.segments)
            else:
                b0 = [0, self.shape[-1], 0, self.shape[-2]]
            self.rendering = RenderingKaolin(self.config, self.faces, b0[3] - b0[2], b0[1] - b0[0]).to(self.device)

            with torch.no_grad():
                if self.config.use_gt:
                    observed_flow = self.tracker.next_flow(stepi)
                elif gt_flows is None:
                    if self.config.flow_model != 'MFT':
                        _, observed_flow = get_flow_from_images(image_prev_x255, image_new_x255, self.model_flow)
                    else:
                        observed_flow, occlusion, uncertainty = get_flow_from_images_mft(image_prev_x255,
                                                                                         image_new_x255,
                                                                                         self.model_flow)
                        # TODO finish this part of the occlusion computation
                        if image_preprev_x255 is not None:
                            _, occlusion_preprev, _ = get_flow_from_images_mft(image_preprev_x255, image_new_x255,
                                                                               self.model_flow)
                        else:
                            occlusion_fraction = 0.0

                    observed_flow[:, 0, ...] = observed_flow[:, 0, ...] / observed_flow.shape[-2]
                    observed_flow[:, 1, ...] = observed_flow[:, 1, ...] / observed_flow.shape[-1]
                else:  # We have ground truth flow annotations
                    # The annotations are assumed to be in the [0, 1] coordinate range
                    observed_flow = torch.load(gt_flows[stepi])[0].to(self.device)  # torch.Size([1, H, W, 2])
                    observed_flow = observed_flow.permute(0, 3, 1, 2)
                    observed_flow = resize_transform(observed_flow)

                self.active_keyframes.observed_flows = torch.cat((self.active_keyframes.observed_flows,
                                                                  observed_flow[None]), dim=1)
                self.active_keyframes.flow_segment_masks = torch.cat((self.active_keyframes.flow_segment_masks,
                                                                      segment[0][None, :, -1:]), dim=1)

            # We have added some keyframes. If it is more than the limit, delete them
            if not self.config.all_frames_keyframes:
                deleted_keyframes = self.active_keyframes.trim_keyframes(self.config.max_keyframes)
                self.recently_flushed_keyframes, _, _ = KeyframeBuffer.merge(self.recently_flushed_keyframes,
                                                                             deleted_keyframes)

            if self.config.stochastically_add_keyframes:
                self.all_keyframes, active_buffer_indices, _ = KeyframeBuffer.merge(self.active_keyframes,
                                                                                    self.recently_flushed_keyframes)
            else:
                self.all_keyframes = self.active_keyframes
                active_buffer_indices = list(range(len(self.active_keyframes.keyframes)))

            self.last_encoder_result = EncoderResult(*[tensor.clone()
                                                       if tensor is not None else None for tensor in
                                                       self.encoder(self.all_keyframes.keyframes)])
            self.last_encoder_result_rgb = EncoderResult(*[tensor.clone()
                                                           if tensor is not None else None for tensor in
                                                           self.rgb_encoder(self.all_keyframes.keyframes)])

            frame_result = self.apply(self.all_keyframes.images_feat[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                      self.all_keyframes.segments[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                      self.all_keyframes.observed_flows[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                      self.all_keyframes.flow_segment_masks[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                      self.all_keyframes.keyframes, self.all_keyframes.flow_keyframes, step_i=stepi)

            encoder_result = frame_result.encoder_result

            silh_losses = np.array(self.best_model["losses"]["silh"])

            our_losses[stepi - 1] = silh_losses[-1]
            print('Elapsed time in seconds: ', time.time() - start, "Frame ", stepi, "out of",
                  self.config.input_frames)

            tex = None
            if self.config.features == 'deep':
                self.rgb_apply(self.all_keyframes.images[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                               self.all_keyframes.segments[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                               self.all_keyframes.observed_flows[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                               self.all_keyframes.flow_segment_masks[:, :, :, b0[0]:b0[1], b0[2]:b0[3]])
                tex = torch.nn.Sigmoid()(self.rgb_encoder.texture_map)

            if self.config.write_results:
                with torch.no_grad():
                    visualize_theoretical_flow(self, frame_result.theoretical_flow.clone().detach(), b0,
                                               self.all_keyframes.observed_flows[
                                               :, -1, :, b0[0]:b0[1], b0[2]:b0[3]].clone().detach(),
                                               self.all_keyframes.keyframes, stepi)

                    self.write_results.write_results(self, b0=b0,
                                                     bboxes=bboxes, our_losses=our_losses,
                                                     silh_losses=silh_losses, stepi=stepi,
                                                     encoder_result=encoder_result,
                                                     ground_truth_segments=self.all_keyframes.segments,
                                                     images=self.all_keyframes.images,
                                                     images_feat=self.all_keyframes.images_feat,
                                                     tex=tex,
                                                     frame_losses=frame_result.frame_losses)

                    gt_mesh_vertices = self.gt_mesh_prototype.vertices[None].to(self.device) \
                        if self.gt_mesh_prototype is not None else None
                    # self.write_results.evaluate_metrics(stepi=stepi, tracking6d=self,
                    #                                     keyframes=self.active_keyframes.keyframes,
                    #                                     predicted_vertices=encoder_result.vertices,
                    #                                     predicted_quaternion=encoder_result.quaternions,
                    #                                     predicted_translation=encoder_result.translations,
                    #                                     predicted_mask=frame_result.renders[:, :, 0, -1, ...],
                    #                                     gt_vertices=gt_mesh_vertices,
                    #                                     gt_rotation=self.gt_rotations,
                    #                                     gt_translation=self.gt_translations,
                    #                                     gt_object_mask=self.active_keyframes.segments[:, :, 1, ...])

                    # Visualize flow we get from the video
                    visualize_flow(observed_flow.detach().clone(), image, image_new_x255, image_prev_x255, segment,
                                   stepi, self.write_folder, frame_result.per_pixel_flow_error)

            self.encoder.clear_logs()

            keep_keyframes = (silh_losses < 0.8)  # remove really bad ones (IoU < 0.2)
            keep_keyframes = keep_keyframes[active_buffer_indices]
            min_index = np.argmin(silh_losses[active_buffer_indices])
            keep_keyframes[min_index] = True  # keep the best (in case all are bad)

            # normTdist = compute_trandist(renders)

            angles = consecutive_quaternions_angular_difference(encoder_result.quaternions)
            # angles = consecutive_quaternions_angular_difference2(encoder_result.quaternions)
            print("Angles:", angles)

            rot_degree_th = 45
            small_rotation = angles.shape[0] > 1 and abs(angles[-1]) < rot_degree_th and abs(angles[-2]) < rot_degree_th
            if small_rotation:  # and small_translation):
                keep_keyframes[-1] = True
                keep_keyframes[-2] = False or self.config.all_frames_keyframes  # Default False
                keep_keyframes[-3] = True

            if not self.config.all_frames_keyframes:
                deleted_keyframes = self.active_keyframes.keep_selected_keyframes(keep_keyframes)
                self.recently_flushed_keyframes, _, _ = KeyframeBuffer.merge(self.recently_flushed_keyframes,
                                                                             deleted_keyframes)
                self.recently_flushed_keyframes.stochastic_update(4)

            prev_image = image[0]
            prev_segment = segment[0]

        return self.best_model

    def get_rendered_image_features(self, lights, quaternion, texture_maps, translation, vertices):
        feat_renders_crop = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture_maps,
                                           lights, True)
        feat_renders_crop = feat_renders_crop[:, :, :, :-1]
        feat_renders_crop = torch.cat((feat_renders_crop[:, :, :, :3], feat_renders_crop[:, :, :, -1:]), 3)
        return feat_renders_crop

    def get_rendered_image(self, bounding_box, lights, quaternion, texture, translation, vertices):
        """

        :param bounding_box: [y0, y1, x0, x1] location of the bounding box
        :param lights:
        :param quaternion: Rotation of the mesh
        :param texture: Mesh texture
        :param translation: Mesh translation
        :param vertices: Mesh vertices
        :return:
            renders_cropped: [4, 272, 224] RGB-A rendering of the image
            renders: [4, 300, 225] RGB-A rendering of the image, where the rendering is put inside the bounding box
        """
        renders_crop = self.rendering(translation, quaternion, vertices, self.encoder.face_features, texture, lights)
        renders_crop = torch.cat((renders_crop[:, :, :, :3], renders_crop[:, :, :, -1:]), 3)
        renders = self.write_image_into_bbox(bounding_box, renders_crop)
        return renders, renders_crop

    def write_image_into_bbox(self, bounding_box, renders_crop):
        """

        :param bounding_box: List specifying the bounding box starts, resp. end
        :param renders_crop: Image of shape ..., C, H, W
        :return:
        """
        renders = torch.zeros(renders_crop.shape[:-2] + self.all_keyframes.images_feat.shape[-2:]).to(self.device)
        renders[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]] = renders_crop
        return renders

    def apply(self, observed_images, observed_segmentations, observed_flows, observed_flows_segmentations,
              keyframes, flow_frames, step_i=0):

        self.config.loss_fl_not_obs_rend_weight = self.config.loss_flow_weight
        self.config.loss_fl_obs_and_rend_weight = self.config.loss_flow_weight

        self.logged_sgd_quaternions = []
        self.logged_sgd_translations = []

        # Updates offset of the next rotation
        self.encoder.compute_next_offset(step_i)
        self.encoder.clear_logs()

        self.write_results.set_tensorboard_log_for_frame(step_i)

        frame_losses = []
        if self.config.write_results:
            save_image(observed_images[0, :, :3], os.path.join(self.write_folder, 'im.png'),
                       nrow=self.config.max_keyframes + 1)
            save_image(torch.cat((observed_images[0, :, :3], observed_segmentations[0, :, [1]]), 1),
                       os.path.join(self.write_folder, 'segments.png'), nrow=self.config.max_keyframes + 1)
            if self.config.weight_by_gradient:
                save_image(torch.cat((observed_segmentations[0, :, [0, 0, 0]], 0 * observed_images[0, :, :1] + 1), 1),
                           os.path.join(self.write_folder, 'weights.png'))

        # Restore the learning rate on its prior values
        self.reset_learning_rate()

        if self.config.use_lr_scheduler:
            self.config.loss_rgb_weight = 0
            if step_i <= 2:
                self.config.loss_flow_weight = 0
            else:
                self.config.loss_flow_weight = self.config_copy.loss_flow_weight

        scheduler_positional_params = lr_scheduler.ReduceLROnPlateau(self.optimizer_positional_parameters,
                                                                     mode='min', factor=0.8,
                                                                     patience=self.config.lr_scheduler_patience,
                                                                     verbose=False)

        def lambda_schedule(epoch):
            return 1 / (1 + np.exp(-0.25 * (epoch - self.config.optimize_non_positional_params_after)))

        scheduler_non_positional_params = lr_scheduler.LambdaLR(self.optimizer_non_positional_parameters,
                                                                lambda_schedule)

        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0

        encoder_result = None
        theoretical_flow = None
        renders = None
        per_pixel_error = None

        # rotation_quaternion = angle_axis_to_quaternion(self.gt_rotations, order=QuaternionCoeffOrder.WXYZ)
        # self.encoder.quaternion_offsets = rotation_quaternion.clone()
        # self.encoder.translation_offsets = self.gt_translations.clone()

        best_loss = math.inf
        no_improvements = 0
        epoch = 0
        loss_improvement_threshold = 1e-4

        encoder_result = self.run_lebesgue_marquardt_method(flow_frames, keyframes, observed_images,
                                                            observed_segmentations, observed_flows,
                                                            observed_flows_segmentations)

        self.run_lbfgs_optimization(epoch, observed_images, observed_segmentations, observed_flows,
                                    observed_flows_segmentations, keyframes, flow_frames, frame_losses)

        # First optimize the positional parameters first while preventing steps that increase the loss
        print("Optimizing positional parameters using linear learning rate scheduling")
        if self.config.coordinate_descent:
            optim_result = self.coordinate_descent_with_linear_lr_schedule(best_loss, epoch, flow_frames, frame_losses,
                                                                           keyframes, loss_improvement_threshold,
                                                                           observed_flows,
                                                                           observed_flows_segmentations,
                                                                           observed_images, observed_segmentations)
        else:
            optim_result = self.gradient_descent_with_linear_lr_schedule(best_loss, encoder_result, epoch, flow_frames,
                                                                         frame_losses, keyframes,
                                                                         loss_improvement_threshold, no_improvements,
                                                                         observed_flows, observed_flows_segmentations,
                                                                         observed_images, observed_segmentations,
                                                                         per_pixel_error, renders, theoretical_flow)

        best_loss, encoder_result, epoch, per_pixel_error, renders, theoretical_flow = optim_result

        self.encoder.load_state_dict(self.best_model["encoder"])

        # Now optimize all the parameters jointly using normal gradient descent
        print("Optimizing all parameters")

        # self.reset_learning_rate()

        for epoch in range(epoch, self.config.iterations):

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

            model_loss = self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)
            if abs(model_loss - self.best_model["value"]) > 1e-3:
                iters_without_change = 0
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                if self.config.write_intermediate:
                    write_renders(torch.cat((renders[:, :, :, :3], renders[:, :, :, -1:]), 3), self.write_folder,
                                  self.config.max_keyframes + 1)
            else:
                iters_without_change += 1

            if self.config.loss_rgb_weight == 0 and self.config_copy.loss_rgb_weight:
                if epoch > 100 or model_loss < 0.1:
                    self.config.loss_rgb_weight = self.config_copy.loss_rgb_weight
                    self.best_model["value"] = 100
            else:
                if epoch > self.config.allow_break_sgd_after and \
                        abs(self.best_model["value"] - model_loss) <= 1e-3 and \
                        iters_without_change > self.config.break_sgd_after_iters_with_no_change:
                    break
            if epoch < self.config.iterations - 1:
                joint_loss = joint_loss.mean()
                self.optimizer_non_positional_parameters.zero_grad()
                self.optimizer_positional_parameters.zero_grad()

                joint_loss.backward()

                # self.optimizer_non_positional_parameters.step()
                # self.optimizer_positional_parameters.step()

                if self.config.use_lr_scheduler:
                    scheduler_positional_params.step(joint_loss)
                    scheduler_non_positional_params.step()

        self.encoder.load_state_dict(self.best_model["encoder"])

        if self.config.visualize_loss_landscape and (step_i in {0, 1, 2, 3} or step_i % 18 == 0):
            self.write_results.visualize_loss_landscape(self, observed_images, observed_segmentations,
                                                        observed_flows, observed_flows_segmentations, step_i)

        # Inferring the most up-to date state after the optimization is finished
        infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                        observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

        frame_result = self.FrameResult(theoretical_flow=theoretical_flow,
                                        encoder_result=encoder_result,
                                        renders=renders,
                                        frame_losses=frame_losses,
                                        per_pixel_flow_error=per_pixel_error)

        return frame_result

    def run_levenberg_marquardt_method(self, flow_frames, keyframes, observed_images, observed_segmentations,
                                       observed_flows, observed_flows_segmentations, frame_losses):

        def loss_function_wrapper(translations_quaternions_, encoder_result_, encoder_result_flow_frames_):
            # print("--Lf wrap start", int(torch.cuda.memory_allocated() / 1024))
            translations_ = translations_quaternions_[..., :3]
            quaternions_ = translations_quaternions_[..., 3:]
            # quaternions_ = encoder_result_.quaternions
            encoder_result_ = encoder_result_._replace(translations=translations_, quaternions=quaternions_)

            # print("---Render start", int(torch.cuda.memory_allocated() / 1024))
            renders_ = self.rendering(translations_, quaternions_, encoder_result_.vertices,
                                      self.encoder.face_features, encoder_result_.texture_maps,
                                      None)
            # print("---Render size ", int(renders_.nelement() * renders_.element_size() / 1024))

            # print("---Render end  ", int(torch.cuda.memory_allocated() / 1024))
            # print("---The fl start", int(torch.cuda.memory_allocated() / 1024))
            flow_result_ = self.rendering.compute_theoretical_flow(encoder_result_, encoder_result_flow_frames_)
            # print("---The fl start", int(torch.cuda.memory_allocated() / 1024))
            theoretical_flow_, rendered_flow_segmentation_ = flow_result_
            rendered_flow_segmentation_ = rendered_flow_segmentation_[None]

            # Renormalization compensating for the fact that we render into bounding box that is smaller than the
            # actual image
            theoretical_flow_ = self.normalize_rendered_flows(theoretical_flow_)

            rendered_silhouettes_ = renders_[0, :, :, -1:]

            loss_result = self.loss_function.forward(rendered_images=renders_, observed_images=observed_images,
                                                     rendered_silhouettes=rendered_silhouettes_,
                                                     observed_silhouettes=observed_segmentations,
                                                     rendered_flow=theoretical_flow_,
                                                     observed_flow=observed_flows,
                                                     observed_flow_segmentation=observed_flows_segmentations,
                                                     rendered_flow_segmentation=rendered_flow_segmentation_,
                                                     keyframes_encoder_result=encoder_result_,
                                                     last_keyframes_encoder_result=self.last_encoder_result,
                                                     return_end_point_errors=True)

            del renders_
            del theoretical_flow_
            del rendered_silhouettes_

            return loss_result.to(torch.float)

        self.best_model["value"] = 100

        encoder_result, encoder_result_flow_frames = self.frames_and_flow_frames_inference(keyframes, flow_frames,
                                                                                           encoder_type='deep_features')
        kf_translations = encoder_result.translations[0]
        kf_quaternions = encoder_result.quaternions
        translations_quaternions = torch.cat([kf_translations, kf_quaternions], dim=-1)
        additional_args = (encoder_result, encoder_result_flow_frames)
        fun = lambda p: loss_function_wrapper(p, *additional_args)

        # coeffs_list = lsq_gna_custom(p=translations_quaternions, function=fun, args=())
        coeffs_list = lsq_lma_custom(p=translations_quaternions, function=fun, args=())

        for epoch in range(len(coeffs_list)):
            coeff_row = coeffs_list[epoch]
            row_translation = coeff_row[None, :, :, :3]
            row_quaternion = coeff_row[:, :, 3:]
            encoder_result = encoder_result._replace(translations=row_translation, quaternions=row_quaternion)
            renders = self.rendering(encoder_result.translations, encoder_result.quaternions, encoder_result.vertices,
                                     self.encoder.face_features, encoder_result.texture_maps, None)

            flow_result = self.rendering.compute_theoretical_flow(encoder_result, encoder_result_flow_frames)
            theoretical_flow, rendered_flow_segmentation = flow_result
            rendered_flow_segmentation = rendered_flow_segmentation[None]

            theoretical_flow_ = self.normalize_rendered_flows(theoretical_flow)
            rendered_silhouettes = renders[0, :, :, -1:]

            loss_result = self.loss_function.forward(rendered_images=renders, observed_images=observed_images,
                                                     rendered_silhouettes=rendered_silhouettes,
                                                     observed_silhouettes=observed_segmentations,
                                                     rendered_flow=theoretical_flow_,
                                                     observed_flow=observed_flows,
                                                     observed_flow_segmentation=observed_flows_segmentations,
                                                     rendered_flow_segmentation=rendered_flow_segmentation,
                                                     keyframes_encoder_result=encoder_result,
                                                     last_keyframes_encoder_result=self.last_encoder_result,
                                                     return_end_point_errors=False)

            losses_all, losses, joint_loss, per_pixel_error = loss_result
            joint_loss = joint_loss.mean()
            if joint_loss < self.best_model["value"]:
                self.best_model["losses"] = losses_all
                self.best_model["value"] = joint_loss

                self.encoder.translation_offsets[0, 0, keyframes, :] = encoder_result.translations[0, 0, :, :].detach()
                self.encoder.quaternion_offsets[0, keyframes, :] = encoder_result.quaternions[0, :].detach()

                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())

            self.log_inference_results(joint_loss, epoch, frame_losses, joint_loss, losses, encoder_result)

        return encoder_result

    def run_lbfgs_optimization(self, epoch, observed_images, observed_segmentations, observed_flows,
                               observed_flows_segmentations, keyframes, flow_frames, frame_losses):
        """
            Runs the LBFGS optimization on the positional parameters of the model.

            This method is an experimental approach to optimizing the positional parameters using the LBFGS algorithm.
            The optimization is terminated either after 5 iterations or when the best loss becomes less than or
            equal to 1e-4.

            Parameters:
            - epoch (int): The starting epoch for the optimization.
            - observed_images (torch.Tensor): Observed images for the model inference.
            - observed_segmentations (torch.Tensor): Segmentations corresponding to the observed images.
            - observed_flows (torch.Tensor): Observed optical flows.
            - observed_flows_segmentations (torch.Tensor): Segmentations corresponding to the observed optical flows.
            - keyframes (torch.Tensor): Keyframes used for model inference.
            - flow_frames (torch.Tensor): Optical flow frames used for model inference.
            - frame_losses (list): List to store individual frame losses for logging purposes.

            Returns:
            - float: Best loss achieved during the optimization.
            - int: Last epoch after optimization.

            Note:
            The method saves the state dictionary of the best encoder model achieved during the optimization.
        """
        self.optimizer_positional_parameters = torch.optim.LBFGS(self.positional_params,
                                                                 lr=self.config.learning_rate)
        best_loss = math.inf
        iters_without_change = 0

        def closure():
            nonlocal iters_without_change
            nonlocal best_loss

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result
            model_loss = self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)

            joint_loss = joint_loss.mean()

            if abs(best_loss - joint_loss) > 1e-3:
                iters_without_change = 0
                best_loss = joint_loss
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
            else:
                iters_without_change += 1

            self.optimizer_positional_parameters.zero_grad()
            joint_loss.backward()

            return joint_loss

        for i in range(5):
            self.optimizer_positional_parameters.step(closure)
            epoch += 1
            if best_loss <= 1e-4:
                break
        self.encoder.load_state_dict(self.best_model["encoder"])

        return best_loss, epoch

    def gradient_descent_with_linear_lr_schedule(self, best_loss, encoder_result, epoch, flow_frames, frame_losses,
                                                 keyframes, loss_improvement_threshold, no_improvements, observed_flows,
                                                 observed_flows_segmentations, observed_images, observed_segmentations,
                                                 per_pixel_error, renders, theoretical_flow):
        while no_improvements < self.config.break_sgd_after_iters_with_no_change:
            self.config.loss_fl_not_obs_rend_weight = self.config.loss_flow_weight
            self.config.loss_fl_obs_and_rend_weight = self.config.loss_flow_weight

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

            joint_loss = joint_loss.mean()
            self.optimizer_positional_parameters.zero_grad()
            joint_loss.backward()

            loss_improvement = best_loss - joint_loss
            if loss_improvement > loss_improvement_threshold:
                best_loss = joint_loss
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                for param_group in self.optimizer_positional_parameters.param_groups:
                    param_group['lr'] *= 2.0
            elif loss_improvement < 0:
                self.encoder.load_state_dict(self.best_model["encoder"])
                for param_group in self.optimizer_positional_parameters.param_groups:
                    param_group['lr'] /= 2.0
            elif 0 <= loss_improvement <= loss_improvement_threshold:
                self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)
                self.optimizer_positional_parameters.step()
                epoch += 1
                no_improvements += 1

        self.encoder.load_state_dict(self.best_model["encoder"])
        infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                        observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result
        return best_loss, encoder_result, epoch, per_pixel_error, renders, theoretical_flow

    def coordinate_descent_with_linear_lr_schedule(self, best_loss, epoch, flow_frames, frame_losses, keyframes,
                                                   loss_improvement_threshold, observed_flows,
                                                   observed_flows_segmentations, observed_images,
                                                   observed_segmentations):
        no_improvements = 0

        # observed_flows_clone = observed_flows.detach().clone()
        # observed_flows_segmentations_clone = observed_flows_segmentations.detach().clone()
        # observed_images_clone = observed_images.detach().clone()
        # observed_segmentations_clone = observed_segmentations.detach().clone()

        infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                        observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result
        joint_loss = joint_loss.mean()
        self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())

        self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)

        # loss_seq = []
        # loss_seq_all = []
        # parameters_newest = []
        # best_translation = None

        while no_improvements < self.config.break_sgd_after_iters_with_no_change:

            # TODO inferring the model three times per step is ubiquitous, it is sufficient to remember the values
            # TODO from the last iteration, and if we revert to the latest checkpoint, one can save the values from
            # TODO the checkpoint as well

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

            joint_loss = joint_loss.mean()

            self.optimizer_rotational_parameters.zero_grad()
            joint_loss.backward()
            self.optimizer_rotational_parameters.step()

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

            joint_loss = joint_loss.mean()

            self.optimizer_translational_parameters.zero_grad()
            joint_loss.backward()
            self.optimizer_translational_parameters.step()

            infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                            observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result
            joint_loss = joint_loss.mean()

            loss_improvement = best_loss - joint_loss

            # loss_seq_all.append((float(joint_loss), float(self.encoder.translation[0, 0, 1, 0]), float(loss_improvement)))

            if loss_improvement >= 0:
                # loss_seq.append((float(joint_loss), float(self.encoder.translation[0, 0, 1, 0]), float(loss_improvement), 'saved_encoder'))
                best_loss = joint_loss
                # best_translation = encoder_result.translations

                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                for param_group in self.optimizer_translational_parameters.param_groups:
                    param_group['lr'] *= 2.0
                for param_group in self.optimizer_rotational_parameters.param_groups:
                    param_group['lr'] *= 2.0
                for param_group in self.optimizer_positional_parameters.param_groups:
                    param_group['lr'] *= 2.0

                epoch += 1
                if loss_improvement <= loss_improvement_threshold:
                    no_improvements += 1
                else:
                    no_improvements = 0
                self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)

            elif loss_improvement < 0:
                self.encoder.load_state_dict(self.best_model["encoder"])
                for param_group in self.optimizer_translational_parameters.param_groups:
                    param_group['lr'] /= 2.0
                for param_group in self.optimizer_rotational_parameters.param_groups:
                    param_group['lr'] /= 2.0
                for param_group in self.optimizer_positional_parameters.param_groups:
                    param_group['lr'] /= 2.0
                no_improvements += 1

            parameters_newest = [param.detach().clone() for param in self.encoder.parameters()]

        # for it in loss_seq:
        #     print(it)
        #
        # print("-----------------")
        #
        # for it in loss_seq_all:
        #     print(it)

        # print("Encoder before reloading", float(self.encoder.translation[0, 0, 1, 0]))

        self.encoder.load_state_dict(self.best_model["encoder"])

        # print("Encoder  after reloading", float(self.encoder.translation[0, 0, 1, 0]))
        #
        # parameters_best = [param.detach().clone() for param in self.encoder.parameters()]
        #
        # print([torch.equal(parameters_newest[i], parameters_best[i]) for i in range(len(parameters_best))])
        #
        # values_list = [val for val in self.best_model['encoder'].values()]
        # print([torch.equal(values_list[i], parameters_newest[i]) for i in range(len(parameters_best))])

        infer_result = self.infer_model(observed_images, observed_segmentations, observed_flows,
                                        observed_flows_segmentations, keyframes, flow_frames, 'deep_features')
        encoder_result3, joint_loss3, losses3, losses_all3, per_pixel_error3, renders3, theoretical_flow3 = infer_result
        # joint_loss3 = joint_loss3.mean()

        # encoder_result2_tensors = [encres for encres in encoder_result if type(encres) is torch.Tensor]
        # encoder_result3_tensors = [encres for encres in encoder_result3 if type(encres) is torch.Tensor]
        # print("Enc res comparison", [torch.equal(encoder_result2_tensors[i], encoder_result3_tensors[i]) for i in range(len(encoder_result2_tensors))])
        #
        # print("-----------------")
        # print(float(joint_loss3), float(self.encoder.translation[0, 0, 1, 0]))
        # print(f'Best model: {best_translation}\n', f'reloaded best model: {encoder_result3.translations}\n',
        #       f'last model: {encoder_result.translations}')
        #
        # objective_functions_defaults = [
        #                                 torch.equal(observed_flows, observed_flows_clone),
        #                                 torch.equal(observed_images, observed_images_clone),
        #                                 torch.equal(observed_flows_segmentations, observed_flows_segmentations_clone),
        #                                 torch.equal(observed_segmentations, observed_segmentations_clone),
        #                                 ]
        # print("Objective function defaults", objective_functions_defaults)
        # breakpoint()

        return best_loss, encoder_result3, epoch, per_pixel_error3, renders3, theoretical_flow3

    def normalize_rendered_flows(self, rendered_flows):
        rendered_flows[..., 0] = rendered_flows[..., 0] * (self.rendering.width / self.shape[-1])
        rendered_flows[..., 1] = rendered_flows[..., 1] * (self.rendering.height / self.shape[-2])

        return rendered_flows

    def reset_learning_rate(self):
        for param_group in self.optimizer_non_positional_parameters.param_groups:
            param_group['lr'] = self.config.learning_rate
        for param_group in self.optimizer_positional_parameters.param_groups:
            param_group['lr'] = self.config.learning_rate
            if 'name' in param_group.keys():
                if param_group['name'] in ['axes_quat', 'half_cosine', 'axis_angle']:
                    param_group['lr'] *= self.config.quaternion_learning_rate_coef
                elif param_group['name'] == 'trans':
                    param_group['lr'] *= self.config.translation_learning_rate_coef

    def log_inference_results(self, best_loss, epoch, frame_losses, joint_loss, losses, encoder_result):

        self.logged_sgd_translations.append(encoder_result.translations)
        self.logged_sgd_quaternions.append(encoder_result.quaternions)

        frame_losses.append(float(joint_loss))
        self.write_into_tensorboard_logs(joint_loss, losses, epoch)
        if "model" in losses:
            model_loss = losses["model"].mean().item()
        else:
            model_loss = losses["silh"].mean().item()
        if self.config.verbose and epoch % self.config.training_print_status_frequency == 0:
            print("Epoch {:4d}".format(epoch + 1), end=" ")
            for ls in losses:
                print(", {} {:.3f}".format(ls, losses[ls].mean().item()), end=" ")
            print("; joint {:.3f}".format(joint_loss.item()), end='')
            print("; best {:.3f}".format(best_loss),
                  f'lr: {self.optimizer_positional_parameters.param_groups[0]["lr"]}')
        return model_loss

    def infer_model(self, observed_images, observed_segmentations, observed_flows, observed_flows_segmentations,
                    keyframes, flow_frames, encoder_type):

        encoder_result, encoder_result_flow_frames = self.frames_and_flow_frames_inference(keyframes, flow_frames,
                                                                                           encoder_type=encoder_type)
        renders = self.rendering(encoder_result.translations, encoder_result.quaternions, encoder_result.vertices,
                                 self.encoder.face_features, encoder_result.texture_maps, encoder_result.lights)
        flow_result = self.rendering.compute_theoretical_flow(encoder_result, encoder_result_flow_frames)
        theoretical_flow, rendered_flow_segmentation = flow_result
        rendered_flow_segmentation = rendered_flow_segmentation[None]

        # Renormalization compensating for the fact that we render into bounding box that is smaller than the
        # actual image
        theoretical_flow = self.normalize_rendered_flows(theoretical_flow)

        if encoder_type == 'rgb':
            loss_function = self.rgb_loss_function
        else:  # 'deep_features'
            loss_function = self.loss_function

        rendered_silhouettes = renders[0, :, :, -1:]

        loss_result = loss_function.forward(rendered_images=renders, observed_images=observed_images.clone(),
                                            rendered_silhouettes=rendered_silhouettes,
                                            observed_silhouettes=observed_segmentations.clone(),
                                            rendered_flow=theoretical_flow, observed_flow=observed_flows.clone(),
                                            observed_flow_segmentation=observed_flows_segmentations.clone(),
                                            rendered_flow_segmentation=rendered_flow_segmentation,
                                            keyframes_encoder_result=encoder_result,
                                            last_keyframes_encoder_result=self.last_encoder_result)
        losses_all, losses, joint_loss, per_pixel_error = loss_result

        return encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow

    def write_into_tensorboard_logs(self, jloss, losses, sgd_iter):
        dict_tensorboard_values1 = {
            k + '_loss': float(v) for k, v in losses.items()
        }
        dict_tensorboard_values2 = {
            "joint_loss": float(jloss),
            'loss_laplacian_weight': self.config.loss_laplacian_weight,
            'loss_tv_weight': self.config.loss_tv_weight,
            'loss_iou_weight': self.config.loss_iou_weight,
            'loss_dist_weight': self.config.loss_dist_weight,
            'loss_q_weight': self.config.loss_q_weight,
            'loss_texture_change_weight': self.config.loss_texture_change_weight,
            'loss_t_weight': self.config.loss_t_weight,
            'loss_rgb_weight': self.config.loss_rgb_weight,
            'loss_flow_weight': self.config.loss_flow_weight,
            'non_positional_params_lr': self.optimizer_non_positional_parameters.param_groups[0]['lr'],
            'positional_params_lr': self.optimizer_positional_parameters.param_groups[0]['lr']
        }
        dict_tensorboard_values = {**dict_tensorboard_values1, **dict_tensorboard_values2}
        self.write_results.write_into_tensorboard_log(sgd_iter, dict_tensorboard_values)

    def frames_and_flow_frames_inference(self, keyframes, flow_frames, encoder_type='deep_features'):
        joined_frames = sorted(set(keyframes + flow_frames))
        not_optimized_frames = set(flow_frames) - set(keyframes)
        optimized_frames = list(sorted(set(joined_frames) - not_optimized_frames))

        joined_frames_idx = {frame: idx for idx, frame in enumerate(joined_frames)}

        frames_join_idx = [joined_frames_idx[frame] for frame in keyframes]
        flow_frames_join_idx = [joined_frames_idx[frame] for frame in flow_frames]

        if encoder_type == 'rgb':
            encoder = self.rgb_encoder
        elif encoder_type == 'gt_encoder':
            encoder = self.gt_encoder
        else:  # 'deep_features' - Deep features encoder
            encoder = self.encoder

        joined_encoder_result: EncoderResult = encoder(optimized_frames)

        optimized_translations = joined_encoder_result.translations[:, :, joined_frames]
        optimized_quaternions = joined_encoder_result.quaternions[:, joined_frames]

        keyframes_translations = optimized_translations[:, :, frames_join_idx]
        keyframes_quaternions = optimized_quaternions[:, frames_join_idx]
        flow_frames_translations = optimized_translations[:, :, flow_frames_join_idx]
        flow_frames_quaternions = optimized_quaternions[:, flow_frames_join_idx]

        keyframes_tdiff, keyframes_qdiff = encoder.compute_tdiff_qdiff(keyframes, optimized_quaternions[:, -1],
                                                                       joined_encoder_result.quaternions,
                                                                       joined_encoder_result.translations)
        flow_frames_tdiff, flow_frames_qdiff = encoder.compute_tdiff_qdiff(flow_frames, optimized_quaternions[:, -1],
                                                                           joined_encoder_result.quaternions,
                                                                           joined_encoder_result.translations)

        encoder_result = EncoderResult(translations=keyframes_translations,
                                       quaternions=keyframes_quaternions,
                                       vertices=joined_encoder_result.vertices,
                                       texture_maps=joined_encoder_result.texture_maps,
                                       lights=joined_encoder_result.lights,
                                       translation_difference=keyframes_tdiff,
                                       quaternion_difference=keyframes_qdiff)

        encoder_result_flow_frames = EncoderResult(translations=flow_frames_translations,
                                                   quaternions=flow_frames_quaternions,
                                                   vertices=joined_encoder_result.vertices,
                                                   texture_maps=joined_encoder_result.texture_maps,
                                                   lights=joined_encoder_result.lights,
                                                   translation_difference=flow_frames_tdiff,
                                                   quaternion_difference=flow_frames_qdiff)

        # if encoder_type != 'gt_encoder':
        #     quaternion = encoder_result.quaternions
        #     flow_quaternion = encoder_result_flow_frames.quaternions
        #     for i in range(20):
        #         quaternion = quaternion_multiply(quaternion, quaternion)
        #         flow_quaternion = quaternion_multiply(flow_quaternion, flow_quaternion)

        return encoder_result, encoder_result_flow_frames

    def rgb_apply(self, input_batch, segments, observed_flows, flow_segment_masks):
        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)

        for epoch in range(self.config.rgb_iters):
            infer_result = self.infer_model(observed_images=input_batch, observed_segmentations=segments,
                                            observed_flows=observed_flows,
                                            observed_flows_segmentations=flow_segment_masks,
                                            keyframes=self.all_keyframes.keyframes,
                                            flow_frames=self.all_keyframes.flow_keyframes, encoder_type='deep')

            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, theoretical_flow = infer_result

            if epoch < self.config.iterations - 1:
                joint_loss = joint_loss.mean()
                self.rgb_optimizer.zero_grad()
                joint_loss.backward(retain_graph=True)
                self.rgb_optimizer.step()

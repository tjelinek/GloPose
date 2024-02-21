import math
import copy

from dataclasses import dataclass, replace

import imageio
import kaolin
import numpy as np
import os
import time
import torch
import torchvision.transforms as transforms
from kaolin.io.utils import mesh_handler_naive_triangulate
from kornia.geometry.conversions import QuaternionCoeffOrder, angle_axis_to_quaternion
from pathlib import Path
from torch.optim import lr_scheduler
from typing import Optional, Tuple, Any

from OSTrack.S2DNet.s2dnet import S2DNet
from auxiliary_scripts.logging import WriteResults, load_gt_annotations_file
from flow import RAFTFlowProvider, FlowProvider, GMAFlowProvider, MFTFlowProvider, normalize_flow_to_unit_range, \
    MFTEnsembleFlowProvider
from keyframe_buffer import KeyframeBuffer, FrameObservation, FlowObservation
from main_settings import g_ext_folder
from models.encoder import Encoder, EncoderResult
from models.flow_loss_model import LossFunctionWrapper
from models.initial_mesh import generate_face_features
from models.kaolin_wrapper import load_obj
from models.loss import FMOLoss, iou_loss, random_points_from_binary_mask
from models.rendering import RenderingKaolin, infer_normalized_renderings, RenderedFlowResult
from optimization import lsq_lma_custom, levenberg_marquardt_ceres
from segmentations import (PrecomputedTracker, CSRTrack, OSTracker, MyTracker, SyntheticDataGeneratingTracker,
                           BaseTracker)
from tracker_config import TrackerConfig
from utils import consecutive_quaternions_angular_difference, normalize_vertices, normalize_rendered_flows, rad_to_deg


@dataclass
class FrameResult:
    flow_render_result: RenderedFlowResult
    encoder_result: EncoderResult
    # TODO add typing
    renders: Any
    frame_losses: Any
    per_pixel_flow_error: Any


class Tracking6D:

    def __init__(self, config: TrackerConfig, device, write_folder, file0, bbox0, init_mask=None):
        # Encoders and related components
        self.encoder: Optional[Encoder] = None
        self.gt_encoder: Optional[Encoder] = None
        self.rgb_encoder: Optional[Encoder] = None
        self.last_encoder_result = None
        self.last_encoder_result_rgb = None

        # Rendering and mesh related
        self.rendering: Optional[RenderingKaolin] = None
        self.rendering_backview: Optional[RenderingKaolin] = None
        self.faces = None
        self.gt_mesh_prototype = None
        self.gt_texture = None
        self.gt_texture_features = None

        # Features
        self.feat = None
        self.feat_rgb = None

        # Loss functions and optimizers
        self.all_parameters = None
        self.translational_params = None
        self.rotational_params = None
        self.positional_params = None
        self.non_positional_params = None
        self.loss_function: Optional[FMOLoss] = None
        self.rgb_loss_function: Optional[FMOLoss] = None
        self.optimizer_translational_parameters = None
        self.optimizer_rotational_parameters = None
        self.optimizer_positional_parameters = None
        self.optimizer_non_positional_parameters = None
        self.optimizer_all_parameters = None
        self.rgb_optimizer = None

        # Network related
        self.net = None
        self.short_flow_model: Optional[FlowProvider] = None
        self.long_flow_provider: Optional[MFTFlowProvider] = None

        # Ground truth related
        self.gt_rotations = None
        self.gt_translations = None

        # Keyframes
        self.active_keyframes: Optional[KeyframeBuffer] = None
        self.active_keyframes_backview: Optional[KeyframeBuffer] = None

        # Flow tracks
        self.need_to_init_mft = True
        self.need_to_init_mft_backview = True
        self.flow_tracks_inits = [0]

        # Tracker
        self.tracker: Optional[BaseTracker] = None

        # Other utilities and flags
        self.write_results = None
        self.logged_sgd_translations = []
        self.logged_sgd_quaternions = []

        self.shape: Optional[torch.Size] = None
        self.write_folder = Path(write_folder)
        self.config = config
        self.config_copy = copy.deepcopy(self.config)
        self.device = device

        iface_features, ivertices = self.initialize_mesh()
        self.initialize_flow_model()
        self.initialize_feature_extractor()
        self.initialize_gt_texture()
        self.initialize_gt_tracks()
        self.initialize_tracker(bbox0, file0, init_mask)

        self.shape = (self.config.max_width, self.config.max_width)

        torch.backends.cudnn.benchmark = True
        self.initialize_renderer()
        self.initialize_encoders(iface_features, ivertices)

        if self.config.generate_synthetic_observations_if_possible:
            assert self.gt_translations is not None and self.gt_rotations is not None

            self.tracker = SyntheticDataGeneratingTracker(self.config, self.rendering, self.gt_encoder, self.gt_texture,
                                                          self.feat)

            self.tracker_backview = SyntheticDataGeneratingTracker(self.config, self.rendering_backview,
                                                                   self.gt_encoder, self.gt_texture, self.feat)
            # Re-render the images using the synthetic tracker
        images, images_feat, observed_flows_generated, segments = self.get_initial_images(file0, bbox0, init_mask)

        self.initialize_optimizer_and_loss(ivertices)

        if self.config.features == 'deep':
            self.initialize_rgb_encoder(self.faces, iface_features, ivertices, self.shape)

        self.best_model = {"value": 100,
                           "face_features": self.encoder.face_features.detach().clone(),
                           "faces": self.faces,
                           "encoder": copy.deepcopy(self.encoder.state_dict())}
        self.initialize_keyframes()

        if self.config.verbose:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))

    def initialize_renderer(self):
        self.rendering = RenderingKaolin(self.config, self.faces, self.shape[-1], self.shape[-2]).to(self.device)

    def initialize_tracker(self, bbox0, file0, init_mask):
        if type(bbox0) is dict:
            self.tracker = PrecomputedTracker(self.config.image_downsample, self.config.max_width, bbox0, self.feat)
        else:
            if self.config.tracker_type == 'csrt':
                self.tracker = CSRTrack(self.config.image_downsample, self.config.max_width)
            elif self.config.tracker_type == 'ostrack':
                self.tracker = OSTracker(self.config.image_downsample, self.config.max_width)
            else:  # d3s
                self.tracker = MyTracker(self.config.image_downsample, self.config.max_width)

    def initialize_gt_texture(self):
        if self.config.gt_texture_path is not None:
            texture = torch.from_numpy(imageio.v2.imread(Path(self.config.gt_texture_path)))
            texture = texture.permute(2, 0, 1)[None].to(self.device) / 255.0
            if max(texture.shape[-2:]) > self.config.texture_size:
                resize = transforms.Resize(size=self.config.texture_size)
                texture = resize(texture)

            self.gt_texture = texture
        self.gt_texture_features = self.feat(self.gt_texture[None])[0].detach()

    def initialize_gt_tracks(self):
        if self.config.gt_track_path is not None:
            _, gt_rotations, gt_translations = load_gt_annotations_file(self.config.gt_track_path)
            self.gt_rotations = gt_rotations.to(self.device)
            self.gt_translations = gt_translations.to(self.device)

    def initialize_encoders(self, iface_features, ivertices):
        self.encoder = Encoder(self.config, ivertices, iface_features, self.shape[-1], self.shape[-2],
                               self.config.features_channels).to(self.device)

        if not self.config.optimize_texture and self.gt_texture is not None:
            self.encoder.texture_map = self.gt_texture_features

        def set_encoder_poses(encoder, rotations, translations):
            rotation_quaternion = angle_axis_to_quaternion(rotations, order=QuaternionCoeffOrder.WXYZ)
            encoder.quaternion_w = torch.nn.Parameter(rotation_quaternion[..., 0, None])
            encoder.quaternion_x = torch.nn.Parameter(rotation_quaternion[..., 1, None])
            encoder.quaternion_y = torch.nn.Parameter(rotation_quaternion[..., 2, None])
            encoder.quaternion_z = torch.nn.Parameter(rotation_quaternion[..., 3, None])

            encoder.axis_angle_x = torch.nn.Parameter(rotations[..., 0, None])
            encoder.axis_angle_y = torch.nn.Parameter(rotations[..., 1, None])
            encoder.axis_angle_z = torch.nn.Parameter(rotations[..., 2, None])

            encoder.translation = torch.nn.Parameter(translations)

        self.encoder.train()

        if not self.config.optimize_pose:
            if self.gt_rotations is not None and self.gt_translations is not None:
                set_encoder_poses(self.encoder, self.gt_rotations, self.gt_translations)

                # Do not optimize the poses
                for param in [self.encoder.quaternion_w, self.encoder.quaternion_x, self.encoder.quaternion_y,
                              self.encoder.quaternion_z, self.encoder.translation,
                              self.encoder.axis_angle_x, self.encoder.axis_angle_y, self.encoder.axis_angle_z]:
                    param.detach_()
            else:
                raise ValueError("Required not to optimize pose even though no ground truth "
                                 "rotations and translations are provided.")

        #  Ground truth encoder for synthetic data generation
        self.gt_encoder = Encoder(self.config, ivertices, iface_features,
                                  self.shape[-1], self.shape[-2], 3).to(self.device)
        for name, param in self.gt_encoder.named_parameters():
            if isinstance(param, torch.Tensor):
                param.detach_()

        if self.gt_rotations is not None and self.gt_translations is not None:
            set_encoder_poses(self.gt_encoder, self.gt_rotations, self.gt_translations)

        if self.gt_texture is not None:
            self.gt_encoder.gt_texture = self.gt_texture

    def get_initial_images(self, file0, bbox0, init_mask):
        if type(self.tracker) is SyntheticDataGeneratingTracker:
            file0 = 0
        images, segments = self.tracker.init_bbox(file0, bbox0, init_mask)
        images, segments = images.to(self.device), segments.to(self.device)
        images_feat = self.feat(images).detach()
        observed_flows = torch.zeros(1, 0, 2, *segments.shape[-2:])
        return images, images_feat, observed_flows, segments

    def initialize_optimizer_and_loss(self, ivertices):
        self.all_parameters = set(list(self.encoder.parameters()))
        self.translational_params = {self.encoder.translation}
        self.rotational_params = {self.encoder.quaternion_w, self.encoder.quaternion_x,
                                  self.encoder.quaternion_y, self.encoder.quaternion_z}
        self.positional_params = self.translational_params | self.rotational_params
        # rotational_params = [
        #     {'params': [self.encoder.quaternion_x, self.encoder.quaternion_y, self.encoder.quaternion_z],
        #      'lr': self.config.learning_rate,
        #      'name': 'axes_quat'},
        #     {'params': [self.encoder.axis_angle_x, self.encoder.axis_angle_y, self.encoder.axis_angle_z],
        #      'lr': self.config.learning_rate * 1e-0,
        #      'name': 'axis_angle'},
        #     {'params': [self.encoder.quaternion_w],
        #      'lr': self.config.learning_rate * 1e-0,
        #      'name': 'half_cosine'}
        # ]
        self.non_positional_params = self.all_parameters - self.positional_params
        # positional_params = [
        #     {'params': [self.encoder.quaternion_x, self.encoder.quaternion_y, self.encoder.quaternion_z],
        #      'lr': self.config.learning_rate,
        #      'name': 'axes_quat'},
        #     {'params': [self.encoder.axis_angle_x, self.encoder.axis_angle_y, self.encoder.axis_angle_z],
        #      'lr': self.config.learning_rate * 1e-0,
        #      'name': 'axis_angle'},
        #     {'params': [self.encoder.quaternion_w],
        #      'lr': self.config.learning_rate * 1e-0,
        #      'name': 'half_cosine'},
        #     {'params': list(translational_params),
        #      'lr': self.config.learning_rate * self.config.translation_learning_rate_coef,
        #      'name': 'trans'},
        # ]
        self.optimizer_non_positional_parameters = torch.optim.Adam(self.non_positional_params,
                                                                    lr=self.config.learning_rate)
        self.optimizer_positional_parameters = torch.optim.SGD(self.positional_params, lr=self.config.learning_rate)
        self.optimizer_translational_parameters = torch.optim.SGD(self.translational_params,
                                                                  lr=self.config.learning_rate)
        self.optimizer_all_parameters = torch.optim.Adam(self.all_parameters, lr=self.config.learning_rate)
        self.optimizer_rotational_parameters = torch.optim.SGD(self.rotational_params, lr=self.config.learning_rate)
        self.loss_function = FMOLoss(self.config, ivertices, self.faces).to(self.device)

    def initialize_feature_extractor(self):
        if self.config.features == 'deep':
            self.net = S2DNet(device=self.device, checkpoint_path=g_ext_folder).to(self.device)
            self.feat = lambda x: self.net(x[0])[0][None][:, :, :self.config.features_channels]
            self.feat_rgb = lambda x: x
        else:
            self.feat = lambda x: x

    def initialize_mesh(self):
        if self.config.gt_mesh_path is not None:
            self.gt_mesh_prototype = kaolin.io.obj.import_mesh(str(self.config.gt_mesh_path), with_materials=True)

        if not self.config.optimize_shape:
            ivertices = normalize_vertices(self.gt_mesh_prototype.vertices).numpy()
            self.faces = self.gt_mesh_prototype.faces
            iface_features = self.gt_mesh_prototype.uvs[self.gt_mesh_prototype.face_uvs_idx].numpy()
        elif self.config.initial_mesh_path is not None:
            mesh = load_obj(self.config.initial_mesh_path)
            ivertices = normalize_vertices(mesh.vertices).numpy()
            self.faces = mesh.faces.numpy()
            iface_features = generate_face_features(ivertices, self.faces)
        else:
            mesh = load_obj(os.path.join('./prototypes/sphere.obj'))
            ivertices = normalize_vertices(mesh.vertices).numpy()
            self.faces = mesh.faces.numpy()
            iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()
        return iface_features, ivertices

    def initialize_rgb_encoder(self, faces, iface_features, ivertices, shape):
        config = copy.deepcopy(self.config)
        config.features = 'rgb'

        texture_map_init = None
        if not self.config.optimize_texture and self.gt_texture is not None:
            texture_map_init = self.gt_texture.detach()

        self.rgb_encoder = Encoder(config, ivertices, iface_features, shape[-1], shape[-2],
                                   3, texture_map_init).to(self.device)

        rgb_parameters = [self.rgb_encoder.texture_map]
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

    def initialize_keyframes(self):

        self.active_keyframes = KeyframeBuffer()
        self.active_keyframes_backview = KeyframeBuffer()

    def initialize_flow_model(self):

        short_flow_models = {
            'RAFT': RAFTFlowProvider,
            'GMA': GMAFlowProvider
        }
        long_flow_models = {
            'MFT': MFTFlowProvider,
            'MFTEnsemble': MFTEnsembleFlowProvider,
        }

        # For short_flow_model
        if self.config.short_flow_model in short_flow_models:
            self.short_flow_model = short_flow_models[self.config.short_flow_model]()
        else:
            # Default case or raise an error if you don't want a default FlowProvider
            raise ValueError(f"Unsupported short flow model: {self.config.short_flow_model}")

        # For long_flow_model
        if self.config.long_flow_model in long_flow_models:
            self.long_flow_provider = long_flow_models[self.config.long_flow_model](self.config.MFT_backbone_cfg)
        else:
            raise ValueError(f"Unsupported long flow model: {self.config.long_flow_model}")

    def run_tracking(self, files, bboxes):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1
        if type(bboxes) is dict:
            sorted_bb_keys = sorted(list(bboxes.keys()))
            bboxes = {i: bboxes[sorted_bb_keys[i]] for i, key in zip(range(len(bboxes)), sorted_bb_keys)}

        our_losses = -np.ones((files.shape[0] - 1, 1))

        self.write_results = WriteResults(write_folder=self.write_folder, shape=self.shape, num_frames=files.shape[0],
                                          tracking_config=self.config)

        new_frame_observation = self.tracker.next(0)
        new_frame_observation_from_back = self.tracker_backview.next(0)
        self.active_keyframes.add_new_keyframe_observation(new_frame_observation, 0)
        self.active_keyframes_backview.add_new_keyframe_observation(new_frame_observation_from_back, 0)

        self.last_encoder_result_rgb = self.rgb_encoder(self.active_keyframes.keyframes)
        self.last_encoder_result = self.encoder(self.active_keyframes.keyframes)

        for stepi in range(1, self.config.input_frames):

            if type(self.tracker) is SyntheticDataGeneratingTracker:
                next_tracker_frame = stepi  # Index of a frame
            else:
                next_tracker_frame = files[stepi]  # Name of file

            new_frame_observation = self.tracker.next(next_tracker_frame)

            self.active_keyframes.add_new_keyframe_observation(new_frame_observation, stepi)

            new_frame_observation_backview = self.tracker.next(next_tracker_frame)
            self.active_keyframes_backview.add_new_keyframe_observation(new_frame_observation_backview, stepi)

            start = time.time()

            b0 = [0, self.shape[-1], 0, self.shape[-2]]
            self.rendering = RenderingKaolin(self.config, self.faces, b0[3] - b0[2], b0[1] - b0[0]).to(self.device)

            with torch.no_grad():
                if self.config.add_flow_arcs_strategy == 'all-previous':
                    flow_arcs = {(flow_source, stepi) for flow_source in
                                 set(self.active_keyframes.flow_frames) | set(self.active_keyframes.keyframes)
                                 if flow_source < stepi}
                else:  # self.config.add_flow_arcs_strategy == 'single-previous'
                    flow_arcs = {(stepi - 1, stepi)}

                for flow_arc in flow_arcs:
                    flow_source_frame, flow_target_frame = flow_arc
                    observed_flow, occlusions, uncertainties = self.next_gt_flow(flow_source_frame, flow_target_frame)

                    self.active_keyframes.add_new_flow(observed_flow, new_frame_observation.observed_segmentation,
                                                       occlusions, uncertainties, flow_source_frame, flow_target_frame)

                if self.long_flow_provider is not None:
                    long_flow_arc = (self.flow_tracks_inits[-1], stepi)
                    flow_source_frame, flow_target_frame = long_flow_arc

                    already_present = long_flow_arc in flow_arcs

                    flow_arcs |= {(flow_source_frame, flow_target_frame)}

                    observed_flow, occlusions, uncertainties = self.next_gt_flow(flow_source_frame, flow_target_frame,
                                                                                 mode='long')
                    if not already_present:
                        self.active_keyframes.add_new_flow(observed_flow, new_frame_observation.observed_segmentation,
                                                           occlusions, uncertainties, flow_source_frame,
                                                           flow_target_frame)

                    if not already_present:
                        self.active_keyframes_backview.add_new_flow(observed_flow,
                                                                    new_frame_observation_backview.observed_segmentation,
                                                                    occlusions, uncertainties, flow_source_frame,
                                                                    flow_target_frame)

            active_buffer_indices = list(range(len(self.active_keyframes.keyframes)))

            self.last_encoder_result = EncoderResult(*[tensor.clone()
                                                       if tensor is not None else None for tensor in
                                                       self.encoder(self.active_keyframes.keyframes)])
            self.last_encoder_result_rgb = EncoderResult(*[tensor.clone()
                                                           if tensor is not None else None for tensor in
                                                           self.rgb_encoder(self.active_keyframes.keyframes)])

            all_frame_observations: FrameObservation = \
                self.active_keyframes.get_observations_for_all_keyframes(bounding_box=b0)
            all_flow_observations: FlowObservation = self.active_keyframes.get_flows_observations(bounding_box=b0)
            flow_arcs = sorted(self.active_keyframes.G.edges(), key=lambda x: x[::-1])

            all_frame_observations_backview: FrameObservation = \
                self.active_keyframes.get_observations_for_all_keyframes(bounding_box=b0)
            all_flow_observations_backview: FlowObservation = self.active_keyframes_backview.get_flows_observations(
                bounding_box=b0)

            all_frame_observations = replace(
                all_frame_observations,
                observed_image=torch.cat(
                    [all_frame_observations.observed_image, all_frame_observations_backview.observed_image], dim=1),
                observed_image_features=torch.cat([all_frame_observations.observed_image_features,
                                                   all_frame_observations_backview.observed_image_features], dim=1),
                observed_segmentation=torch.cat([all_frame_observations.observed_segmentation,
                                                 all_frame_observations_backview.observed_segmentation], dim=1)
            )

            all_flow_observations = replace(
                all_flow_observations,
                observed_flow=torch.cat(
                    [all_flow_observations.observed_flow, all_flow_observations_backview.observed_flow], dim=1),
                observed_flow_occlusion=torch.cat([all_flow_observations.observed_flow_occlusion,
                                                   all_flow_observations_backview.observed_flow_occlusion], dim=1),
                observed_flow_segmentation=torch.cat([all_flow_observations.observed_flow_segmentation,
                                                      all_flow_observations_backview.observed_flow_segmentation],
                                                     dim=1),
                observed_flow_uncertainty=torch.cat([all_flow_observations.observed_flow_uncertainty,
                                                     all_flow_observations_backview.observed_flow_uncertainty], dim=1)
            )

            frame_result = self.apply(all_frame_observations, all_flow_observations, self.active_keyframes.keyframes,
                                      self.active_keyframes.flow_frames, flow_arcs, frame_index=stepi)

            encoder_result = frame_result.encoder_result

            silh_losses = np.array(self.best_model["losses"]["silh"])

            our_losses[stepi - 1] = silh_losses[-1]
            print('Elapsed time in seconds: ', time.time() - start, "Frame ", stepi, "out of",
                  self.config.input_frames)

            tex = None
            if self.config.features == 'deep':
                self.rgb_apply(self.active_keyframes.keyframes, self.active_keyframes.flow_frames, flow_arcs,
                               all_frame_observations, all_flow_observations, frame_result.frame_losses)
                tex = torch.nn.Sigmoid()(self.rgb_encoder.texture_map)

            if self.config.write_results:
                with torch.no_grad():
                    new_flow_arcs = [arc for arc in flow_arcs if arc[1] == stepi]

                    self.write_results.write_results(bounding_box=b0, our_losses=our_losses, frame_i=stepi,
                                                     encoder_result=encoder_result, tex=tex,
                                                     new_flow_arcs=new_flow_arcs, frame_result=frame_result,
                                                     active_keyframes=self.active_keyframes,
                                                     active_keyframes_backview=self.active_keyframes_backview,
                                                     logged_sgd_translations=self.logged_sgd_translations,
                                                     logged_sgd_quaternions=self.logged_sgd_quaternions,
                                                     deep_encoder=self.encoder, rgb_encoder=self.rgb_encoder,
                                                     renderer=self.rendering, renderer_backview=self.rendering_backview,
                                                     best_model=self.best_model, observations=all_frame_observations,
                                                     observations_backview=all_frame_observations_backview,
                                                     gt_encoder=self.gt_encoder)

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

            angles = consecutive_quaternions_angular_difference(encoder_result.quaternions)
            # angles = consecutive_quaternions_angular_difference2(encoder_result.quaternions)
            print("Angles:", angles)

            if self.config.points_fraction_visible_new_track is not None:
                longest_flow = self.active_keyframes.get_flows_between_frames(self.flow_tracks_inits[-1], stepi)
                last_observed_segmentation = longest_flow.observed_flow_segmentation[:, -1:, -1:]
                last_observed_occlusion = longest_flow.observed_flow_occlusion[:, -1:]
                fraction_points_visible = float(iou_loss(last_observed_segmentation, last_observed_occlusion))

                print(f"Fraction points not occluded {fraction_points_visible} flow tracks beginnings at "
                      f"{self.flow_tracks_inits}")
                if fraction_points_visible < self.config.points_fraction_visible_new_track:
                    self.flow_tracks_inits.append(stepi)
                    self.need_to_init_mft = True

        return self.best_model

    def next_gt_flow(self, flow_source_frame, flow_target_frame, mode='short'):
        occlusion = None
        uncertainty = None
        if self.config.gt_flow_source == 'GenerateSynthetic':
            keyframes = [flow_target_frame]
            flow_frames = [flow_source_frame]
            flow_arcs_indices = [(0, 0)]

            encoder_result, enc_flow = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

            observed_renderings = self.rendering.compute_theoretical_flow(encoder_result, enc_flow, flow_arcs_indices)

            observed_flow = observed_renderings.theoretical_flow.detach()
            observed_flow = normalize_rendered_flows(observed_flow, self.rendering.width, self.rendering.height,
                                                     self.shape[-1], self.shape[-2])
            # occlusion = observed_renderings.rendered_flow_occlusion.detach()

        elif self.config.gt_flow_source == 'FlowNetwork':

            last_keyframe_observations = self.active_keyframes.get_observations_for_keyframe(flow_target_frame)
            last_flowframe_observations = self.active_keyframes.get_observations_for_keyframe(flow_source_frame)

            last_keyframe_observations_back = self.active_keyframes_backview.get_observations_for_keyframe(flow_target_frame)
            last_flowframe_observations_back = self.active_keyframes_backview.get_observations_for_keyframe(flow_source_frame)

            image_new_x255 = last_keyframe_observations.observed_image.float() * 255
            image_prev_x255 = last_flowframe_observations.observed_image.float() * 255

            image_new_x255_back = last_keyframe_observations_back.observed_image.float() * 255
            image_prev_x255_back = last_flowframe_observations_back.observed_image.float() * 255

            if mode == 'long':
                assert (flow_source_frame == self.flow_tracks_inits[-1] and
                        flow_target_frame == max(self.active_keyframes.keyframes))

                if self.config.matching_target_to_backview:
                    template = torch.cat([image_prev_x255, image_prev_x255_back], dim=-1)
                    target = torch.cat([image_new_x255, image_new_x255_back], dim=-1)
                else:
                    template = image_prev_x255
                    target = image_new_x255

                if self.need_to_init_mft and self.flow_tracks_inits[-1] == flow_source_frame:
                    self.long_flow_provider.init(template)
                    self.need_to_init_mft = False

                observed_flow, occlusion, uncertainty = self.long_flow_provider.next_flow(template, target)

            elif mode == 'short':
                observed_flow = self.short_flow_modl(image_prev_x255, image_new_x255)
            else:
                raise ValueError("Unknown mode")

            observed_flow = normalize_flow_to_unit_range(observed_flow)

        else:
            raise ValueError("'gt_flow_source' must be either 'GenerateSynthetic' or 'FlowNetwork'")

        if occlusion is None:
            occlusion = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)
        if uncertainty is None:
            uncertainty = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)

        return observed_flow, occlusion, uncertainty

    def apply(self, observations, flow_observations, keyframes, flow_frames, flow_arcs, frame_index) -> FrameResult:

        self.config.loss_fl_not_obs_rend_weight = self.config.loss_flow_weight
        self.config.loss_fl_obs_and_rend_weight = self.config.loss_flow_weight

        self.logged_sgd_quaternions = []
        self.logged_sgd_translations = []

        # Updates offset of the next rotation
        self.encoder.compute_next_offset(frame_index)

        self.write_results.set_tensorboard_log_for_frame(frame_index)

        frame_losses = []
        # Restore the learning rate on its prior values
        self.reset_learning_rate()

        if self.config.use_lr_scheduler:
            self.config.loss_rgb_weight = 0
            if frame_index <= 2:
                self.config.loss_flow_weight = 0
            else:
                self.config.loss_flow_weight = self.config_copy.loss_flow_weight

        scheduler_positional_params = lr_scheduler.ReduceLROnPlateau(self.optimizer_positional_parameters,
                                                                     mode='min', factor=0.8,
                                                                     patience=self.config.lr_scheduler_patience,
                                                                     verbose=False)

        def lambda_schedule(epoch_):
            return 1 / (1 + np.exp(-0.25 * (epoch_ - self.config.optimize_non_positional_params_after)))

        scheduler_non_positional_params = lr_scheduler.LambdaLR(self.optimizer_non_positional_parameters,
                                                                lambda_schedule)

        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0

        # rotation_quaternion = angle_axis_to_quaternion(self.gt_rotations, order=QuaternionCoeffOrder.WXYZ)
        # self.encoder.quaternion_offsets = rotation_quaternion.clone()
        # self.encoder.translation_offsets = self.gt_translations.clone()

        no_improvements = 0
        epoch = 0
        loss_improvement_threshold = 1e-4

        # First inference just to log the results
        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result
        self.log_inference_results(self.best_model["value"], epoch, frame_losses, joint_loss, losses, encoder_result)

        if self.config.preinitialization_method is not None:
            print("Pre-initializing the objects position")
        # First optimize the positional parameters first while preventing steps that increase the loss
        if self.config.preinitialization_method == 'levenberg-marquardt':
            self.run_levenberg_marquardt_method(observations, flow_observations, flow_frames, keyframes, flow_arcs,
                                                frame_losses)
        elif self.config.preinitialization_method == 'essential_matrix_decomposition':
            self.essential_matrix_pre_initialization(observations, flow_observations, flow_arcs,
                                                     frame_losses)

        elif self.config.preinitialization_method == 'gradient_descent':
            self.coordinate_descent_with_linear_lr_schedule(observations, flow_observations, epoch, keyframes,
                                                            flow_frames, flow_arcs, frame_losses,
                                                            loss_improvement_threshold)
        elif self.config.preinitialization_method == 'coordinate_descent':
            self.gradient_descent_with_linear_lr_schedule(observations, flow_observations, epoch, frame_losses,
                                                          keyframes, flow_frames, flow_arcs, loss_improvement_threshold,
                                                          no_improvements)

        self.encoder.load_state_dict(self.best_model["encoder"])

        # Now optimize all the parameters jointly using normal gradient descent
        print("Optimizing all parameters")

        # self.reset_learning_rate()

        for epoch in range(epoch, self.config.iterations):

            infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                            'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

            model_loss = self.log_inference_results(self.best_model["value"], epoch, frame_losses, joint_loss,
                                                    losses, encoder_result)
            if abs(model_loss - self.best_model["value"]) > 1e-3:
                iters_without_change = 0
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
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
                self.optimizer_all_parameters.zero_grad()

                self.optimizer_positional_parameters.zero_grad()
                self.optimizer_rotational_parameters.zero_grad()
                self.optimizer_translational_parameters.zero_grad()
                self.optimizer_non_positional_parameters.zero_grad()

                joint_loss.backward()

                self.optimizer_all_parameters.step()

                if self.config.use_lr_scheduler:
                    scheduler_positional_params.step(joint_loss)
                    scheduler_non_positional_params.step()

        self.encoder.load_state_dict(self.best_model["encoder"])

        if (self.config.visualize_loss_landscape and
                (frame_index in {0, 1, 2, 3} or frame_index % self.config.loss_landscape_visualization_frequency == 0)):
            self.write_results.visualize_loss_landscape(observations, flow_observations, self, frame_index,
                                                        relative_mode=True)

        # Inferring the most up-to date state after the optimization is finished
        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

        frame_result = FrameResult(flow_render_result=rendered_flow_result,
                                   encoder_result=encoder_result,
                                   renders=renders,
                                   frame_losses=frame_losses,
                                   per_pixel_flow_error=per_pixel_error)

        return frame_result

    def run_levenberg_marquardt_method(self, observations: FrameObservation, flow_observations: FlowObservation,
                                       flow_frames, keyframes, flow_arcs, frame_losses):
        loss_coefs_names = [
            'loss_laplacian_weight', 'loss_tv_weight', 'loss_iou_weight',
            'loss_dist_weight', 'loss_q_weight', 'loss_texture_change_weight',
            'loss_t_weight', 'loss_rgb_weight', 'loss_flow_weight'
        ]

        # We only care about the flow loss at the moment
        for field_name in loss_coefs_names:
            if field_name != "loss_flow_weight":
                setattr(self.config, field_name, 0)

        observed_images = observations.observed_image
        observed_segmentations = observations.observed_segmentation
        observed_flows = flow_observations.observed_flow
        observed_flows_segmentations = flow_observations.observed_flow_segmentation
        observed_flows_occlusion = flow_observations.observed_flow_occlusion
        observed_flows_uncertainty = flow_observations.observed_flow_uncertainty

        flow_arcs_indices = [(flow_frames.index(pair[0]), keyframes.index(pair[1])) for pair in flow_arcs]

        self.best_model["value"] = 100

        encoder_result, encoder_result_flow_frames = self.encoder.frames_and_flow_frames_inference(keyframes,
                                                                                                   flow_frames)
        kf_translations = encoder_result.translations[0].detach()
        kf_quaternions = encoder_result.quaternions.detach()
        trans_quats = torch.cat([kf_translations, kf_quaternions], dim=-1).squeeze().flatten()

        flow_loss_model = LossFunctionWrapper(encoder_result, encoder_result_flow_frames, self.encoder, self.rendering,
                                              flow_arcs_indices, self.loss_function, observed_flows, observed_flows_segmentations,
                                              self.rendering.width, self.rendering.height, self.shape[-1],
                                              self.shape[-2])

        fun = flow_loss_model.forward
        jac_function = None
        if self.config.use_custom_jacobian:
            jac_function = flow_loss_model.compute_jacobian
        if self.config.levenberg_marquardt_implementation == 'ceres':
            coefficients_list = levenberg_marquardt_ceres(p=trans_quats, cost_function=fun,
                                                          num_residuals=self.config.flow_sgd_n_samples * len(flow_arcs))
        elif self.config.levenberg_marquardt_implementation == 'custom':
            coefficients_list = lsq_lma_custom(p=trans_quats, function=fun, args=(), jac_function=jac_function,
                                               max_iter=self.config.levenberg_marquardt_max_ter)
        else:
            raise ValueError("'levenberg_marquardt_implementation' must be either 'custom' or 'ceres'")

        for epoch in range(len(coefficients_list)):
            trans_quats = coefficients_list[epoch]
            trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 7, 7))

            row_translation = trans_quats[None, :, :, :3]
            row_quaternion = trans_quats[:, :, 3:]
            encoder_result = encoder_result._replace(translations=row_translation, quaternions=row_quaternion)

            inference_result = infer_normalized_renderings(self.rendering, self.encoder.face_features, encoder_result,
                                                           encoder_result_flow_frames, flow_arcs_indices,
                                                           self.shape[-1], self.shape[-2])
            renders, rendered_silhouettes, rendered_flow_result = inference_result

            loss_result = self.loss_function.forward(rendered_images=renders,
                                                     observed_images=observed_images,
                                                     rendered_silhouettes=rendered_silhouettes,
                                                     observed_silhouettes=observed_segmentations,
                                                     rendered_flow=rendered_flow_result.theoretical_flow,
                                                     observed_flow=observed_flows,
                                                     observed_flow_segmentation=observed_flows_segmentations,
                                                     rendered_flow_segmentation=rendered_flow_result.rendered_flow_segmentation,
                                                     observed_flow_occlusion=observed_flows_occlusion,
                                                     rendered_flow_occlusion=rendered_flow_result.rendered_flow_occlusion,
                                                     observed_flow_uncertainties=observed_flows_uncertainty,
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
            self.log_inference_results(joint_loss, epoch, frame_losses, joint_loss, losses, encoder_result,
                                       write_all=True)

        for field_name in loss_coefs_names:
            if field_name != "loss_flow_weight":
                setattr(self.config, field_name, getattr(self.config_copy, field_name))

        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')
        return infer_result

    def gradient_descent_with_linear_lr_schedule(self, observations, flow_observations, epoch, frame_losses, keyframes,
                                                 flow_frames, flow_arcs, loss_improvement_threshold, no_improvements):
        best_loss = math.inf
        while no_improvements < self.config.break_sgd_after_iters_with_no_change:
            self.config.loss_fl_not_obs_rend_weight = self.config.loss_flow_weight
            self.config.loss_fl_obs_and_rend_weight = self.config.loss_flow_weight

            infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                            'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

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
                model_loss = self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses,
                                                        encoder_result)
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.optimizer_positional_parameters.step()
                epoch += 1
                no_improvements += 1

        self.encoder.load_state_dict(self.best_model["encoder"])
        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')
        return infer_result

    def coordinate_descent_with_linear_lr_schedule(self, observations, flow_observations, epoch, keyframes, flow_frames,
                                                   flow_arcs, frame_losses, loss_improvement_threshold):
        no_improvements = 0
        best_loss = math.inf

        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')
        encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result
        joint_loss = joint_loss.mean()
        self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())

        self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses, encoder_result)

        while no_improvements < self.config.break_sgd_after_iters_with_no_change:

            # TODO inferring the model three times per step is ubiquitous, it is sufficient to remember the values
            # TODO from the last iteration, and if we revert to the latest checkpoint, one can save the values from
            # TODO the checkpoint as well

            infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                            'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

            joint_loss = joint_loss.mean()

            self.optimizer_rotational_parameters.zero_grad()
            joint_loss.backward()
            self.optimizer_rotational_parameters.step()

            infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                            'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

            joint_loss = joint_loss.mean()

            self.optimizer_translational_parameters.zero_grad()
            joint_loss.backward()
            self.optimizer_translational_parameters.step()

            infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                            'deep_features')
            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result
            joint_loss = joint_loss.mean()

            loss_improvement = best_loss - joint_loss

            if loss_improvement >= 0:
                best_loss = joint_loss

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
                model_loss = self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, losses,
                                                        encoder_result)
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all

            elif loss_improvement < 0:
                self.encoder.load_state_dict(self.best_model["encoder"])
                for param_group in self.optimizer_translational_parameters.param_groups:
                    param_group['lr'] /= 2.0
                for param_group in self.optimizer_rotational_parameters.param_groups:
                    param_group['lr'] /= 2.0
                for param_group in self.optimizer_positional_parameters.param_groups:
                    param_group['lr'] /= 2.0
                no_improvements += 1

        self.encoder.load_state_dict(self.best_model["encoder"])

        infer_result = self.infer_model(observations, flow_observations, keyframes, flow_frames, flow_arcs,
                                        'deep_features')

        return infer_result

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

    def log_inference_results(self, best_loss, epoch, frame_losses, joint_loss, losses, encoder_result,
                              write_all=False):

        self.logged_sgd_translations.append(encoder_result.translations.detach().clone())
        self.logged_sgd_quaternions.append(encoder_result.quaternions.detach().clone())
        joint_loss = joint_loss.detach().clone()

        frame_losses.append(float(joint_loss))
        self.write_into_tensorboard_logs(joint_loss, losses, epoch)
        if "model" in losses:
            model_loss = losses["model"].mean().item()
        else:
            model_loss = losses["silh"].mean().item()
        if self.config.verbose and (epoch % self.config.training_print_status_frequency == 0 or write_all):
            print("Epoch {:4d}".format(epoch + 1), end=" ")
            for ls in losses:
                print(", {} {:.3f}".format(ls, losses[ls].mean().item()), end=" ")
            print("; joint {:.3f}".format(joint_loss.item()), end='')
            print("; best {:.3f}".format(best_loss),
                  f'lr: {self.optimizer_positional_parameters.param_groups[0]["lr"]}')
        return model_loss

    def infer_model(self, observations: FrameObservation, flow_observations: FlowObservation, keyframes, flow_frames,
                    flow_arcs, encoder_type) -> Tuple[EncoderResult, Any, Any, Any, Any, Any, RenderedFlowResult]:

        if encoder_type == 'rgb':
            encoder = self.rgb_encoder
        elif encoder_type == 'gt_encoder':
            encoder = self.gt_encoder
        else:  # 'deep_features' - Deep features encoder
            encoder = self.encoder

        encoder_result, encoder_result_flow_frames = encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        flow_arcs_indices = [(flow_frames.index(pair[0]), keyframes.index(pair[1])) for pair in flow_arcs]
        flow_arcs_indices_sorted = self.sort_flow_arcs_indices(flow_arcs_indices)

        inference_result = infer_normalized_renderings(self.rendering, self.encoder.face_features, encoder_result,
                                                       encoder_result_flow_frames, flow_arcs_indices_sorted,
                                                       self.shape[-1], self.shape[-2])
        inference_result_backview = infer_normalized_renderings(self.rendering_backview, self.encoder.face_features,
                                                                encoder_result, encoder_result_flow_frames,
                                                                flow_arcs_indices_sorted, self.shape[-1],
                                                                self.shape[-2])

        renders, rendered_silhouettes, rendered_flow_result = inference_result
        renders_backview, rendered_silhouettes_backview, rendered_flow_result_backview = inference_result_backview

        renders = torch.cat([renders, renders_backview], dim=1)
        rendered_silhouettes = torch.cat([rendered_silhouettes, rendered_silhouettes_backview], dim=1)

        rendered_flow_result = rendered_flow_result._replace(theoretical_flow=torch.cat(
            [rendered_flow_result.theoretical_flow,
             rendered_flow_result_backview.theoretical_flow],
            dim=1),
            rendered_flow_segmentation=torch.cat(
                [rendered_flow_result.rendered_flow_segmentation,
                 rendered_flow_result_backview.rendered_flow_segmentation],
                dim=1),
            rendered_flow_occlusion=torch.cat(
                [rendered_flow_result.rendered_flow_occlusion,
                 rendered_flow_result_backview.rendered_flow_occlusion],
                dim=1)
        )

        if encoder_type == 'rgb':
            loss_function = self.rgb_loss_function
            observed_images = observations.observed_image
        else:  # 'deep_features'
            loss_function = self.loss_function
            observed_images = observations.observed_image_features

        loss_result = loss_function.forward(rendered_images=renders, observed_images=observed_images,
                                            rendered_silhouettes=rendered_silhouettes,
                                            observed_silhouettes=observations.observed_segmentation,
                                            rendered_flow=rendered_flow_result.theoretical_flow,
                                            observed_flow=flow_observations.observed_flow,
                                            observed_flow_segmentation=flow_observations.observed_flow_segmentation,
                                            rendered_flow_segmentation=rendered_flow_result.rendered_flow_segmentation,
                                            observed_flow_occlusion=flow_observations.observed_flow_occlusion,
                                            rendered_flow_occlusion=rendered_flow_result.rendered_flow_occlusion,
                                            observed_flow_uncertainties=flow_observations.observed_flow_uncertainty,
                                            keyframes_encoder_result=encoder_result,
                                            last_keyframes_encoder_result=self.last_encoder_result)
        losses_all, losses, joint_loss, per_pixel_error = loss_result

        return encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result

    @staticmethod
    def sort_flow_arcs_indices(flow_arcs_indices):
        flow_arcs_indices_sorted = sorted(flow_arcs_indices)
        return flow_arcs_indices_sorted

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

    def rgb_apply(self, keyframes, flow_frames, flow_arcs, observations: FrameObservation,
                  flow_observations: FlowObservation, frame_losses):
        start_time = time.time()

        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)

        print("Texture optimization")
        epoch = 0
        for epoch in range(self.config.rgb_iters):
            infer_result = self.infer_model(observations, flow_observations, keyframes=keyframes,
                                            flow_frames=flow_frames, flow_arcs=flow_arcs, encoder_type='rgb')

            encoder_result, joint_loss, losses, losses_all, per_pixel_error, renders, rendered_flow_result = infer_result

            if epoch % self.config.training_print_status_frequency == 0:
                self.log_inference_results(self.best_model["value"], epoch, frame_losses,
                                           joint_loss, losses, encoder_result)

            joint_loss = joint_loss.mean()
            self.rgb_optimizer.zero_grad()
            joint_loss.backward(retain_graph=True)
            self.rgb_optimizer.step()

        print(f'Elapsed time in seconds: {time.time() - start_time} for total of {epoch + 1} epochs.')

    def essential_matrix_pre_initialization(self, observations: FrameObservation, flow_observations: FlowObservation,
                                            flow_arcs, frame_losses):
        n_samples_for_ransac = 1000
        import pygcransac
        import cv2
        from kornia.geometry import rotation_matrix_to_angle_axis
        K1 = K2 = self.rendering.camera_intrinsics.numpy(force=True)

        height, width = self.shape
        for flow_arc_idx, flow_arc in enumerate(flow_arcs):
            flow_source_frame, flow_target_frame = flow_arc

            not_occluded_binary_mask = ~(flow_observations.observed_flow_occlusion[:, [flow_arc_idx]] >
                                         self.config.occlusion_coef_threshold)
            segmentation_binary_mask = (flow_observations.observed_flow_segmentation[:, [flow_arc_idx]] >
                                        self.config.segmentation_mask_threshold)

            not_occluded_object_points_mask = (not_occluded_binary_mask * segmentation_binary_mask)[0, 0]

            sampled_points_mask = random_points_from_binary_mask(not_occluded_object_points_mask, n_samples_for_ransac)

            optical_flow = flow_observations.observed_flow[0, flow_arc_idx]

            src_pts = torch.nonzero(sampled_points_mask[0])
            dst_pts = optical_flow[:, src_pts[:, 0], src_pts[:, 1]].permute(1, 0) + src_pts

            src_pts_np = src_pts.numpy(force=True).astype(np.float64)
            dst_pts_np = dst_pts.numpy(force=True).astype(np.float64)
            correspondences = np.concatenate([src_pts_np, dst_pts_np], axis=1)

            E, mask = pygcransac.findEssentialMatrix(correspondences, K1, K2, height, width, height, width, 3.0)
            # pose, mask = pygcransac.find6DPose(correspondences, height, width, height, width, 3.0)

            R1, R2, t = cv2.decomposeEssentialMat(E)
            r1 = rotation_matrix_to_angle_axis(torch.from_numpy(R1))
            r2 = rotation_matrix_to_angle_axis(torch.from_numpy(R2))

            r1_deg = rad_to_deg(r1)
            r2_deg = rad_to_deg(r2)

            print(f"r1_deg: {r1_deg}, r2_deg: {r2_deg}")

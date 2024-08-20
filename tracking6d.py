import math
import copy

from dataclasses import replace

import kaolin
import numpy as np
import time
import torch
from kaolin.render.camera import PinholeIntrinsics
from kornia.geometry import Quaternion, Se3
from kornia.geometry.conversions import quaternion_to_axis_angle
from pathlib import Path

from torch.optim import lr_scheduler
from typing import Optional, NamedTuple, List, Callable, Union

from auxiliary_scripts.image_utils import get_shape, ImageShape
from data_structures.pose_icosphere import PoseIcosphere
from pose.epipolar_pose_estimator import EpipolarPoseEstimator
from repositories.OSTrack.S2DNet.s2dnet import S2DNet
from data_structures.data_graph import DataGraph
from auxiliary_scripts.cameras import Cameras
from auxiliary_scripts.logging import WriteResults
from auxiliary_scripts.math_utils import (consecutive_quaternions_angular_difference,
                                          get_object_pose_after_in_plane_rot_in_cam_space,
                                          quaternion_minimal_angular_difference, Se3_epipolar_cam_from_Se3_obj,
                                          Se3_obj_from_epipolar_Se3_cam)
from auxiliary_scripts.flow_provider import (RAFTFlowProvider, FlowProvider, GMAFlowProvider, MFTFlowProvider,
                                             MFTEnsembleFlowProvider, MFTIQFlowProvider, MFTIQSyntheticFlowProvider)
from flow import flow_image_coords_to_unit_coords, normalize_rendered_flows
from data_structures.keyframe_buffer import KeyframeBuffer, FrameObservation, FlowObservation, MultiCameraObservation, \
    SyntheticFlowObservation, generate_rotated_observations
from main_settings import g_ext_folder
from models.encoder import Encoder, EncoderResult
from models.flow_loss_model import LossFunctionWrapper
from models.initial_mesh import generate_face_features
from models.loss import FMOLoss, LossResult
from models.rendering import RenderingKaolin, infer_normalized_renderings, RenderedFlowResult
from optimization import lsq_lma_custom, levenberg_marquardt_ceres
from segmentations import SyntheticDataGeneratingTracker, BaseTracker, PrecomputedTracker, \
    PrecomputedTrackerSegmentAnything, PrecomputedTrackerXMem, PrecomputedTrackerSegmentAnything2
from tracker_config import TrackerConfig
from utils import normalize_vertices, pinhole_intrinsics_from_tensor


class InferenceResult(NamedTuple):
    encoder_result: EncoderResult
    loss_result: LossResult
    renders: torch.Tensor
    rendered_flow_result: RenderedFlowResult


class Tracking6D:

    def __init__(self, config: TrackerConfig, write_folder, gt_texture=None, gt_mesh=None, gt_rotations=None,
                 gt_translations=None, images_paths: List[Path] = None, segmentation_paths: List[Path] = None,
                 cam_intrinsics: torch.Tensor = None):
        # Encoders and related components
        self.encoder: Optional[Encoder] = None
        self.gt_encoder: Optional[Encoder] = None
        self.rgb_encoder: Optional[Encoder] = None
        self.last_encoder_result = None
        self.last_encoder_result_rgb = None

        # Rendering and mesh related
        self.rendering: Optional[RenderingKaolin] = None
        self.faces = None
        self.gt_mesh_prototype: Optional[kaolin.rep.SurfaceMesh] = gt_mesh
        self.gt_texture = gt_texture
        self.gt_texture_features = None

        # External camera
        self.cam_intrinsics: Optional[PinholeIntrinsics] = None

        # Features
        self.feat: Optional[Callable] = None
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

        # Feature extraction
        self.feature_extractor: Optional[S2DNet] = None

        # Optical flow
        self.short_flow_model: Optional[Union[FlowProvider, MFTFlowProvider]] = None
        self.long_flow_provider: Optional[MFTFlowProvider] = None

        # Ground truth related
        assert torch.all(gt_rotations.eq(0)) or config.rot_init is None  # Conflicting setting handling
        assert torch.all(gt_translations.eq(0)) or config.tran_init is None  # Conflicting setting handling

        config.rot_init = tuple(gt_rotations[0].numpy(force=True))
        config.tran_init = tuple(gt_translations[0].numpy(force=True))

        self.gt_rotations: Optional[torch.Tensor] = gt_rotations
        self.gt_translations: Optional[torch.Tensor] = gt_translations

        # Keyframes
        self.active_keyframes: Optional[KeyframeBuffer] = None

        # Data graph
        self.data_graph: Optional[DataGraph] = None

        # Flow tracks
        self.flow_tracks_inits = [0]
        self.pose_icosphere: Optional[PoseIcosphere] = PoseIcosphere()

        # Tracker
        self.tracker: Optional[BaseTracker] = None

        # Other utilities and flags
        self.write_results = None
        self.logged_sgd_translations = []
        self.logged_sgd_quaternions = []

        self.image_shape: Optional[ImageShape] = None
        self.write_folder = Path(write_folder)
        self.config = config
        self.config_copy = copy.deepcopy(self.config)
        self.device = 'cuda'

        iface_features, ivertices = self.initialize_mesh()
        self.initialize_feature_extractor()
        if self.gt_texture is not None:
            self.gt_texture_features = self.feat(self.gt_texture[None])[0].detach()

        if self.config.generate_synthetic_observations_if_possible:
            self.image_shape = ImageShape(width=int(self.config.image_downsample * self.config.max_width),
                                          height=int(self.config.image_downsample * self.config.max_width))
        else:
            self.image_shape = get_shape(images_paths[0], self.config.image_downsample)

        if cam_intrinsics is not None:
            self.cam_intrinsics = pinhole_intrinsics_from_tensor(cam_intrinsics, self.image_shape.width,
                                                                 self.image_shape.height)

        torch.backends.cudnn.benchmark = True
        self.initialize_renderer()
        self.initialize_encoders(iface_features, ivertices)

        if self.config.gt_flow_source != 'GenerateSynthetic':  # This provides a significant speed-up for debugging
            self.initialize_flow_model()

        self.used_cameras = [Cameras.FRONTVIEW]
        if self.config.matching_target_to_backview:
            self.used_cameras.append(Cameras.BACKVIEW)
        self.data_graph = DataGraph(used_cameras=self.used_cameras)

        if self.config.generate_synthetic_observations_if_possible:
            assert self.gt_translations is not None and self.gt_rotations is not None

            self.tracker = SyntheticDataGeneratingTracker(self.config, self.rendering, self.gt_encoder, self.gt_texture,
                                                          self.feat)

        else:
            assert not self.config.matching_target_to_backview

            if self.config.segmentation_tracker == 'precomputed':
                self.tracker = PrecomputedTracker(self.config, self.feat, images_paths, segmentation_paths)
            elif config.segmentation_tracker == 'SAM':
                self.tracker = PrecomputedTrackerSegmentAnything(self.config, self.feat, images_paths,
                                                                 segmentation_paths)
            elif config.segmentation_tracker == 'SAM2':
                self.tracker = PrecomputedTrackerSegmentAnything2(self.config, self.feat, images_paths,
                                                                  segmentation_paths)
            elif config.segmentation_tracker == 'XMem':
                self.tracker = PrecomputedTrackerXMem(self.config, self.feat, images_paths, segmentation_paths)
            else:
                raise ValueError('Unknown value of "segmentation_tracker"')

        self.initialize_optimizer_and_loss(ivertices)

        if self.config.features == 'deep':
            self.initialize_rgb_encoder(self.faces, iface_features, ivertices, self.image_shape)

        self.best_model = {"value": 100.,
                           "face_features": self.encoder.face_features.detach().clone(),
                           "faces": self.faces,
                           "encoder": copy.deepcopy(self.encoder.state_dict())}
        self.initialize_keyframes()

        self.epipolar_pose_estimator = EpipolarPoseEstimator(self.config, self.data_graph, self.gt_rotations,
                                                             self.gt_translations, self.rendering, self.gt_encoder,
                                                             self.cam_intrinsics)

        if self.config.verbose:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))

    def initialize_renderer(self):
        self.rendering = RenderingKaolin(self.config, self.faces, self.image_shape.width,
                                         self.image_shape.height).to(self.device)

    def initialize_encoders(self, iface_features, ivertices):
        self.encoder = Encoder(self.config, ivertices, iface_features, self.image_shape.width,
                               self.image_shape.height, self.config.features_channels).to(self.device)

        if not self.config.optimize_texture and self.gt_texture is not None:
            self.encoder.texture_map = torch.nn.Parameter(self.gt_texture_features)
            self.encoder.texture_map.requires_grad = False

        self.encoder.train()

        if not self.config.optimize_pose:
            if self.gt_rotations is not None and self.gt_translations is not None:
                assert self.gt_rotations.shape == torch.Size((self.config.input_frames, 3))
                assert self.gt_translations.shape == torch.Size((self.config.input_frames, 3))
                self.encoder.set_encoder_poses(self.gt_rotations, self.gt_translations)

                # Do not optimize the poses
                for param in [self.encoder.quaternion, self.encoder.translation]:
                    param.detach_()
            else:
                raise ValueError("Required not to optimize pose even though no ground truth "
                                 "rotations and translations are provided.")

        #  Ground truth encoder for synthetic data generation
        self.gt_encoder = Encoder(self.config, ivertices, iface_features,
                                  self.image_shape.width, self.image_shape.height, 3).to(self.device)
        for name, param in self.gt_encoder.named_parameters():
            if isinstance(param, torch.Tensor):
                param.detach_()

        if self.gt_rotations is not None and self.gt_translations is not None:
            self.gt_encoder.set_encoder_poses(self.gt_rotations, self.gt_translations)

        if self.gt_texture is not None:
            self.gt_encoder.gt_texture = self.gt_texture

    def initialize_optimizer_and_loss(self, ivertices):
        self.all_parameters = set(list(self.encoder.parameters()))
        self.translational_params = {self.encoder.translation}
        self.rotational_params = {self.encoder.quaternion}
        # self.rotational_params = {self.encoder.quaternion.q.q}
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
            self.feature_extractor = S2DNet(device=self.device, checkpoint_path=g_ext_folder).to(self.device)
            self.feat = lambda x: self.feature_extractor(x[0])[0][None][:, :, :self.config.features_channels]
            self.feat_rgb = lambda x: x
        else:
            self.feat = lambda x: x

    def initialize_mesh(self):
        if not self.config.optimize_shape and self.gt_mesh_prototype is not None:
            ivertices = normalize_vertices(self.gt_mesh_prototype.vertices).numpy()
            self.faces = self.gt_mesh_prototype.faces
            iface_features = self.gt_mesh_prototype.uvs[self.gt_mesh_prototype.face_uvs_idx].numpy()
        elif self.config.initial_mesh_path is not None:
            path = self.config.initial_mesh_path
            print("Loading mesh located at", path)
            mesh = kaolin.io.obj.import_mesh(path, with_materials=True)
            ivertices = normalize_vertices(mesh.vertices).numpy()
            self.faces = mesh.faces.numpy()
            iface_features = generate_face_features(ivertices, self.faces)
        else:
            path = Path('./prototypes/sphere.obj')
            print("Loading mesh located at", path)
            mesh = kaolin.io.obj.import_mesh(str(path), with_materials=True)
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

        self.active_keyframes = KeyframeBuffer(storage_device='cpu')

    def initialize_flow_model(self):

        short_flow_models = {
            'RAFT': RAFTFlowProvider,
            'GMA': GMAFlowProvider,
            'MFT': MFTFlowProvider,
            'MFT_Synth': MFTIQSyntheticFlowProvider,
        }
        long_flow_models = {
            'MFT': MFTFlowProvider,
            'MFTEnsemble': MFTEnsembleFlowProvider,
            'MFT_IQ': MFTIQFlowProvider,
            'MFT_Synth': MFTIQSyntheticFlowProvider,
        }

        # For short_flow_model
        if self.config.short_flow_model in short_flow_models:
            self.short_flow_model = short_flow_models[self.config.short_flow_model](self.config.MFT_short_backbone_cfg,
                                                                                    config=self.config,
                                                                                    faces=self.faces,
                                                                                    gt_encoder=self.gt_encoder)
        else:
            # Default case or raise an error if you don't want a default FlowProvider
            raise ValueError(f"Unsupported short flow model: {self.config.short_flow_model}")

        # For long_flow_model
        if self.config.long_flow_model in long_flow_models:
            self.long_flow_provider = long_flow_models[self.config.long_flow_model](self.config.MFT_backbone_cfg,
                                                                                    config=self.config,
                                                                                    faces=self.faces,
                                                                                    gt_encoder=self.gt_encoder
                                                                                    )
        else:
            raise ValueError(f"Unsupported long flow model: {self.config.long_flow_model}")

    def run_tracking(self):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1

        T_world_to_cam = self.rendering.camera_transformation_matrix_4x4()

        our_losses = -np.ones((self.config.input_frames - 1, 1))

        self.write_results = WriteResults(write_folder=self.write_folder, shape=self.image_shape,
                                          tracking_config=self.config, rendering=self.rendering,
                                          gt_encoder=self.gt_encoder, deep_encoder=self.encoder,
                                          rgb_encoder=self.rgb_encoder, data_graph=self.data_graph,
                                          cameras=self.used_cameras)

        self.data_graph.add_new_frame(0)
        self.data_graph.get_frame_data(0).gt_rot_axis_angle = self.gt_rotations[0]
        self.data_graph.get_frame_data(0).gt_translation = self.gt_translations[0]

        initial_predicted_quat = Quaternion.from_axis_angle(self.gt_rotations[[0]])
        initial_predicted_Se3 = Se3(initial_predicted_quat, self.gt_translations[[0]])
        self.data_graph.get_frame_data(0).predicted_object_se3_total = initial_predicted_Se3

        template_frame_observation = self.tracker.next(0)
        self.active_keyframes.add_new_keyframe_observation(template_frame_observation, 0)
        self.data_graph.get_camera_specific_frame_data(0, Cameras.FRONTVIEW).frame_observation = (
            template_frame_observation.send_to_device('cpu'))

        initial_rotation = self.encoder.quaternion_offsets[[0]]

        if self.config.icosphere_add_inplane_rotatiosn:
            self.insert_templates_into_icosphere(T_world_to_cam, template_frame_observation, initial_rotation,
                                                 self.config.icosphere_trust_region_degrees, 0)
        else:
            self.pose_icosphere.insert_new_reference(template_frame_observation, Quaternion(initial_rotation), 0)

        for frame_i in range(1, self.config.input_frames):

            self.data_graph.add_new_frame(frame_i)
            self.data_graph.get_frame_data(frame_i).gt_rot_axis_angle = self.gt_rotations[frame_i]
            self.data_graph.get_frame_data(frame_i).gt_translation = self.gt_translations[frame_i]

            next_tracker_frame = frame_i  # Index of a frame

            new_frame_observation = self.tracker.next(next_tracker_frame)
            self.data_graph.get_camera_specific_frame_data(frame_i, Cameras.FRONTVIEW).frame_observation = (
                new_frame_observation.send_to_device('cpu'))
            self.active_keyframes.add_new_keyframe_observation(new_frame_observation, frame_i)

            start = time.time()

            b0 = (0, self.image_shape.height, 0, self.image_shape.width)
            self.rendering = RenderingKaolin(self.config, self.faces, self.image_shape.width,
                                             self.image_shape.height).to(self.device)

            self.add_new_flows(frame_i)

            all_frame_observations: FrameObservation = \
                self.active_keyframes.get_observations_for_all_keyframes(bounding_box=b0)
            all_flow_observations: FlowObservation = self.active_keyframes.get_flows_observations(bounding_box=b0)

            flow_arcs = sorted(self.active_keyframes.G.edges(), key=lambda x: x[::-1])

            multi_camera_observations = MultiCameraObservation.from_kwargs(
                frontview=all_frame_observations)
            multi_camera_flow_observations = MultiCameraObservation.from_kwargs(
                frontview=all_flow_observations)

            self.apply(multi_camera_observations, multi_camera_flow_observations, self.active_keyframes.keyframes,
                       self.active_keyframes.flow_frames, flow_arcs, frame_index=frame_i)

            silh_losses = np.array(self.best_model["losses"]["silh"])

            our_losses[frame_i - 1] = silh_losses[-1]
            print('Elapsed time in seconds: ', time.time() - start, "Frame ", frame_i, "out of",
                  self.config.input_frames)

            tex = None
            if self.config.features == 'deep':
                if self.config.optimize_texture:
                    self.rgb_apply(self.active_keyframes.keyframes, self.active_keyframes.flow_frames, flow_arcs,
                                   multi_camera_observations, multi_camera_flow_observations)
                tex = torch.nn.Sigmoid()(self.rgb_encoder.texture_map)

            if self.config.write_results:
                new_flow_arcs = [arc for arc in flow_arcs if arc[1] == frame_i]

                self.write_results.write_results(frame_i=frame_i, tex=tex, new_flow_arcs=new_flow_arcs,
                                                 active_keyframes=self.active_keyframes, best_model=self.best_model,
                                                 observations=all_frame_observations, gt_rotations=self.gt_rotations,
                                                 gt_translations=self.gt_translations,
                                                 flow_tracks_inits=self.flow_tracks_inits,
                                                 pose_icosphere=self.pose_icosphere)

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

            angles = consecutive_quaternions_angular_difference(
                self.data_graph.get_frame_data(frame_i).encoder_result.quaternions)

            print("Angles:", angles)

            current_pose = Quaternion(self.encoder.quaternion_offsets[[frame_i]])
            closest_node, angular_dist = self.pose_icosphere.get_closest_reference(current_pose)
            print(
                f">>>>>>>>>>>>>>>>>>>>Angular dist {angular_dist}, closest frame: {closest_node.keyframe_idx_observed}")

            if self.long_flow_provider is not None and 'direct' in self.config.MFT_backbone_cfg:
                self.long_flow_provider.need_to_init = True

            if angular_dist >= 1.1 * self.config.icosphere_trust_region_degrees:  # Need to add a new frame

                self.active_keyframes.remove_frames(self.flow_tracks_inits[:])
                self.encoder.quaternion_offsets[frame_i + 1:] = self.encoder.quaternion_offsets[frame_i]

                if self.config.icosphere_add_inplane_rotatiosn:
                    obj_rotation_q = self.encoder.quaternion_offsets[[frame_i]]

                    self.insert_templates_into_icosphere(T_world_to_cam, new_frame_observation, obj_rotation_q,
                                                         self.config.icosphere_trust_region_degrees, frame_i)

                else:
                    obj_rotation_q = Quaternion(self.encoder.quaternion_offsets[[frame_i]])
                    self.pose_icosphere.insert_new_reference(new_frame_observation, obj_rotation_q, frame_i)

                self.flow_tracks_inits.append(frame_i)

            else:
                self.active_keyframes.remove_frames(self.flow_tracks_inits[:])

                self.active_keyframes.add_new_keyframe_observation(closest_node.observation,
                                                                   closest_node.keyframe_idx_observed)

                self.encoder.quaternion_offsets[frame_i + 1:] = closest_node.quaternion.q

                self.flow_tracks_inits.append(closest_node.keyframe_idx_observed)

            if self.active_keyframes.G.number_of_nodes() > self.config.max_keyframes:
                if self.config.max_keyframes <= 1:
                    nodes_to_remove = sorted(list(self.active_keyframes.G.nodes))[1:]
                else:
                    nodes_to_remove = sorted(list(self.active_keyframes.G.nodes))[1:-self.config.max_keyframes]

                self.active_keyframes.remove_frames(nodes_to_remove)

                print(f"Removed nodes {nodes_to_remove}")

            del all_frame_observations
            del all_flow_observations

        return self.best_model

    def insert_templates_into_icosphere(self, T_world_to_cam_4x4, frame_observation, obj_rotation_q,
                                        degree_delta, frame_i):
        rotated_observations, degrees = generate_rotated_observations(frame_observation, 2 * degree_delta)
        for i, degree in enumerate(degrees):
            q_obj_rotated_world = Quaternion(get_object_pose_after_in_plane_rot_in_cam_space(obj_rotation_q,
                                                                                             T_world_to_cam_4x4,
                                                                                             degree))

            rotated_observation = rotated_observations[i]
            self.pose_icosphere.insert_new_reference(rotated_observation, q_obj_rotated_world, frame_i)

    @torch.no_grad()
    def add_new_flows(self, frame_i):

        def process_flow_arc(flow_source_frame, flow_target_frame, mode=None):
            observed_flow, occlusions, uncertainties = self.next_gt_flow(flow_source_frame, flow_target_frame,
                                                                         mode=mode)

            # Determine the appropriate renderer and keyframes based on the backview flag
            renderer = self.rendering
            active_keyframes = self.active_keyframes

            # Render the flow
            synthetic_flow = renderer.render_flow_for_frame(self.gt_encoder, flow_source_frame, flow_target_frame)

            # Convert synthetic flow results to CPU
            synthetic_flow_cpu = synthetic_flow.send_to_device('cpu')

            # Get the observed segmentation for the flow source frame
            frame_observation = (self.data_graph.get_camera_specific_frame_data(flow_source_frame).
                                 frame_observation.send_to_device('cuda'))
            segment = frame_observation.observed_segmentation

            flow_observation = FlowObservation(observed_flow=observed_flow,
                                               observed_flow_segmentation=segment,
                                               observed_flow_uncertainty=uncertainties,
                                               observed_flow_occlusion=occlusions)

            # Add new flow to active keyframes
            active_keyframes.add_new_flow_observation(flow_observation, flow_source_frame, flow_target_frame)

            # Update the edge data with synthetic flow results
            camera = Cameras.FRONTVIEW
            edge_data = self.data_graph.get_edge_observations(flow_source_frame, flow_target_frame, camera)
            edge_data.synthetic_flow_result = synthetic_flow_cpu
            edge_data.observed_flow = flow_observation.send_to_device('cpu')

        if self.config.add_flow_arcs_strategy == 'single-previous':
            short_flow_arcs = {(frame_i - 1, frame_i)}
        elif self.config.add_flow_arcs_strategy is None:
            short_flow_arcs = set()
        else:
            raise ValueError("Invalid value for 'add_flow_arcs_strategy'.")

        for flow_arc in short_flow_arcs:
            flow_source_frame, flow_target_frame = flow_arc
            self.data_graph.add_new_arc(flow_source_frame, flow_target_frame)
            process_flow_arc(flow_source_frame, flow_target_frame, mode='short')

        if self.config.long_flow_model is not None:
            long_flow_arc = (self.flow_tracks_inits[-1], frame_i)
            if long_flow_arc not in short_flow_arcs:
                flow_source_frame, flow_target_frame = long_flow_arc
                self.data_graph.add_new_arc(flow_source_frame, flow_target_frame)
                process_flow_arc(flow_source_frame, flow_target_frame, mode='long')

    def next_gt_flow(self, flow_source_frame, flow_target_frame, mode='short'):

        if self.config.gt_flow_source == 'GenerateSynthetic':
            keyframes = [flow_target_frame]
            flow_frames = [flow_source_frame]
            flow_arcs_indices = [(0, 0)]

            encoder_result, enc_flow = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

            renderer = self.rendering

            observed_renderings = renderer.compute_theoretical_flow(encoder_result, enc_flow, flow_arcs_indices)

            observed_flow = observed_renderings.theoretical_flow.detach()
            observed_flow = normalize_rendered_flows(observed_flow, self.rendering.width, self.rendering.height,
                                                     self.image_shape.width, self.image_shape.height)
            occlusion = observed_renderings.rendered_flow_occlusion.detach()
            uncertainty = torch.zeros_like(occlusion)

        elif self.config.gt_flow_source == 'FlowNetwork':

            source_frame_obs = self.data_graph.get_camera_specific_frame_data(flow_source_frame).frame_observation
            target_frame_obs = self.data_graph.get_camera_specific_frame_data(flow_target_frame).frame_observation

            target_image = target_frame_obs.observed_image.float() * 255
            template_image = source_frame_obs.observed_image.float() * 255

            if mode == 'long':
                assert (flow_source_frame == self.flow_tracks_inits[-1] and
                        flow_target_frame == max(self.active_keyframes.keyframes))

                if self.long_flow_provider.need_to_init and self.flow_tracks_inits[-1] == flow_source_frame:
                    self.long_flow_provider.init(template_image)
                    self.long_flow_provider.need_to_init = False

                observed_flow, occlusion, uncertainty = self.long_flow_provider.next_flow(template_image, target_image)

            elif mode == 'short':
                if isinstance(self.short_flow_model, MFTFlowProvider):
                    self.short_flow_model.init(template_image)
                    self.short_flow_model.need_to_init = False
                observed_flow, occlusion, uncertainty = self.short_flow_model.next_flow(template_image,
                                                                                        target_image)
            else:
                raise ValueError("Unknown mode")

            observed_flow = flow_image_coords_to_unit_coords(observed_flow)

        else:
            raise ValueError("'gt_flow_source' must be either 'GenerateSynthetic' or 'FlowNetwork'")

        return observed_flow, occlusion, uncertainty

    def apply(self, observations: MultiCameraObservation, flow_observations: MultiCameraObservation,
              keyframes: List, flow_frames: List, flow_arcs: List, frame_index: int) -> None:

        self.config.loss_fl_not_obs_rend_weight = self.config.loss_flow_weight
        self.config.loss_fl_obs_and_rend_weight = self.config.loss_flow_weight

        stacked_observations: FrameObservation = observations.stack()
        stacked_flow_observations: FlowObservation = flow_observations.stack()

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
                                                                     patience=self.config.lr_scheduler_patience)

        def lambda_schedule(epoch_):
            return 1 / (1 + np.exp(-0.25 * (epoch_ - self.config.optimize_non_positional_params_after)))

        scheduler_non_positional_params = lr_scheduler.LambdaLR(self.optimizer_non_positional_parameters,
                                                                lambda_schedule)

        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0

        epoch = 0

        # First inference just to log the results
        with torch.no_grad():
            self.infer_model_and_log_results(flow_arcs, epoch, keyframes, flow_frames, frame_index, frame_losses,
                                             stacked_flow_observations, stacked_observations)

        if self.config.preinitialization_method is not None:
            self.run_preinitializations(flow_arcs, flow_frames, keyframes,
                                        stacked_flow_observations, stacked_observations)

            epoch += 1
            self.infer_model_and_log_results(flow_arcs, epoch, keyframes, flow_frames, frame_index, frame_losses,
                                             stacked_flow_observations, stacked_observations)

        self.encoder.load_state_dict(self.best_model["encoder"])

        # Now optimize all the parameters jointly using normal gradient descent
        print("Optimizing all parameters")

        # self.reset_learning_rate()

        if self.config.run_main_optimization_loop:
            for epoch in range(epoch, self.config.iterations):

                infer_result = self.infer_model(stacked_observations, stacked_flow_observations, keyframes, flow_frames,
                                                flow_arcs, 'deep_features')
                encoder_result, loss_result, renders, rendered_flow_result = infer_result
                loss_result: LossResult = loss_result

                model_loss = self.log_inference_results(self.best_model["value"], epoch, frame_losses, loss_result.loss,
                                                        loss_result.losses, encoder_result, frame_index)
                if abs(model_loss - self.best_model["value"]) > 1e-3:
                    iters_without_change = 0
                    self.best_model["value"] = model_loss
                    self.best_model["losses"] = loss_result.losses_all
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
                    joint_loss = loss_result.loss.mean()
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
        with torch.no_grad():
            infer_result = self.infer_model(stacked_observations, stacked_flow_observations, keyframes, flow_frames,
                                            flow_arcs, 'deep_features')
        encoder_result, loss_result, renders, rendered_flow_result = infer_result

        data_graph_common_frame_data = self.data_graph.get_frame_data(frame_index)

        data_graph_common_frame_data.frame_losses = frame_losses
        data_graph_common_frame_data.encoder_result = encoder_result

        self.infer_model_and_log_results(flow_arcs, epoch, keyframes, flow_frames, frame_index, frame_losses,
                                         stacked_flow_observations, stacked_observations)

    def infer_model_and_log_results(self, flow_arcs, epoch, keyframes, flow_frames, frame_index, frame_losses,
                                    stacked_flow_observations, stacked_observations):
        infer_result: InferenceResult = self.infer_model(stacked_observations, stacked_flow_observations, keyframes,
                                                         flow_frames, flow_arcs, 'deep_features')
        encoder_result, loss_result, renders, rendered_flow_result = infer_result
        loss_result: LossResult = loss_result
        self.log_inference_results(self.best_model["value"], epoch, frame_losses, loss_result.loss,
                                   loss_result.losses, encoder_result, frame_index)

    def run_preinitializations(self, flow_arcs, flow_frames, keyframes,
                               stacked_flow_observations, stacked_observations):

        print("Pre-initializing the objects position")

        if self.config.preinitialization_method == 'levenberg-marquardt':
            self.run_levenberg_marquardt_method(stacked_observations, stacked_flow_observations,
                                                flow_frames, keyframes, flow_arcs)
        elif self.config.preinitialization_method == 'essential_matrix_decomposition':
            self.essential_matrix_preinitialization(keyframes)
        else:
            raise ValueError("Unknown pre-init method.")

        infer_result = self.infer_model(stacked_observations, stacked_flow_observations, keyframes, flow_frames,
                                        flow_arcs, 'deep_features')

        joint_loss = infer_result.loss_result.loss.mean()
        self.best_model["losses"] = infer_result.loss_result.losses_all
        self.best_model["value"] = float(joint_loss)
        self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())

    @torch.no_grad()
    def essential_matrix_preinitialization(self, keyframes):

        frame_i = max(keyframes)

        flow_arc_long_jump = (self.flow_tracks_inits[-1], frame_i)

        flow_arc_short_jump = (frame_i - 1, frame_i)

        flow_long_jump_source, flow_long_jump_target = flow_arc_long_jump

        flow_long_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_long_jump).
                                                        observed_flow).cuda()
        flow_short_jump_observations: FlowObservation = (self.data_graph.get_edge_observations(*flow_arc_short_jump).
                                                         observed_flow).cuda()

        Se3_cam_short_jump = self.epipolar_pose_estimator.estimate_pose_using_optical_flow(flow_short_jump_observations,
                                                                                           flow_arc_short_jump)

        Se3_cam_long_jump = self.epipolar_pose_estimator.estimate_pose_using_optical_flow(flow_long_jump_observations,
                                                                                          flow_arc_long_jump)

        Se3_obj_reference_frame = self.encoder.get_se3_at_frame_vectorized()[[flow_long_jump_source]]

        Se3_world_to_cam_1st_frame = self.rendering.camera_transformation_matrix_Se3()
        Se3_obj_long_jump = Se3_obj_from_epipolar_Se3_cam(Se3_cam_long_jump, Se3_world_to_cam_1st_frame)

        Se3_obj_chained_long_jump = Se3_obj_reference_frame * Se3_obj_long_jump

        # Se3_cam_chained_short_jumps = Se3.identity(batch_size=1, device='cuda')
        # for i in range(flow_long_jump_source, frame_i-1):
        #     Se3_cam_chained_short_jumps *= self.data_graph.get_edge_observations(i, i+1).predicted_cam_delta_se3
        # Se3_cam_chained_short_jumps *= Se3_cam_short_jump
        #
        # Se3_cam_chained_short_jumps_total = Se3_cam_1st_frame_to_ref_frame * Se3_cam_chained_short_jumps
        #
        # short_long_chain_ang_diff = quaternion_minimal_angular_difference(Se3_cam_chained_long_jump.quaternion,
        #                                                                   Se3_cam_chained_short_jumps_total.quaternion)
        #
        # Se3_obj_chained_short_jumps_total = Se3_obj_from_epipolar_Se3_cam(Se3_cam_chained_short_jumps_total,
        #                                                                   Se3_world_to_cam)

        # print(f'-----------------------------------Long, short chain diff: {short_long_chain_ang_diff}')
        # if short_long_chain_ang_diff > 1 and frame_i - 1 > 0:
        #     print(f'-----------------------------------Last long jump axis-angle '
        #           f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_reference_frame.quaternion.q))}')
        #     print(f'-----------------------------------Chained long jump axis-angle '
        #           f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_chained_long_jump.quaternion.q))}')
        #     print(f'-----------------------------------Chained short jumps axis-angle '
        #           f'{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_chained_short_jumps_total.quaternion.q))}')
        #
        #     # self.flow_tracks_inits[-1] = frame_i - 1
        #     prev_node_idx = frame_i - 1
        #     prev_node_observation = self.data_graph.get_camera_specific_frame_data(prev_node_idx).frame_observation
        #     prev_node_pose = self.data_graph.get_frame_data(prev_node_idx).predicted_object_se3_total.quaternion
        #     self.pose_icosphere.insert_new_reference(prev_node_observation, prev_node_pose, prev_node_idx)

        self.encoder.quaternion_offsets[flow_long_jump_target] = Se3_obj_chained_long_jump.quaternion.q

        datagraph_node = self.data_graph.get_frame_data(frame_i)
        datagraph_camera_node = self.data_graph.get_camera_specific_frame_data(frame_i)

        datagraph_short_edge = self.data_graph.get_edge_observations(*flow_arc_short_jump)
        datagraph_long_edge = self.data_graph.get_edge_observations(*flow_arc_long_jump)

        datagraph_node.predicted_object_se3_total = self.encoder.get_se3_at_frame_vectorized()[[flow_long_jump_target]]
        datagraph_short_edge.predicted_cam_delta_se3 = Se3_cam_short_jump
        datagraph_long_edge.predicted_cam_delta_se3 = Se3_cam_long_jump

        datagraph_camera_node.predicted_obj_delta_se3 = Se3_obj_long_jump
        datagraph_camera_node.predicted_cam_delta_se3 = Se3_cam_long_jump

        print(
            f"Frame {flow_long_jump_target} offset: "
            f"{torch.rad2deg(quaternion_to_axis_angle(self.encoder.quaternion_offsets[flow_long_jump_target])).numpy(force=True).round(2)}")
        print(
            f"Frame {flow_long_jump_target} qtotal: "
            f"{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_long_jump.quaternion.data)).numpy(force=True).round(2)}")
        print(
            f"Frame {flow_long_jump_target} new_of: "
            f"{torch.rad2deg(quaternion_to_axis_angle(Se3_obj_chained_long_jump.quaternion.q)).numpy(force=True).round(2)}")
    def run_levenberg_marquardt_method(self, observations: FrameObservation, flow_observations: FlowObservation,
                                       flow_frames, keyframes, flow_arcs):
        loss_coefs_names = [
            'loss_laplacian_weight', 'loss_tv_weight', 'loss_iou_weight',
            'loss_dist_weight', 'loss_q_weight',
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
                                              flow_arcs_indices, self.loss_function, observed_flows,
                                              observed_flows_segmentations,
                                              self.rendering.width, self.rendering.height, self.image_shape.width,
                                              self.image_shape.height)

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

            row_translation = trans_quats[:, :3]
            row_quaternion = trans_quats[:, 3:]
            encoder_result = encoder_result._replace(translations=row_translation, quaternions=row_quaternion)

            inference_result = infer_normalized_renderings(self.rendering, self.encoder.face_features, encoder_result,
                                                           encoder_result_flow_frames, flow_arcs_indices,
                                                           self.image_shape.width, self.image_shape.height)
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
                                                     keyframes_encoder_result=encoder_result)

            losses_all, losses, joint_loss, per_pixel_error = loss_result
            joint_loss = joint_loss.mean()
            if joint_loss < self.best_model["value"]:
                self.best_model["losses"] = losses_all
                self.best_model["value"] = joint_loss

                self.encoder.translation_offsets[keyframes] = encoder_result.translations.detach()
                self.encoder.quaternion_offsets[keyframes] = encoder_result.quaternions.detach()

                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())

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
            encoder_result, loss_result, renders, rendered_flow_result = infer_result
            loss_result: LossResult = loss_result

            joint_loss = loss_result.loss.mean()
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
                model_loss = self.log_inference_results(best_loss, epoch, frame_losses, joint_loss, loss_result.losses,
                                                        encoder_result, flow_arcs[-1][1])
                self.best_model["value"] = model_loss
                self.best_model["losses"] = loss_result.losses_all
                self.optimizer_positional_parameters.step()
                epoch += 1
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

    def log_inference_results(self, best_loss, epoch, frame_losses, joint_loss, losses, encoder_result, frame_i,
                              write_all=False):

        frame_data = self.data_graph.get_frame_data(frame_i)
        frame_data.quaternions_during_optimization.append(encoder_result.quaternions.detach().clone())
        frame_data.translations_during_optimization.append(encoder_result.translations.detach().clone())

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
                    flow_arcs, encoder_type) -> InferenceResult:

        if encoder_type == 'rgb':
            encoder = self.rgb_encoder
        elif encoder_type == 'gt_encoder':
            encoder = self.gt_encoder
        elif encoder_type == 'deep_features':
            encoder = self.encoder
        else:
            raise ValueError("Unknown encoder")

        encoder_result, encoder_result_flow_frames = encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        flow_arcs_indices = [(flow_frames.index(pair[0]), keyframes.index(pair[1])) for pair in flow_arcs]
        flow_arcs_indices_sorted = self.sort_flow_arcs_indices(flow_arcs_indices)

        inference_result = infer_normalized_renderings(self.rendering, self.encoder.face_features, encoder_result,
                                                       encoder_result_flow_frames, flow_arcs_indices_sorted,
                                                       self.image_shape.width, self.image_shape.height)

        renders, rendered_silhouettes, rendered_flow_result = inference_result

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
                                            keyframes_encoder_result=encoder_result)

        result = InferenceResult(encoder_result, loss_result, renders, rendered_flow_result)

        return result

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
            'loss_t_weight': self.config.loss_t_weight,
            'loss_rgb_weight': self.config.loss_rgb_weight,
            'loss_flow_weight': self.config.loss_flow_weight,
            'non_positional_params_lr': self.optimizer_non_positional_parameters.param_groups[0]['lr'],
            'positional_params_lr': self.optimizer_positional_parameters.param_groups[0]['lr']
        }
        dict_tensorboard_values = {**dict_tensorboard_values1, **dict_tensorboard_values2}
        self.write_results.write_into_tensorboard_log(sgd_iter, dict_tensorboard_values)

    def rgb_apply(self, keyframes, flow_frames, flow_arcs, observations: MultiCameraObservation,
                  flow_observations: MultiCameraObservation):
        start_time = time.time()

        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)

        stacked_observations = observations.stack()
        stacked_flow_observations = flow_observations.stack()

        print("Texture optimization")
        epoch = 0
        for epoch in range(self.config.rgb_iters):
            infer_result = self.infer_model(stacked_observations, stacked_flow_observations, keyframes=keyframes,
                                            flow_frames=flow_frames, flow_arcs=flow_arcs, encoder_type='rgb')

            encoder_result, loss_result, renders, rendered_flow_result = infer_result
            loss_result: LossResult = loss_result

            # if epoch % self.config.training_print_status_frequency == 0:
            #     self.log_inference_results(self.best_model["value"], epoch, frame_losses,
            #                                joint_loss, losses, encoder_result)

            joint_loss = loss_result.loss.mean()
            self.rgb_optimizer.zero_grad()
            joint_loss.backward(retain_graph=True)
            self.rgb_optimizer.step()

        print(f'Elapsed time in seconds: {time.time() - start_time} for total of {epoch + 1} epochs.')

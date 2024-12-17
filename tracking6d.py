import time
from pathlib import Path
from typing import Optional, List, Callable, Union

import kaolin
import torch
from kornia.geometry import Quaternion, Se3, PinholeCamera

from data_providers.flow_wrappers import (FlowProvider, MFTFlowProvider,
                                          RoMaFlowProvider)
from auxiliary_scripts.image_utils import get_shape, ImageShape
from auxiliary_scripts.logging import WriteResults
from auxiliary_scripts.math_utils import Se3_epipolar_cam_from_Se3_obj
from data_providers.flow_provider import RoMaFlowProviderDirect
from data_providers.frame_provider import PrecomputedTracker, BaseTracker, SyntheticDataGeneratingTracker
from data_structures.data_graph import DataGraph
from data_structures.keyframe_buffer import FlowObservation
from data_structures.pose_icosphere import PoseIcosphere
from flow import flow_image_coords_to_unit_coords, normalize_rendered_flows
from models.encoder import Encoder
from models.initial_mesh import generate_face_features
from models.rendering import RenderingKaolin
from pose.frame_filter import FrameFilter
from pose.glomap import GlomapWrapper
from tracker_config import TrackerConfig
from utils import normalize_vertices, homogenize_3x3_camera_intrinsics


class Tracking6D:

    def __init__(self, config: TrackerConfig, write_folder, gt_texture=None, gt_mesh=None, gt_rotations=None,
                 gt_translations=None, images_paths: List[Path] = None, segmentation_paths: List[Path] = None):
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

        # Features
        self.feat: Optional[Callable] = None
        self.feat_rgb = None

        # Feature extraction
        self.feature_extractor = None

        # Optical flow
        self.short_flow_model: Optional[Union[FlowProvider, MFTFlowProvider]] = None
        self.long_flow_provider: Optional[MFTFlowProvider] = None
        self.roma_flow_provider: Optional[RoMaFlowProvider] = None

        # Ground truth related
        assert torch.all(gt_rotations.eq(0)) or config.rot_init is None  # Conflicting setting handling
        assert torch.all(gt_translations.eq(0)) or config.tran_init is None  # Conflicting setting handling

        config.rot_init = tuple(gt_rotations[0].numpy(force=True))
        config.tran_init = tuple(gt_translations[0].numpy(force=True))

        self.gt_rotations: Optional[torch.Tensor] = gt_rotations
        self.gt_translations: Optional[torch.Tensor] = gt_translations

        # Data graph
        self.data_graph: Optional[DataGraph] = None

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Flow tracks
        self.flow_tracks_inits = [0]
        self.pose_icosphere: Optional[PoseIcosphere] = PoseIcosphere()

        # Tracker
        self.tracker: Optional[BaseTracker] = None

        # Other utilities and flags
        self.results_writer = None
        self.logged_sgd_translations = []
        self.logged_sgd_quaternions = []

        self.image_shape: Optional[ImageShape] = None
        self.write_folder = Path(write_folder)
        self.config = config
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

        torch.backends.cudnn.benchmark = True
        self.initialize_renderer()
        self.initialize_encoders(iface_features, ivertices)

        if self.config.gt_flow_source != 'GenerateSynthetic':  # This provides a significant speed-up for debugging
            self.initialize_flow_model()

        self.data_graph = DataGraph()

        if self.config.generate_synthetic_observations_if_possible:
            assert self.gt_translations is not None and self.gt_rotations is not None

            self.tracker = SyntheticDataGeneratingTracker(self.config, self.rendering, self.gt_encoder, self.gt_texture,
                                                          self.feat)

        else:
            assert not self.config.matching_target_to_backview

            if self.config.segmentation_tracker == 'precomputed':
                self.tracker = PrecomputedTracker(self.config, self.feat, images_paths, segmentation_paths)
            else:
                raise ValueError('Unknown value of "segmentation_tracker"')

        if self.config.camera_intrinsics is None:
            camera_intrinsics = homogenize_3x3_camera_intrinsics(self.rendering.camera_intrinsics)[None]
        else:
            camera_intrinsics = homogenize_3x3_camera_intrinsics(torch.from_numpy(self.config.camera_intrinsics).cuda())[None]
            assert camera_intrinsics.shape == homogenize_3x3_camera_intrinsics(self.rendering.camera_intrinsics)[None].shape
        if self.config.camera_extrinsics is None:
            camera_extrinsics = self.rendering.camera_transformation_matrix_Se3().matrix()
        else:
            camera_extrinsics = torch.from_numpy(self.config.camera_extrinsics).cuda()[None]

        orig_image_width = torch.Tensor([self.image_shape.width / self.config.image_downsample]).cuda()
        orig_image_height = torch.Tensor([self.image_shape.height / self.config.image_downsample]).cuda()
        self.pinhole_params = PinholeCamera(camera_intrinsics, camera_extrinsics,
                                            orig_image_width, orig_image_height)
        self.pinhole_params.scale_(self.config.image_downsample)

        self.glomap_wrapper = GlomapWrapper(self.write_folder, self.config, self.data_graph, self.image_shape,
                                            self.pose_icosphere)

        self.results_writer = WriteResults(write_folder=self.write_folder, shape=self.image_shape,
                                           tracking_config=self.config, rendering=self.rendering,
                                           gt_encoder=self.gt_encoder, deep_encoder=self.encoder,
                                           data_graph=self.data_graph,
                                           pinhole_params=self.pinhole_params, pose_icosphere=self.pose_icosphere)

        self.flow_provider = RoMaFlowProviderDirect(self.data_graph, self.config.device)

        self.frame_filter = FrameFilter(self.config, self.data_graph, self.pose_icosphere, self.image_shape,
                                        self.flow_provider)

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

    def initialize_feature_extractor(self):
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

    def initialize_flow_model(self):

        self.roma_flow_provider = RoMaFlowProvider(self.config.MFT_backbone_cfg, config=self.config)
        self.short_flow_model = self.roma_flow_provider

    def run_tracking(self):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1

        frame_i = 0

        T_world_to_cam = self.rendering.camera_transformation_matrix_4x4()
        Se3_world_to_cam = Se3.from_matrix(T_world_to_cam)

        self.init_datagraph_frame(Se3_world_to_cam, frame_i)

        frame = self.data_graph.get_frame_data(frame_i)
        frame.predicted_object_se3_long_jump = frame.gt_pose_cam

        new_frame_observation = self.tracker.next(frame_i)
        frame.frame_observation = new_frame_observation.send_to_device('cpu')

        initial_pose = self.encoder.get_se3_at_frame_vectorized()[[frame_i]]

        self.pose_icosphere.insert_new_reference(new_frame_observation, initial_pose, frame_i)

        for frame_i in range(1, self.config.input_frames):

            self.init_datagraph_frame(Se3_world_to_cam, frame_i)

            new_frame_observation = self.tracker.next(frame_i)
            self.data_graph.get_frame_data(frame_i).frame_observation = new_frame_observation.send_to_device('cpu')

            start = time.time()

            # self.add_new_flows(frame_i)

            self.frame_filter.filter_frames(frame_i)

            print('Elapsed time in seconds: ', time.time() - start, "Frame ", frame_i, "out of",
                  self.config.input_frames)

            if self.config.write_results:
                self.results_writer.write_results(frame_i=frame_i)

            if self.long_flow_provider is not None and 'direct' in self.config.MFT_backbone_cfg:
                self.long_flow_provider.need_to_init = True

        pose_icosphere_node_idxs = [p.keyframe_idx_observed for p in self.pose_icosphere.reference_poses]
        images_paths = []
        segmentation_paths = []
        matching_pairs = []
        for node_idx in pose_icosphere_node_idxs:
            self.glomap_wrapper.dump_frame_node_for_glomap(node_idx)
            frame_data = self.data_graph.get_frame_data(node_idx)

            images_paths.append(frame_data.image_save_path)
            segmentation_paths.append(frame_data.segmentation_save_path)

        for frame1_idx, frame2_idx in self.data_graph.G.edges:
            arc_data = self.data_graph.get_edge_observations(frame1_idx, frame2_idx)
            if (arc_data.is_match_reliable and frame1_idx in pose_icosphere_node_idxs and frame2_idx in pose_icosphere_node_idxs
                    and frame1_idx != frame2_idx and (frame1_idx, frame2_idx) not in matching_pairs):
                u_index = pose_icosphere_node_idxs.index(frame1_idx)
                v_index = pose_icosphere_node_idxs.index(frame2_idx)
                matching_pairs.append((u_index, v_index))

        time.sleep(1)
        reconstruction = self.glomap_wrapper.run_glomap_from_image_list(images_paths, segmentation_paths,
                                                                        matching_pairs)

        self.glomap_wrapper.normalize_reconstruction(reconstruction)
        self.results_writer.visualize_colmap_track(frame_i, reconstruction)
        return

    def init_datagraph_frame(self, Se3_world_to_cam, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)
        frame_node.gt_rot_axis_angle = self.gt_rotations[frame_i]
        frame_node.gt_translation = self.gt_translations[frame_i]

        gt_Se3_obj = Se3(Quaternion.from_axis_angle(self.gt_rotations[[frame_i]]), self.gt_translations[[frame_i]])
        gt_Se3_cam = Se3_epipolar_cam_from_Se3_obj(gt_Se3_obj, Se3_world_to_cam)
        frame_node.gt_pose_cam = gt_Se3_cam

        frame_node.gt_pinhole_params = self.pinhole_params

    @torch.no_grad()
    def add_new_flows(self, frame_i):

        def process_flow_arc(flow_source_frame, flow_target_frame):
            observed_flow, occlusions, uncertainties = self.next_gt_flow(flow_source_frame, flow_target_frame)

            # Determine the appropriate renderer and keyframes based on the backview flag
            renderer = self.rendering

            # Render the flow
            synthetic_flow = renderer.render_flow_for_frame(self.gt_encoder, flow_source_frame, flow_target_frame)

            # Convert synthetic flow results to CPU
            synthetic_flow_cpu = synthetic_flow.send_to_device('cpu')

            # Get the observed segmentation for the flow source frame
            frame_observation = (self.data_graph.get_frame_data(flow_source_frame).
                                 frame_observation.send_to_device('cuda'))
            segment = frame_observation.observed_segmentation

            flow_observation = FlowObservation(observed_flow=observed_flow,
                                               observed_flow_segmentation=segment,
                                               observed_flow_uncertainty=uncertainties,
                                               observed_flow_occlusion=occlusions,
                                               flow_source_frames=[flow_source_frame],
                                               flow_target_frames=[flow_target_frame])

            # Update the edge data with synthetic flow results
            edge_data = self.data_graph.get_edge_observations(flow_source_frame, flow_target_frame)
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
            process_flow_arc(flow_source_frame, flow_target_frame)

    def next_gt_flow(self, flow_source_frame, flow_target_frame):

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

            source_frame_obs = self.data_graph.get_frame_data(flow_source_frame).frame_observation
            target_frame_obs = self.data_graph.get_frame_data(flow_target_frame).frame_observation

            target_image = target_frame_obs.observed_image.float() * 255
            template_image = source_frame_obs.observed_image.float() * 255

            if isinstance(self.short_flow_model, MFTFlowProvider):
                self.short_flow_model.init(template_image)
                self.short_flow_model.need_to_init = False
            observed_flow, occlusion, uncertainty = self.short_flow_model.next_flow(template_image,
                                                                                    target_image)

            observed_flow = flow_image_coords_to_unit_coords(observed_flow)

        else:
            raise ValueError("'gt_flow_source' must be either 'GenerateSynthetic' or 'FlowNetwork'")

        return observed_flow, occlusion, uncertainty

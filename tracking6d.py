import time
from pathlib import Path
from typing import Optional, List, Callable

import kaolin
import torch
from kornia.geometry import Quaternion, Se3, PinholeCamera
from kornia.image import ImageSize

from data_providers.flow_wrappers import RoMaFlowProvider
from auxiliary_scripts.image_utils import get_shape
from auxiliary_scripts.logging import WriteResults
from auxiliary_scripts.math_utils import Se3_epipolar_cam_from_Se3_obj
from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect
from data_providers.frame_provider import PrecomputedTracker, BaseTracker, SyntheticDataGeneratingTracker
from data_structures.data_graph import DataGraph
from data_structures.pose_icosphere import PoseIcosphere
from models.encoder import Encoder
from models.initial_mesh import generate_face_features
from pose.frame_filter import FrameFilter
from pose.glomap import GlomapWrapper
from tracker_config import TrackerConfig
from utils import normalize_vertices


class Tracking6D:

    def __init__(self, config: TrackerConfig, write_folder, gt_texture=None, gt_mesh=None, gt_rotations=None,
                 gt_translations=None, images_paths: List[Path] = None, segmentation_paths: List[Path] = None,
                 gt_Se3_world_to_cam: Se3 = None):
        # Encoders and related components
        self.encoder: Optional[Encoder] = None
        self.gt_encoder: Optional[Encoder] = None

        # Rendering and mesh related
        self.gt_mesh_prototype: Optional[kaolin.rep.SurfaceMesh] = gt_mesh
        self.gt_texture = gt_texture
        self.gt_texture_features = None

        # Paths
        self.images_paths: Optional[List[Path]] = images_paths
        self.segmentation_paths: Optional[List[Path]] = segmentation_paths

        # Features
        self.feat: Optional[Callable] = None
        self.feat_rgb = None

        # Feature extraction
        self.feature_extractor = None

        # Optical flow
        self.roma_flow_provider: Optional[RoMaFlowProvider] = None

        # Ground truth related
        assert torch.all(gt_rotations.eq(0)) or config.rot_init is None  # Conflicting setting handling
        assert torch.all(gt_translations.eq(0)) or config.tran_init is None  # Conflicting setting handling

        config.rot_init = tuple(gt_rotations[0].numpy(force=True))
        config.tran_init = tuple(gt_translations[0].numpy(force=True))

        self.gt_rotations: Optional[torch.Tensor] = gt_rotations
        self.gt_translations: Optional[torch.Tensor] = gt_translations
        self.gt_Se3_world_to_cam: Optional[Se3] = gt_Se3_world_to_cam
        self.world_to_cam: Optional[Se3] = None
        if self.gt_Se3_world_to_cam is None:
            self.world_to_cam: Optional[Se3] = Se3(Quaternion.identity(1),
                                                   torch.tensor([[0., 0., 1.]])).to(config.device)
        else:
            self.world_to_cam = self.gt_Se3_world_to_cam

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

        self.image_shape: Optional[ImageSize] = None
        self.write_folder = Path(write_folder)
        self.config = config
        self.device = 'cuda'

        iface_features, ivertices = self.initialize_mesh()
        self.initialize_feature_extractor()
        if self.gt_texture is not None:
            self.gt_texture_features = self.feat(self.gt_texture[None])[0].detach()

        if self.config.generate_synthetic_observations_if_possible:
            self.image_shape = ImageSize(width=int(self.config.image_downsample * self.config.max_width),
                                         height=int(self.config.image_downsample * self.config.max_width))
        else:
            self.image_shape = get_shape(images_paths[0], self.config.image_downsample)

        self.initialize_encoders(iface_features, ivertices)

        self.data_graph = DataGraph()

        if self.config.generate_synthetic_observations_if_possible:
            assert self.gt_translations is not None and self.gt_rotations is not None
            assert gt_mesh is not None
            assert self.gt_texture is not None

            self.tracker = SyntheticDataGeneratingTracker(self.config, self.gt_encoder, self.gt_texture, self.feat,
                                                          gt_mesh)

        else:
            if self.config.segmentation_tracker == 'precomputed':
                self.tracker = PrecomputedTracker(self.config, self.feat, images_paths, segmentation_paths)
            else:
                raise ValueError('Unknown value of "segmentation_tracker"')

        self.results_writer = WriteResults(write_folder=self.write_folder, shape=self.image_shape,
                                           tracking_config=self.config, data_graph=self.data_graph,
                                           pose_icosphere=self.pose_icosphere, images_paths=self.images_paths,
                                           segmentation_paths=self.segmentation_paths,
                                           Se3_world_to_cam=self.world_to_cam)

        self.cache_folder: Path = Path('/mnt/personal/jelint19/cache/flow_cache') / config.dataset / config.sequence
                                   #f'{config.experiment_name}_{self.image_shape.width}x{self.image_shape.height}px')
        self.flow_provider = PrecomputedRoMaFlowProviderDirect(self.data_graph, self.config.device, self.cache_folder,
                                                               images_paths)

        self.frame_filter = FrameFilter(self.config, self.data_graph, self.pose_icosphere, self.image_shape,
                                        self.flow_provider)

        self.glomap_wrapper = GlomapWrapper(self.write_folder, self.config, self.data_graph, self.image_shape,
                                            self.pose_icosphere, self.flow_provider)

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

    def run_tracking(self):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1

        frame_i = 0

        for frame_i in range(0, self.config.input_frames):

            self.init_datagraph_frame(frame_i)

            new_frame_observation = self.tracker.next(frame_i)
            self.data_graph.get_frame_data(frame_i).frame_observation = new_frame_observation.send_to_device('cpu')

            start = time.time()

            if frame_i == 0:
                initial_pose = self.encoder.get_se3_at_frame_vectorized()[[frame_i]]

                self.pose_icosphere.insert_new_reference(new_frame_observation, initial_pose, frame_i)

            else:
                if self.config.matcher == 'RoMa':
                    self.frame_filter.filter_frames(frame_i)

                self.results_writer.write_results(frame_i=frame_i)

            print('Elapsed time in seconds: ', time.time() - start, "Frame ", frame_i, "out of",
                  self.config.input_frames)

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

        assert len(pose_icosphere_node_idxs) > 2

        time.sleep(1)
        print(matching_pairs)
        if self.config.matcher == 'RoMa':
            reconstruction = self.glomap_wrapper.run_glomap_from_image_list(images_paths, segmentation_paths,
                                                                            matching_pairs)
        elif self.config.matcher == 'SIFT':
            reconstruction = self.glomap_wrapper.run_glomap_from_image_list_sift(images_paths, segmentation_paths,
                                                                                 matching_pairs)
        else:
            raise ValueError(f'Unknown matcher {self.config.matcher}')

        reconstruction = self.glomap_wrapper.normalize_reconstruction(reconstruction)
        self.write_gt_poses()
        self.results_writer.visualize_colmap_track(frame_i, reconstruction)

        self.evaluate_reconstruction(reconstruction)

        return

    def evaluate_reconstruction(self, reconstruction, csv_output_path: Optional[Path] = None):
        """
        Evaluate the reconstruction and save statistics to a CSV file.

        Parameters:
            reconstruction: The reconstructed data.
            csv_output_path: Path to the output CSV file.
        """
        import pandas as pd
        import os

        stats = []

        if csv_output_path is None:
            csv_output_path = self.write_folder.parent.parent / 'stats.csv'

        images_paths_to_frame_index = {str(self.images_paths[i].name): i for i in range(len(self.images_paths))}

        for image in reconstruction.images.values():
            image_frame_id = images_paths_to_frame_index[image.name]

            t_cam_pred = torch.tensor(image.cam_from_world.translation)
            q_cam_xyzw_pred = torch.tensor(image.cam_from_world.rotation.quat)
            q_cam_wxyz_pred = q_cam_xyzw_pred[[3, 0, 1, 2]]
            Se3_cam_pred = Se3(Quaternion(q_cam_wxyz_pred), t_cam_pred)

            frame_data = self.data_graph.get_frame_data(image_frame_id)
            Se3_cam_gt = frame_data.gt_pose_cam

            # Ground-truth rotation and translation
            gt_rotation = Se3_cam_gt.rotation.matrix().tolist()
            gt_translation = Se3_cam_gt.translation.tolist()

            pred_rotation = Se3_cam_pred.rotation.matrix().tolist()
            pred_translation = Se3_cam_pred.translation.tolist()

            # Add stats for the current image frame
            stats.append({
                'dataset': self.config.dataset,
                'sequence': self.config.sequence,
                'image_frame_id': image_frame_id,
                'gt_rotation': gt_rotation,
                'gt_translation': gt_translation,
                'pred_rotation': pred_rotation,
                'pred_translation': pred_translation
            })

        # Convert stats to a Pandas DataFrame
        stats_df = pd.DataFrame(stats)

        # If the CSV file exists, append; otherwise, create a new one
        if os.path.exists(csv_output_path):
            existing_df = pd.read_csv(csv_output_path)
            filtered_df = existing_df[~((existing_df['dataset'] == self.config.dataset) &
                                        (existing_df['sequence'] == self.config.sequence))]
            updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
            updated_df.to_csv(csv_output_path, index=False)
        else:
            stats_df.to_csv(csv_output_path, index=False)

    def write_gt_poses(self):
        import pandas as pd

        stats = []

        csv_output_path = self.write_folder.parent.parent / 'gt_poses.csv'

        for node_idx in sorted(self.data_graph.G.nodes):

            node_data = self.data_graph.get_frame_data(node_idx)

            Se3_cam_gt = node_data.gt_pose_cam

            # Ground-truth rotation and translation
            gt_rotation = Se3_cam_gt.rotation.matrix().tolist()
            gt_translation = Se3_cam_gt.translation.tolist()

            image_name = str(node_data.image_filename)
            # Add stats for the current image frame
            stats.append({
                'dataset': self.config.dataset,
                'sequence': self.config.sequence,
                'frame_name': image_name,
                'gt_R_w2c': gt_rotation,
                'gt_t_Rw2c': gt_translation,
                'gt_cam_K': node_data.gt_pinhole_K.camera_matrix.numpy(force=True).tolist(),
            })

        # Convert stats to a Pandas DataFrame
        stats_df = pd.DataFrame(stats)

        stats_df.to_csv(csv_output_path, index=False)

    def init_datagraph_frame(self, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)
        frame_node.gt_rot_axis_angle = self.gt_rotations[frame_i]
        frame_node.gt_translation = self.gt_translations[frame_i]

        gt_Se3_obj = Se3(Quaternion.from_axis_angle(self.gt_rotations[[frame_i]]), self.gt_translations[[frame_i]])

        if self.gt_Se3_world_to_cam is not None:
            Se3_world_to_cam = self.gt_Se3_world_to_cam
        else:
            Se3_world_to_cam = Se3(Quaternion.identity(1), torch.tensor([[0., 0., 1.]])).to(self.device)
        gt_Se3_cam = Se3_epipolar_cam_from_Se3_obj(gt_Se3_obj, Se3_world_to_cam)
        frame_node.gt_pose_cam = gt_Se3_cam

        # camera_intrinsics = homogenize_3x3_camera_intrinsics(self.tracker.get_intrinsics_for_frame(frame_i)[None])
        # orig_image_width = torch.Tensor([self.image_shape.width / self.config.image_downsample]).cuda()
        # orig_image_height = torch.Tensor([self.image_shape.height / self.config.image_downsample]).cuda()
        # pinhole_params = PinholeCamera(camera_intrinsics, camera_extrinsics,
        #                                     orig_image_width, orig_image_height)
        # pinhole_params.scale_(self.config.image_downsample)

        camera_intrinsics = self.tracker.get_intrinsics_for_frame(frame_i)
        frame_node.gt_pinhole_K = camera_intrinsics

        if self.images_paths is not None:
            frame_node.image_filename = self.images_paths[frame_i].name

        if self.segmentation_paths is not None:
            frame_node.segmentation_filename = self.segmentation_paths[frame_i].name

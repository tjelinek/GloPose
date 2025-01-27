import os
import shutil
import time
from pathlib import Path
from typing import Optional, List

import torch
from kornia.geometry import Quaternion, Se3, PinholeCamera
from kornia.image import ImageSize

from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect
from data_providers.frame_provider import BaseTracker
from data_providers.matching_provider_sift import PrecomputedSIFTMatchingProvider
from data_structures.data_graph import DataGraph
from pose.frame_filter import RoMaFrameFilter, FrameFilterSift
from pose.glomap import GlomapWrapper
from tracker_config import TrackerConfig
from utils.logging import WriteResults
from utils.math_utils import Se3_epipolar_cam_from_Se3_obj


class Tracker6D:

    def __init__(self, config: TrackerConfig, write_folder, gt_texture=None, gt_mesh=None,
                 gt_obj_1_to_obj_i_Se3: Optional[Se3] = None, gt_Se3_obj_1_to_cam: Se3 = None,
                 images_paths: List[Path] = None, video_path: Optional[Path] = None,
                 segmentation_video_path: Optional[Path] = None, segmentation_paths: List[Path] = None,
                 initial_image: torch.Tensor = None, initial_segmentation: torch.Tensor = None,):

        config.write_folder = write_folder
        # Paths
        self.images_paths: Optional[List[Path]] = images_paths
        self.segmentation_paths: Optional[List[Path]] = segmentation_paths
        self.video_path: Optional[Path] = video_path
        self.segmentation_video_path: Optional[Path] = segmentation_video_path

        # Ground truth related
        # assert gt_obj_1_to_obj_i_Se3 is None or config.rot_init is None  # Conflicting setting handling

        if gt_obj_1_to_obj_i_Se3 is not None:
            config.rot_init = tuple(gt_obj_1_to_obj_i_Se3.quaternion.to_axis_angle()[0].numpy(force=True))
            config.tran_init = tuple(gt_obj_1_to_obj_i_Se3.translation[0].numpy(force=True))

        self.gt_cam_to_obj_Se3: Optional[Se3]
        self.gt_obj_1_to_obj_i_Se3: Optional[Se3] = gt_obj_1_to_obj_i_Se3

        self.Se3_obj_to_cam: Se3 = gt_Se3_obj_1_to_cam
        if self.Se3_obj_to_cam is None:
            self.Se3_obj_to_cam: Se3 = Se3(Quaternion.identity(1), torch.tensor([[0., 0., 1.]])).to(config.device)

        # Data graph
        self.data_graph: Optional[DataGraph] = None

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Tracker
        self.tracker: Optional[BaseTracker] = None

        # Other utilities and flags
        self.results_writer = None

        self.image_shape: Optional[ImageSize] = None
        self.write_folder = Path(write_folder)
        self.config = config
        self.device = 'cuda'

        self.data_graph = DataGraph()

        cache_folder_RoMA: Path = (Path('/mnt/personal/jelint19/cache/RoMa_cache') /
                                   config.roma_matcher_config.config_name / config.dataset / config.sequence)
        cache_folder_SIFT: Path = (Path('/mnt/personal/jelint19/cache/SIFT_cache') /
                                   config.sift_matcher_config.config_name / config.dataset / config.sequence)
        cache_folder_SAM2: Path = (Path('/mnt/personal/jelint19/cache/SAM_cache2') /
                                   config.sift_matcher_config.config_name / config.dataset / config.sequence)

        self.tracker = BaseTracker(self.config, gt_mesh=gt_mesh, gt_texture=gt_texture,
                                   gt_rotations=gt_obj_1_to_obj_i_Se3.quaternion.to_axis_angle(),
                                   gt_translations=gt_obj_1_to_obj_i_Se3.translation,
                                   initial_segmentation=initial_segmentation,
                                   initial_image=initial_image, images_paths=images_paths, video_path=video_path,
                                   segmentation_paths=segmentation_paths,
                                   segmentation_video_path=segmentation_video_path, sam2_cache_folder=cache_folder_SAM2)
        self.image_shape = self.tracker.get_image_size()

        self.results_writer = WriteResults(write_folder=self.write_folder, shape=self.image_shape,
                                           tracking_config=self.config, data_graph=self.data_graph,
                                           Se3_world_to_cam=self.Se3_obj_to_cam)

        self.flow_provider = PrecomputedRoMaFlowProviderDirect(self.data_graph, self.config.device, cache_folder_RoMA,
                                                               self.image_shape)

        if self.config.frame_filter == 'RoMa':
            self.frame_filter = RoMaFrameFilter(self.config, self.data_graph, self.image_shape,
                                                self.flow_provider)
        elif self.config.frame_filter == 'SIFT':
            sift_matcher = PrecomputedSIFTMatchingProvider(self.data_graph,
                                                           self.config.sift_matcher_config.sift_filter_num_feats,
                                                           cache_folder_SIFT, device=self.config.device)
            self.frame_filter = FrameFilterSift(self.config, self.data_graph, self.image_shape,
                                                sift_matcher)

        self.glomap_wrapper = GlomapWrapper(self.write_folder, self.config, self.data_graph, self.image_shape,
                                            self.flow_provider)

    def run_filtering_with_reconstruction(self):

        self.filter_frames()

        keyframe_graph = self.frame_filter.get_keyframe_graph()

        keyframe_nodes_idxs = list(sorted(keyframe_graph.nodes()))

        images_paths = []
        segmentation_paths = []
        matching_pairs = []
        for node_idx in keyframe_nodes_idxs:
            self.glomap_wrapper.dump_frame_node_for_glomap(node_idx)
            frame_data = self.data_graph.get_frame_data(node_idx)

            images_paths.append(frame_data.image_save_path)
            segmentation_paths.append(frame_data.segmentation_save_path)

        for frame1_idx, frame2_idx in keyframe_graph.edges:
            u_index = keyframe_nodes_idxs.index(frame1_idx)
            v_index = keyframe_nodes_idxs.index(frame2_idx)
            matching_pairs.append((u_index, v_index))

        assert len(keyframe_nodes_idxs) > 2
        print(keyframe_graph.edges)
        reconstruction = self.run_reconstruction(images_paths, segmentation_paths, matching_pairs)

        self.write_gt_poses()
        self.results_writer.visualize_colmap_track(self.config.input_frames - 1, reconstruction)

        self.evaluate_reconstruction(reconstruction)

        return

    def filter_frames(self):
        for frame_i in range(0, self.config.input_frames):
            self.init_datagraph_frame(frame_i)

            new_frame_observation = self.tracker.next(frame_i)
            self.data_graph.get_frame_data(frame_i).frame_observation = new_frame_observation.send_to_device('cpu')

            start = time.time()

            self.frame_filter.filter_frames(frame_i)

            self.results_writer.write_results(frame_i=frame_i, keyframe_graph=self.frame_filter.keyframe_graph)

            print('Elapsed time in seconds: ', time.time() - start, "Frame ", frame_i, "out of",
                  self.config.input_frames)

    def run_reconstruction(self, images_paths, segmentation_paths, matching_pairs):
        if self.config.reconstruction_matches == 'RoMa':
            reconstruction = self.glomap_wrapper.run_glomap_from_image_list(images_paths, segmentation_paths,
                                                                            matching_pairs)
        elif self.config.reconstruction_matches == 'SIFT':
            reconstruction = self.glomap_wrapper.run_glomap_from_image_list_sift(images_paths, segmentation_paths,
                                                                                 matching_pairs)
        else:
            raise ValueError(f'Unknown matcher {self.config.frame_filter}')
        reconstruction = self.glomap_wrapper.normalize_reconstruction(reconstruction)

        return reconstruction

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

        images_paths_to_frame_index = {str(self.data_graph.get_frame_data(i).image_filename.name): i
                                       for i in range(self.config.input_frames)}

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

            gt_pinhole_K = node_data.gt_pinhole_K
            stats.append({
                'dataset': self.config.dataset,
                'sequence': self.config.sequence,
                'frame_name': image_name,
                'gt_R_w2c': gt_rotation,
                'gt_t_Rw2c': gt_translation,
                'gt_cam_K': gt_pinhole_K.numpy(force=True).tolist() if gt_pinhole_K is not None else None,
            })

        # Convert stats to a Pandas DataFrame
        stats_df = pd.DataFrame(stats)

        stats_df.to_csv(csv_output_path, index=False)

    def init_datagraph_frame(self, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)

        gt_Se3_cam = Se3_epipolar_cam_from_Se3_obj(self.gt_obj_1_to_obj_i_Se3[[frame_i]], self.Se3_obj_to_cam)
        frame_node.gt_pose_cam = gt_Se3_cam
        frame_node.gt_obj1_to_obji = self.gt_obj_1_to_obj_i_Se3[[frame_i]]

        if self.images_paths is not None:
            frame_node.image_filename = Path(self.images_paths[frame_i].name)
        elif self.video_path is not None:
            frame_node.image_filename = Path(f'{self.video_path.stem}_{frame_i}.png')

        if self.segmentation_paths is not None:
            frame_node.segmentation_filename = Path(self.segmentation_paths[frame_i].name)
        elif self.segmentation_video_path is not None:
            frame_node.image_filename = Path(f'{self.segmentation_video_path.stem}_{frame_i}.png')


def run_tracking_on_sequence(config: TrackerConfig, write_folder: Path, **kwargs):
    if os.path.exists(write_folder):
        shutil.rmtree(write_folder)

    write_folder.mkdir(exist_ok=True, parents=True)

    print('\n\n\n---------------------------------------------------')
    print("Running tracking on dataset:", config.dataset)
    print("Sequence:", config.sequence)
    print('---------------------------------------------------\n\n')

    t0 = time.time()

    sfb = Tracker6D(config, write_folder, **kwargs)
    sfb.run_filtering_with_reconstruction()

    print(f'{config.input_frames} epochs took {(time.time() - t0) / 1} seconds.')

import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import pandas as pd
import torch
from kornia.geometry import Se3, PinholeCamera
from pycolmap import Reconstruction

from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect
from data_providers.frame_provider import BaseTracker
from data_providers.matching_provider_sift import PrecomputedSIFTMatchingProvider
from data_structures.data_graph import DataGraph
from data_structures.view_graph import view_graph_from_datagraph
from pose.frame_filter import RoMaFrameFilter, FrameFilterSift
from pose.glomap import GlomapWrapper, get_image_Se3_world2cam
from tracker_config import TrackerConfig
from utils.eval import update_global_statistics
from utils.results_logging import WriteResults
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i


class Tracker6D:

    def __init__(self, config: TrackerConfig, write_folder, gt_texture=None, gt_mesh=None,
                 images_paths: List[Path] = None, video_path: Optional[Path] = None,
                 gt_Se3_cam2obj: Optional[Se3 | Dict[int, Se3]] = None, initial_gt_Se3_cam2obj: Optional[Se3] = None,
                 segmentation_video_path: Optional[Path] = None, segmentation_paths: List[Path] = None,
                 initial_image: torch.Tensor | List[torch.Tensor] = None,
                 initial_segmentation: torch.Tensor | List[torch.Tensor] = None, sequence_starts: List[int] = None):

        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)

        write_folder.mkdir(exist_ok=True, parents=True)

        config.write_folder = write_folder
        # Paths
        self.images_paths: Optional[List[Path]] = images_paths
        self.segmentation_paths: Optional[List[Path]] = segmentation_paths
        self.video_path: Optional[Path] = video_path
        self.segmentation_video_path: Optional[Path] = segmentation_video_path
        self.sequence_starts: Optional[List[int]] = sequence_starts

        self.gt_Se3_cam2obj: Optional[Dict[int, Se3]] = None
        # Ground truth related
        if type(gt_Se3_cam2obj) is dict:
            self.gt_Se3_cam2obj: Optional[Dict[Se3]] = gt_Se3_cam2obj
        elif type(gt_Se3_cam2obj) is Se3:
            self.gt_Se3_cam2obj: Optional[Dict[Se3]] = {i: gt_Se3_cam2obj[i] for i in range(gt_Se3_cam2obj.t.shape[0])}

        self.initial_gt_Se3_cam2obj: Optional[Se3] = initial_gt_Se3_cam2obj
        if initial_gt_Se3_cam2obj is None and self.gt_Se3_cam2obj is not None and self.gt_Se3_cam2obj.get(0):
            self.initial_gt_Se3_cam2obj = self.gt_Se3_cam2obj.get(0)

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Frame provider
        self.tracker: Optional[BaseTracker] = None

        # Other utilities and flags
        self.results_writer = None

        self.write_folder = Path(write_folder)
        self.config = config

        self.data_graph: DataGraph = DataGraph(out_device=self.config.device)

        cache_folder_RoMA: Path = (Path('/mnt/personal/jelint19/cache/RoMa_cache') /
                                   config.roma_matcher_config.config_name / config.dataset / config.sequence)
        cache_folder_SIFT: Path = (Path('/mnt/personal/jelint19/cache/SIFT_cache') /
                                   config.sift_matcher_config.config_name / config.dataset /
                                   f'{config.sequence}_{config.special_hash}')

        self.cache_folder_view_graph: Path = (Path('/mnt/personal/jelint19/cache/view_graph_cache') /
                                              config.dataset / f'{config.sequence}_{config.special_hash}')

        self.initialize_frame_provider(gt_mesh, gt_texture, images_paths, initial_image, initial_segmentation,
                                       segmentation_paths, segmentation_video_path, video_path, 0)

        self.results_writer = WriteResults(write_folder=self.write_folder, tracking_config=self.config,
                                           data_graph=self.data_graph)

        self.flow_provider = PrecomputedRoMaFlowProviderDirect(self.config.device, cache_folder_RoMA, self.data_graph,
                                                               purge_cache=self.config.purge_cache)

        self.frame_filter: Union[RoMaFrameFilter, FrameFilterSift]
        if self.config.frame_filter == 'RoMa':
            self.frame_filter = RoMaFrameFilter(self.config, self.data_graph, self.flow_provider)
        elif self.config.frame_filter == 'SIFT':
            sift_matcher = PrecomputedSIFTMatchingProvider(self.data_graph,
                                                           self.config.sift_matcher_config.sift_filter_num_feats,
                                                           cache_folder_SIFT, device=self.config.device)
            self.frame_filter = FrameFilterSift(self.config, self.data_graph, sift_matcher)
        else:
            raise ValueError(f'Unknown frame_filter {self.config.frame_filter}')

        self.glomap_wrapper = GlomapWrapper(self.write_folder, self.config, self.data_graph, self.flow_provider)

    def initialize_frame_provider(self, gt_mesh: torch.Tensor, gt_texture: torch.Tensor, images_paths: List[Path],
                                  initial_image: torch.Tensor | List[torch.Tensor],
                                  initial_segmentation: torch.Tensor | List[torch.Tensor],
                                  segmentation_paths: List[Path], segmentation_video_path: Path, video_path: Path,
                                  frame_i: int):

        cache_folder_SAM2: Path = ((Path('/mnt/personal/jelint19/cache/SAM_cache') / self.config.dataset /
                                   f'{self.config.sequence}_{self.config.special_hash}') /
                                   str(self.config.image_downsample))

        if self.sequence_starts is not None:
            assert type(initial_segmentation) is list
            assert type(initial_image) is list
            assert len(initial_image) == len(self.sequence_starts)
            assert len(initial_segmentation) == len(self.sequence_starts)
            assert frame_i in self.sequence_starts

            sequence_starts_idx = self.sequence_starts.index(frame_i)

            initial_image = initial_image[sequence_starts_idx]
            initial_segmentation = initial_segmentation[sequence_starts_idx]

            first_image_frame = self.sequence_starts[sequence_starts_idx]
            if sequence_starts_idx >= len(self.sequence_starts) - 1:
                last_image_frame = len(self.sequence_starts)
            else:
                last_image_frame = self.sequence_starts[sequence_starts_idx + 1]

            images_paths = [images_paths[i] for i in range(len(images_paths))]
            segmentation_paths = [segmentation_paths[i] for i in range(len(images_paths))]

            cache_folder_SAM2: Path = cache_folder_SAM2 / f'{first_image_frame}_to_{last_image_frame}'

        if self.gt_Se3_cam2obj is not None:

            if self.config.segmentation_provider == 'synthetic' or self.config.frame_provider == 'synthetic':
                assert set(range(self.config.input_frames)).issubset(self.gt_Se3_cam2obj.keys()), \
                    f"Missing keys: {set(range(self.config.input_frames)) - self.gt_Se3_cam2obj.keys()}"

            all_gt_T_cam2obj = [self.gt_Se3_cam2obj[i].matrix() for i in sorted(self.gt_Se3_cam2obj.keys())]
            gt_T_cam2obj = torch.stack(all_gt_T_cam2obj)
            gt_Se3_cam2obj = Se3.from_matrix(gt_T_cam2obj)
            Se3_obj_1_to_obj_i = Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(gt_Se3_cam2obj)
        else:
            Se3_obj_1_to_obj_i = None

        self.tracker = BaseTracker(self.config, gt_mesh=gt_mesh, gt_texture=gt_texture,
                                   gt_Se3_obj1_to_obj_i=Se3_obj_1_to_obj_i,
                                   initial_segmentation=initial_segmentation,
                                   initial_image=initial_image, images_paths=images_paths, video_path=video_path,
                                   segmentation_paths=segmentation_paths,
                                   segmentation_video_path=segmentation_video_path, sam2_cache_folder=cache_folder_SAM2)

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
        self.results_writer.log_keyframe_info(keyframe_graph)
        reconstruction, alignment_success = self.run_reconstruction(images_paths, segmentation_paths, matching_pairs)

        if reconstruction is not None and alignment_success:
            view_graph = view_graph_from_datagraph(keyframe_graph, self.data_graph, reconstruction)
            view_graph.save(self.cache_folder_view_graph, save_images=True)

            reconstruction_path = self.cache_folder_view_graph / 'reconstruction' / '0'
            reconstruction_path.mkdir(exist_ok=True, parents=True)
            reconstruction.write(str(reconstruction_path))

            self.results_writer.visualize_colmap_track(self.config.input_frames - 1, reconstruction)

            view_graph = view_graph_from_datagraph(keyframe_graph, self.data_graph, reconstruction)
            view_graph.save(self.cache_folder_view_graph, save_images=True)

        csv_detailed_stats = self.write_folder.parent.parent / 'stats.csv'
        csv_per_sequence_stats = self.write_folder.parent.parent / 'global_stats.csv'

        if reconstruction is not None:
            self.evaluate_reconstruction(reconstruction, csv_detailed_stats, self.config.dataset, self.config.sequence)

        num_keyframes = len(keyframe_graph.nodes)
        reconstruction_success = reconstruction is not None
        update_global_statistics(csv_detailed_stats, csv_per_sequence_stats, num_keyframes, reconstruction,
                                 self.config.dataset, self.config.sequence, reconstruction_success,
                                 alignment_success)

        return

    def filter_frames(self):
        for frame_i in range(0, self.config.input_frames):
            if self.sequence_starts is not None and frame_i in self.sequence_starts:
                pass

            self.init_datagraph_frame(frame_i)

            new_frame_observation = self.tracker.next(frame_i)

            new_frame_node = self.data_graph.get_frame_data(frame_i)
            new_frame_node.frame_observation = new_frame_observation.send_to_device('cpu')
            new_frame_node.image_shape = self.tracker.get_image_size()

            start = time.time()

            self.frame_filter.filter_frames(frame_i)
            if self.config.densify_view_graph:
                self.frame_filter.densify()

            self.results_writer.write_results(frame_i=frame_i, keyframe_graph=self.frame_filter.keyframe_graph)

            print('Elapsed time in seconds: ', time.time() - start, "Frame ", frame_i, "out of",
                  self.config.input_frames)

    def run_reconstruction(self, images_paths, segmentation_paths, matching_pairs) ->\
            Tuple[Optional[Reconstruction], bool]:

        align_success = True
        reconstruction = None
        try:
            reconstruction = self.glomap_wrapper.run_glomap_from_image_list(images_paths, segmentation_paths,
                                                                            matching_pairs)
        except:
            pass

        if reconstruction is None:
            return reconstruction, False
        if self.config.similarity_transformation == 'first_frame':
            gt_Se3_obj2cam = self.initial_gt_Se3_cam2obj.inverse()
            reconstruction, align_success = self.glomap_wrapper.align_with_first_pose(reconstruction, gt_Se3_obj2cam, 0)
        elif self.config.similarity_transformation == 'kabsch':
            reconstruction = self.glomap_wrapper.align_with_kabsch(reconstruction)
        else:
            raise ValueError(f'Unknown similarity transform method {self.config.similarity_transformation}')

        return reconstruction, align_success

    def evaluate_reconstruction(self, reconstruction, csv_output_path: Path, dataset: str, sequence: str):

        stats = []

        images_paths_to_frame_index = {str(self.data_graph.get_frame_data(i).image_filename.name): i
                                       for i in range(self.config.input_frames)}

        for image in reconstruction.images.values():
            image_frame_id = images_paths_to_frame_index[image.name]

            Se3_obj2cam_pred = get_image_Se3_world2cam(image, 'cpu')

            frame_data = self.data_graph.get_frame_data(image_frame_id)
            Se3_cam2obj_gt = frame_data.gt_Se3_cam2obj
            Se3_obj2cam_gt = Se3_cam2obj_gt.inverse() if Se3_cam2obj_gt is not None else None

            # Ground-truth rotation and translation
            gt_rotation = Se3_obj2cam_gt.rotation.matrix().tolist() if Se3_obj2cam_gt is not None else None
            gt_translation = Se3_obj2cam_gt.translation.tolist() if Se3_obj2cam_gt is not None else None

            pred_rotation = Se3_obj2cam_pred.rotation.matrix().tolist()
            pred_translation = Se3_obj2cam_pred.translation.tolist()

            # Add stats for the current image frame
            stats.append({
                'dataset': dataset,
                'sequence': sequence,
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
            filtered_df = existing_df[~((existing_df['dataset'] == dataset) &
                                        (existing_df['sequence'] == sequence))]
            updated_df = pd.concat([filtered_df, stats_df], ignore_index=True)
            updated_df.to_csv(csv_output_path, index=False)
        else:
            stats_df.to_csv(csv_output_path, index=False)

    def init_datagraph_frame(self, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)

        if self.gt_Se3_cam2obj is not None and frame_i in set(self.gt_Se3_cam2obj.keys()):
            frame_node.gt_Se3_cam2obj = self.gt_Se3_cam2obj[frame_i]

        if self.images_paths is not None:
            frame_node.image_filename = Path(self.images_paths[frame_i].name)
        elif self.video_path is not None:
            frame_node.image_filename = Path(f'{self.video_path.stem}_{frame_i}.png')

        if self.segmentation_paths is not None:
            frame_node.segmentation_filename = Path(self.segmentation_paths[frame_i].name)
        elif self.segmentation_video_path is not None:
            frame_node.segmentation_filename = Path(f'{self.segmentation_video_path.stem}_{frame_i}.png')


def run_tracking_on_sequence(config: TrackerConfig, write_folder: Path, **kwargs):
    sfb = Tracker6D(config, write_folder, **kwargs)
    sfb.run_filtering_with_reconstruction()

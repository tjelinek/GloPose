import copy
import os
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import imageio
import networkx as nx
import torch
from kornia.geometry import Se3, PinholeCamera
from pycolmap import Reconstruction

from data_providers.flow_provider import PrecomputedRoMaFlowProviderDirect, PrecomputedUFMFlowProviderDirect
from data_providers.frame_provider import FrameProviderAll, SAM2SegmentationProvider
from data_providers.matching_provider_sift import PrecomputedSIFTMatchingProvider
from data_structures.data_graph import DataGraph
from data_structures.view_graph import view_graph_from_datagraph
from pose.frame_filter import RoMaFrameFilter, FrameFilterSift, RoMaFrameFilterRANSAC, FrameFilterPassThrough
from pose.glomap import align_reconstruction_with_pose, align_with_kabsch, reconstruct_images_using_sfm
from tracker_config import TrackerConfig
from utils.eval_reconstruction import evaluate_reconstruction, update_sequence_reconstructions_stats, \
    update_dataset_reconstruction_statistics
from utils.eval_sam import update_iou_frame_statistics
from utils.results_logging import WriteResults
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i


class Tracker6D:

    def __init__(self, config: TrackerConfig, write_folder: Path, gt_texture=None, gt_mesh=None,
                 images_paths: Optional[List[Path]] = None, video_path: Optional[Path] = None,
                 gt_Se3_cam2obj: Optional[Dict[int, Se3]] = None, gt_Se3_world2cam: Optional[Dict[int, Se3]] = None,
                 gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = None,
                 segmentation_video_path: Optional[Path] = None, segmentation_paths: List[Path] = None,
                 depth_paths: Optional[List[Path]] = None, initial_image: torch.Tensor | List[torch.Tensor] = None,
                 initial_segmentation: torch.Tensor | List[torch.Tensor] = None, sequence_starts: List[int] = None):

        self.write_folder: Path = write_folder
        self.config: TrackerConfig = config

        if os.path.exists(self.write_folder):
            shutil.rmtree(self.write_folder)

        self.write_folder.mkdir(exist_ok=True, parents=True)

        skip = config.skip_indices
        if skip != 1:
            used_indices = range(0, config.input_frames, skip)
            config.input_frames = config.input_frames // skip

            if gt_Se3_cam2obj is not None:
                gt_Se3_cam2obj = {i // skip: gt_Se3_cam2obj[i] for i in used_indices if i in gt_Se3_cam2obj}
            if gt_pinhole_params is not None:
                gt_pinhole_params = {i // skip: gt_pinhole_params[i] for i in used_indices if i in gt_pinhole_params}
            if gt_Se3_world2cam is not None:
                gt_Se3_world2cam = {i // skip: gt_Se3_world2cam[i] for i in used_indices if i in gt_Se3_world2cam}

        # Paths
        self.images_paths: Optional[List[Path]] = images_paths
        self.segmentation_paths: Optional[List[Path]] = segmentation_paths
        self.depth_paths: Optional[List[Path]] = depth_paths
        self.video_path: Optional[Path] = video_path
        self.segmentation_video_path: Optional[Path] = segmentation_video_path
        self.sequence_starts: Optional[List[int]] = sequence_starts

        # Ground truth related
        self.gt_Se3_cam2obj: Optional[Dict[int, Se3]] = gt_Se3_cam2obj
        self.gt_Se3_world2cam: Optional[Dict[int, Se3]] = gt_Se3_world2cam
        self.gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = gt_pinhole_params

        # Initialization stuff
        self.initial_image: torch.Tensor = initial_image
        self.initial_segmentation: torch.Tensor = initial_segmentation

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Frame provider
        self.tracker: Optional[FrameProviderAll] = None

        # Other utilities and flags
        self.results_writer = None

        self.data_graph: DataGraph = DataGraph(out_device=self.config.device)

        self.matching_cache_folder: Path = (Path(f'/mnt/personal/jelint19/cache/{self.config.dense_matching}_cache') /
                                            config.roma_config.config_name / config.dataset /
                                            f'{config.sequence}_{config.special_hash}')
        self.cache_folder_SIFT: Path = (Path('/mnt/personal/jelint19/cache/SIFT_cache') /
                                        config.sift_matcher_config.config_name / config.dataset /
                                        f'{config.sequence}_{config.special_hash}')
        self.cache_folder_SAM2: Path = ((Path('/mnt/personal/jelint19/cache/SAM_cache') / self.config.dataset /
                                         f'{self.config.sequence}_{self.config.special_hash}') /
                                        str(self.config.image_downsample))

        self.cache_folder_view_graph: Path = (Path('/mnt/personal/jelint19/cache/view_graph_cache') /
                                              config.experiment_name / config.dataset /
                                              f'{config.sequence}_{config.special_hash}')

        self.colmap_base_path: Path = self.write_folder / f'glomap_{self.config.sequence}'
        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'

        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)

        self.initialize_frame_provider(gt_mesh, gt_texture, images_paths, initial_image, initial_segmentation,
                                       segmentation_paths, segmentation_video_path, video_path, depth_paths, 0)

        self.results_writer = WriteResults(write_folder=self.write_folder, tracking_config=self.config,
                                           data_graph=self.data_graph)

        if self.config.dense_matching == 'RoMa':
            self.flow_provider = PrecomputedRoMaFlowProviderDirect(self.config.device, self.config.roma_config,
                                                                   self.matching_cache_folder, self.data_graph,
                                                                   allow_disk_cache=self.config.dense_matching_allow_disk_cache,
                                                                   purge_cache=self.config.purge_cache)
        elif self.config.dense_matching == 'UFM':
            self.flow_provider = PrecomputedUFMFlowProviderDirect(self.config.device, self.config.ufm_config,
                                                                  self.matching_cache_folder, self.data_graph,
                                                                  allow_disk_cache=self.config.dense_matching_allow_disk_cache,
                                                                  purge_cache=self.config.purge_cache)
        else:
            raise ValueError(f'Unknown dense matching option {self.config.dense_matching}')

        self.frame_filter: Union[RoMaFrameFilter, FrameFilterSift]
        if self.config.frame_filter == 'dense_matching':
            self.frame_filter = RoMaFrameFilter(self.config, self.data_graph, self.flow_provider)
        elif self.config.frame_filter == 'passthrough':
            self.frame_filter = FrameFilterPassThrough(self.config, self.data_graph)
        elif self.config.frame_filter == 'RoMaRANSAC':
            self.frame_filter = RoMaFrameFilterRANSAC(self.config, self.data_graph, self.flow_provider)
        elif self.config.frame_filter == 'SIFT':
            sift_matcher = PrecomputedSIFTMatchingProvider(self.data_graph,
                                                           self.config.sift_matcher_config.sift_filter_num_feats,
                                                           self.cache_folder_SIFT, device=self.config.device)
            self.frame_filter = FrameFilterSift(self.config, self.data_graph, sift_matcher)
        else:
            raise ValueError(f'Unknown frame_filter {self.config.frame_filter}')

    def initialize_frame_provider(self, gt_mesh: torch.Tensor, gt_texture: torch.Tensor, images_paths: List[Path],
                                  initial_image: torch.Tensor | List[torch.Tensor],
                                  initial_segmentation: torch.Tensor | List[torch.Tensor],
                                  segmentation_paths: List[Path], segmentation_video_path: Path, video_path: Path,
                                  depth_paths: List[Path], frame_i: int):

        cache_folder_SAM2 = self.cache_folder_SAM2

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

        self.tracker = FrameProviderAll(self.config, gt_mesh=gt_mesh, gt_texture=gt_texture,
                                        gt_Se3_obj1_to_obj_i=Se3_obj_1_to_obj_i,
                                        initial_segmentation=initial_segmentation,
                                        initial_image=initial_image, images_paths=images_paths, video_path=video_path,
                                        segmentation_paths=segmentation_paths, depth_paths=depth_paths,
                                        segmentation_video_path=segmentation_video_path,
                                        sam2_cache_folder=cache_folder_SAM2, write_folder=self.write_folder)

    def dump_frame_node_for_glomap(self, frame_idx: int):

        device = self.config.device

        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to(device)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze(0).permute(1, 2, 0).to(device)

        if frame_data.image_filename is not None:
            image_filename = frame_data.image_filename
        else:
            image_filename = f'node_{frame_idx}.png'

        if frame_data.segmentation_filename is not None:
            seg_filename = frame_data.segmentation_filename
        else:
            seg_filename = f'segment_{frame_idx}.png'

        node_save_path = self.colmap_image_path / image_filename
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8).numpy(force=True))

        segmentation_save_path = self.colmap_seg_path / seg_filename
        imageio.v3.imwrite(segmentation_save_path, (img_seg * 255).to(torch.uint8).repeat(1, 1, 3).numpy(force=True))

        frame_data.image_save_path = copy.deepcopy(node_save_path)
        frame_data.segmentation_save_path = copy.deepcopy(segmentation_save_path)

    def run_pipeline(self):

        if self.config.evaluate_sam2_only:
            self.evaluate_sam()
            return

        start_time = time.time()
        keyframe_graph = self.filter_frames()

        keyframe_nodes_idxs = list(sorted(keyframe_graph.nodes()))
        images_paths, segmentation_paths, matching_pairs = self.prepare_input_for_colmap(keyframe_graph)

        end_time = time.time()

        frame_filtering_time = end_time - start_time

        start_time = time.time()
        reconstruction, alignment_success = self.run_reconstruction(images_paths, segmentation_paths, matching_pairs)
        end_time = time.time()
        reconstruction_time = end_time - start_time

        if self.gt_Se3_world2cam is not None and len(self.gt_Se3_world2cam.keys()) > 0:
            known_gt_poses = all(frm_idx in self.gt_Se3_world2cam.keys() for frm_idx in keyframe_nodes_idxs)
        else:
            known_gt_poses = None
        if reconstruction is not None and alignment_success:
            colmap_db_path = self.colmap_base_path / 'database.db'
            colmap_output_path = self.colmap_base_path / 'output'
            view_graph = view_graph_from_datagraph(keyframe_graph, self.data_graph, reconstruction, colmap_db_path,
                                                   colmap_output_path, self.config.object_id)
            view_graph.save_viewgraph(self.cache_folder_view_graph, reconstruction, save_images=True,
                                      overwrite=True, to_cpu=True)

            self.results_writer.visualize_colmap_track(self.config.input_frames - 1, reconstruction, known_gt_poses)
        elif reconstruction is not None:
            self.results_writer.visualize_colmap_track(self.config.input_frames - 1, reconstruction, False)
        else:
            if reconstruction is None:
                print("!!!Reconstruction failed")
            if not alignment_success:
                print("!!!Alignment failed")

        rec_csv_detailed_stats = self.write_folder.parent.parent / 'reconstruction_keyframe_stats.csv'
        rec_csv_per_sequence_stats = self.write_folder.parent.parent / 'reconstruction_sequence_stats.csv'
        dataset_name_for_eval = self.config.dataset
        if self.config.bop_config.onboarding_type is not None:
            dataset_name_for_eval = f'{dataset_name_for_eval}_{self.config.bop_config.onboarding_type}_onboarding'

        sequence_name = self.config.sequence
        if self.config.special_hash is not None and len(self.config.special_hash) > 0:
            sequence_name = f'{sequence_name}_{self.config.special_hash}'

        if reconstruction is not None:
            image_name_to_frame_id = {}

            for i in range(self.config.input_frames):
                frame_data = self.data_graph.get_frame_data(i)
                image_name_to_frame_id[str(frame_data.image_filename.name)] = i

            if known_gt_poses:
                evaluate_reconstruction(reconstruction, self.gt_Se3_world2cam, image_name_to_frame_id,
                                        rec_csv_detailed_stats, dataset_name_for_eval, sequence_name)

        num_keyframes = len(keyframe_graph.nodes)
        reconstruction_success = reconstruction is not None

        if known_gt_poses:
            update_sequence_reconstructions_stats(rec_csv_detailed_stats, rec_csv_per_sequence_stats, num_keyframes,
                                                  self.config.input_frames, reconstruction, dataset_name_for_eval,
                                                  sequence_name, reconstruction_success, alignment_success,
                                                  frame_filtering_time, reconstruction_time)
            update_dataset_reconstruction_statistics(rec_csv_per_sequence_stats, dataset_name_for_eval)

        return

    def prepare_input_for_colmap(self, keyframe_graph: nx.DiGraph) -> \
            Tuple[List[Path], List[Path], List[Tuple[int, int]]]:
        keyframe_nodes_idxs = list(sorted(keyframe_graph.nodes()))

        images_paths = []
        segmentation_paths = []
        matching_pairs = []
        for node_idx in keyframe_nodes_idxs:
            self.dump_frame_node_for_glomap(node_idx)
            frame_data = self.data_graph.get_frame_data(node_idx)

            images_paths.append(frame_data.image_save_path)
            segmentation_paths.append(frame_data.segmentation_save_path)
        for frame1_idx, frame2_idx in keyframe_graph.edges:
            u_index = keyframe_nodes_idxs.index(frame1_idx)
            v_index = keyframe_nodes_idxs.index(frame2_idx)
            matching_pairs.append((u_index, v_index))
        assert len(keyframe_nodes_idxs) > 2
        print(keyframe_graph.edges)

        return images_paths, segmentation_paths, matching_pairs

    def filter_frames(self) -> nx.DiGraph:
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

            self.results_writer.write_results(frame_i=frame_i, keyframe_graph=self.frame_filter.keyframe_graph)

            print(f'Elapsed time in seconds: {time.time() - start:.3f}s, frame {frame_i} out of '
                  f'{self.config.input_frames - 1}')

        keyframe_graph = self.frame_filter.get_keyframe_graph()

        return keyframe_graph

    def run_reconstruction(self, images_paths, segmentation_paths, matching_pairs) -> \
            Tuple[Optional[Reconstruction], bool]:

        first_frame_data = self.data_graph.get_frame_data(0)
        camera_K = first_frame_data.gt_pinhole_K
        reconstruction = reconstruct_images_using_sfm(images_paths, segmentation_paths, matching_pairs,
                                                      self.config.init_with_first_two_images, self.config.mapper,
                                                      self.flow_provider, self.config.roma_sample_size,
                                                      self.colmap_base_path, camera_K, self.config.device)

        if reconstruction is None or self.gt_Se3_world2cam is None:
            return reconstruction, False
        if self.config.similarity_transformation == 'depths':

            first_image_filename = str(first_frame_data.image_filename)

            gt_Se3_obj2cam = self.gt_Se3_world2cam[0]

            image_depths = {}
            for i in self.data_graph.G.nodes:
                frame_data = self.data_graph.get_frame_data(i)
                image_depths[str(frame_data.image_filename)] = frame_data.frame_observation.depth.squeeze()

            reconstruction, align_success = align_reconstruction_with_pose(reconstruction, gt_Se3_obj2cam, image_depths,
                                                                           first_image_filename)
        elif self.config.similarity_transformation == 'kabsch':
            gt_Se3_world2cam_poses = {
                str(self.data_graph.get_frame_data(n).image_filename):
                    self.data_graph.get_frame_data(n).gt_Se3_world2cam
                for n in self.data_graph.G.nodes
            }
            reconstruction, align_success = align_with_kabsch(reconstruction, gt_Se3_world2cam_poses)
        else:
            raise ValueError(f'Unknown similarity transform method {self.config.similarity_transformation}')

        return reconstruction, align_success

    def init_datagraph_frame(self, frame_i):
        self.data_graph.add_new_frame(frame_i)

        frame_node = self.data_graph.get_frame_data(frame_i)

        if self.gt_Se3_cam2obj is not None and (gt_Se3_cam2obj := self.gt_Se3_cam2obj.get(frame_i)):
            frame_node.gt_Se3_cam2obj = gt_Se3_cam2obj

        if self.gt_pinhole_params is not None and (gt_pinhole_params := self.gt_pinhole_params.get(frame_i)):
            frame_node.gt_pinhole_K = gt_pinhole_params.intrinsics.squeeze()

        if self.gt_Se3_world2cam is not None and (gt_Se3_world2cam := self.gt_Se3_world2cam.get(frame_i)):
            frame_node.gt_Se3_world2cam = gt_Se3_world2cam

        frame_node.image_filename = Path(self.tracker.get_n_th_image_name(frame_i))

        if self.segmentation_paths is not None:
            frame_node.segmentation_filename = Path(self.segmentation_paths[frame_i * self.config.skip_indices].name)
        elif self.segmentation_video_path is not None:
            frame_node.segmentation_filename = Path(f'{self.segmentation_video_path.stem}_'
                                                    f'{frame_i * self.config.skip_indices}.png')

    def evaluate_sam(self):

        from data_providers.frame_provider import PrecomputedSegmentationProvider
        import numpy as np

        precomputed_segmentation_provider = PrecomputedSegmentationProvider(self.config, self.tracker.image_shape,
                                                                            self.segmentation_paths)

        sam2_segmentation_provider = SAM2SegmentationProvider(self.config, self.tracker.image_shape,
                                                              self.initial_segmentation, self.tracker.frame_provider,
                                                              self.write_folder, self.images_paths,
                                                              self.cache_folder_SAM2)

        iou_list = []
        for frame_i in range(self.config.input_frames):
            image = self.tracker.frame_provider.next_image(frame_i)

            gt_segmentation = precomputed_segmentation_provider.next_segmentation(frame_i).squeeze()
            sam2_segmentation = sam2_segmentation_provider.next_segmentation(frame_i, image).squeeze()

            assert gt_segmentation.shape == sam2_segmentation.shape

            iou = (torch.min(gt_segmentation, sam2_segmentation).sum() /
                   (torch.max(gt_segmentation, sam2_segmentation).sum() + 1e-10))

            iou_list.append(iou.item())

        iou_np = np.array(iou_list)

        csv_per_frame_iou_folder = self.write_folder.parent.parent / 'sam_stats'
        csv_per_frame_iou_folder.mkdir(exist_ok=True, parents=True)
        csv_per_frame_iou_stats = csv_per_frame_iou_folder / f'{self.config.dataset}_{self.config.sequence}.csv'

        update_iou_frame_statistics(csv_per_frame_iou_stats, iou_np, self.config.dataset, self.config.sequence)


def run_tracking_on_sequence(config: TrackerConfig, write_folder: Path, **kwargs):
    sfb = Tracker6D(config, write_folder, **kwargs)
    sfb.run_pipeline()

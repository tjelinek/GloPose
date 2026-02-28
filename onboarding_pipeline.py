import copy
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import imageio
import networkx as nx
import torch
from PIL import Image
from kornia.geometry import Se3, PinholeCamera
from pycolmap import Reconstruction

from data_providers.flow_provider import FlowCache, create_flow_provider
from data_providers.frame_provider import FrameProviderAll
from data_structures.data_graph import DataGraph
from data_structures.view_graph import ViewGraph, view_graph_from_datagraph
from eval.eval_onboarding import resolve_gt_model_path
from configs.glopose_config import GloPoseConfig
from pose.frame_filter import create_frame_filter
from pose.glomap import align_reconstruction_with_pose, align_with_kabsch, reconstruct_images_using_sfm
from utils.math_utils import Se3_cam_to_obj_to_Se3_obj_1_to_obj_i
from utils.results_logging import WriteResults


logger = logging.getLogger(__name__)


class OnboardingPipeline:

    def __init__(self, config: GloPoseConfig, write_folder: Path, input_images: Union[List[Path], Path],
                 gt_texture=None, gt_mesh=None, gt_Se3_cam2obj: Optional[Dict[int, Se3]] = None,
                 gt_Se3_world2cam: Optional[Dict[int, Se3]] = None,
                 gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = None,
                 input_segmentations: Union[List[Path], Path] = None, depth_paths: Optional[List[Path]] = None,
                 initial_segmentation: Union[torch.Tensor, List[torch.Tensor]] = None,
                 progress=None):

        self.write_folder: Path = write_folder
        self.config: GloPoseConfig = config

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'Processing sequence written into {write_folder}')
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        self.progress = progress

        cache_root = config.paths.cache_folder
        self.matching_cache_folder: Path = \
            (cache_root / f'{self.config.onboarding.filter_matcher}_cache' /
             config.onboarding.roma.config_name / config.run.dataset / f'{config.run.sequence}_{config.run.special_hash}')
        self.cache_folder_SIFT: Path = (cache_root / 'SIFT_cache' /
                                        config.onboarding.sift.config_name / config.run.dataset /
                                        f'{config.run.sequence}_{config.run.special_hash}')
        self.cache_folder_SAM2: Path = ((cache_root / 'SAM_cache' / self.config.run.dataset /
                                         f'{self.config.run.sequence}_{self.config.run.special_hash}') /
                                        str(self.config.input.image_downsample))

        self.cache_folder_view_graph: Path = (cache_root / 'view_graph_cache' /
                                              config.run.experiment_name / config.run.dataset /
                                              f'{config.run.sequence}_{config.run.special_hash}')

        self.colmap_base_path: Path = self.write_folder / f'glomap_{self.config.run.sequence}'
        self.colmap_image_path = self.colmap_base_path / 'images'
        self.colmap_seg_path = self.colmap_base_path / 'segmentations'

        self.prepare_output_folder()

        skip = config.input.skip_indices
        if skip != 1:
            used_indices = range(0, config.input.input_frames, skip)
            config.input.input_frames = config.input.input_frames // skip

            if gt_Se3_cam2obj is not None:
                gt_Se3_cam2obj = {i // skip: gt_Se3_cam2obj[i] for i in used_indices if i in gt_Se3_cam2obj}
            if gt_pinhole_params is not None:
                gt_pinhole_params = {i // skip: gt_pinhole_params[i] for i in used_indices if i in gt_pinhole_params}
            if gt_Se3_world2cam is not None:
                gt_Se3_world2cam = {i // skip: gt_Se3_world2cam[i] for i in used_indices if i in gt_Se3_world2cam}

        # Paths
        self.input_images: Union[List[Path], Path] = input_images
        self.input_segmentations: Optional[Union[List[Path], Path]] = input_segmentations
        self.depth_paths: Optional[List[Path]] = depth_paths

        # Ground truth related
        self.gt_Se3_cam2obj: Optional[Dict[int, Se3]] = gt_Se3_cam2obj
        self.gt_Se3_world2cam: Optional[Dict[int, Se3]] = gt_Se3_world2cam
        self.gt_pinhole_params: Optional[Dict[int, PinholeCamera]] = gt_pinhole_params

        # Initialization stuff
        self.initial_segmentation: torch.Tensor = initial_segmentation

        # Cameras
        self.pinhole_params: Optional[PinholeCamera] = None

        # Frame provider
        self.tracker: Optional[FrameProviderAll] = None

        # Other utilities and flags
        self.results_writer = None

        self.data_graph: DataGraph = DataGraph(out_device=self.config.run.device)

        self.initialize_frame_provider(gt_mesh, gt_texture, input_images, initial_segmentation,
                                       input_segmentations, depth_paths)

        self.results_writer = WriteResults(write_folder=self.write_folder, tracking_config=self.config,
                                           data_graph=self.data_graph)

        filtering_cache = FlowCache(self.config.run.device, self.matching_cache_folder, self.data_graph,
                                    allow_disk_cache=self.config.onboarding.allow_disk_cache,
                                    purge_cache=self.config.paths.purge_cache)
        self.match_provider_filtering = create_flow_provider(
            self.config.onboarding.filter_matcher, self.config, cache=filtering_cache)
        self.match_provider_reconstruction = create_flow_provider(
            self.config.onboarding.reconstruction_matcher, self.config)
        self.frame_filter = create_frame_filter(
            self.config, self.data_graph, self.match_provider_filtering)

    def initialize_frame_provider(self, gt_mesh: torch.Tensor, gt_texture: torch.Tensor,
                                  images_paths: List[Path] | Path, initial_segmentation: torch.Tensor,
                                  input_segmentations: List[Path] | Path, depth_paths: List[Path]):

        if self.gt_Se3_cam2obj is not None:

            if self.config.input.segmentation_provider == 'synthetic' or self.config.input.frame_provider == 'synthetic':
                assert set(range(self.config.input.input_frames)).issubset(self.gt_Se3_cam2obj.keys()), \
                    f"Missing keys: {set(range(self.config.input.input_frames)) - self.gt_Se3_cam2obj.keys()}"

            all_gt_T_cam2obj = [self.gt_Se3_cam2obj[i].matrix() for i in sorted(self.gt_Se3_cam2obj.keys())]
            gt_T_cam2obj = torch.stack(all_gt_T_cam2obj)
            gt_Se3_cam2obj = Se3.from_matrix(gt_T_cam2obj)
            Se3_obj_1_to_obj_i = Se3_cam_to_obj_to_Se3_obj_1_to_obj_i(gt_Se3_cam2obj)
        else:
            Se3_obj_1_to_obj_i = None

        self.tracker = FrameProviderAll(self.config, gt_mesh=gt_mesh, gt_texture=gt_texture,
                                        gt_Se3_obj1_to_obj_i=Se3_obj_1_to_obj_i,
                                        initial_segmentation=initial_segmentation, input_images=images_paths,
                                        input_segmentations=input_segmentations, depth_paths=depth_paths,
                                        sam2_cache_folder=self.cache_folder_SAM2, write_folder=self.write_folder,
                                        progress=self.progress)

    def dump_frame_node_for_glomap(self, frame_idx: int):

        device = self.config.run.device

        frame_data = self.data_graph.get_frame_data(frame_idx)

        img = frame_data.frame_observation.observed_image.squeeze().permute(1, 2, 0).to(device)
        img_seg = frame_data.frame_observation.observed_segmentation.squeeze(0).permute(1, 2, 0).to(device)

        image_filename = f'{frame_idx}.png'
        seg_filename = f'{frame_idx}.png.png'

        node_save_path = self.colmap_image_path / image_filename
        img_np = (img * 255).to(torch.uint8).numpy(force=True)
        img_pil = Image.fromarray(img_np, mode='RGB')
        imageio.v3.imwrite(node_save_path, (img * 255).to(torch.uint8).numpy(force=True))
        img_pil.save(node_save_path)

        segmentation_save_path = self.colmap_seg_path / seg_filename
        img_seg_np = (img_seg * 255).squeeze().to(torch.uint8).numpy(force=True)
        img_seg_pil = Image.fromarray(img_seg_np, mode='L')
        img_seg_pil.save(segmentation_save_path)

        frame_data.image_save_path = copy.deepcopy(node_save_path)
        frame_data.segmentation_save_path = copy.deepcopy(segmentation_save_path)

    def run_pipeline(self) -> ViewGraph:

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

        # Always create a ViewGraph (even if reconstruction failed)
        colmap_db_path = self.colmap_base_path / 'database.db'
        colmap_output_path = self.colmap_base_path / 'output'
        view_graph = view_graph_from_datagraph(keyframe_graph, self.data_graph, reconstruction, colmap_db_path,
                                               colmap_output_path, self.config.run.object_id)

        # Populate metadata on the ViewGraph
        view_graph.alignment_success = alignment_success and reconstruction is not None
        view_graph.frame_filtering_time = frame_filtering_time
        view_graph.reconstruction_time = reconstruction_time
        view_graph.num_input_frames = self.config.input.input_frames
        view_graph.gt_model_path = resolve_gt_model_path(self.config)

        # Build image_name_to_frame_id mapping
        image_name_to_frame_id = {}
        for i in range(self.config.input.input_frames):
            frame_data = self.data_graph.get_frame_data(i)
            image_name_to_frame_id[str(frame_data.image_filename.name)] = i
        view_graph.image_name_to_frame_id = image_name_to_frame_id

        # Determine GT pose availability for visualization
        if self.gt_Se3_world2cam is not None and len(self.gt_Se3_world2cam.keys()) > 0:
            known_gt_poses = all(frm_idx in self.gt_Se3_world2cam.keys() for frm_idx in keyframe_nodes_idxs)
        else:
            known_gt_poses = None

        # Save ViewGraph and visualize
        if reconstruction is not None and alignment_success:
            view_graph.save_viewgraph(self.cache_folder_view_graph, reconstruction, save_images=True,
                                      overwrite=True, to_cpu=True)
            self.results_writer.visualize_colmap_track(self.config.input.input_frames - 1, reconstruction,
                                                       known_gt_poses)
        elif reconstruction is not None:
            self.results_writer.visualize_colmap_track(self.config.input.input_frames - 1, reconstruction, False)
            logger.warning("Reconstruction succeeded but alignment failed for %s/%s",
                           self.config.run.dataset, self.config.run.sequence)
        else:
            logger.warning("Reconstruction failed for %s/%s (%d keyframes from %d input frames)",
                           self.config.run.dataset, self.config.run.sequence,
                           len(keyframe_nodes_idxs), self.config.input.input_frames)

        return view_graph

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
        print(sorted(keyframe_graph.edges))

        return images_paths, segmentation_paths, matching_pairs

    def filter_frames(self, progress=None, stop_event: threading.Event = None) -> nx.DiGraph:

        for frame_i in range(0, self.tracker.frame_provider.get_input_length()):

            if progress is not None:
                progress(frame_i / float(self.tracker.frame_provider.get_input_length()), desc="Filtering frames...")

            if stop_event is not None and stop_event.is_set():
                print('Computation stopped by the user.')
                return self.frame_filter.get_keyframe_graph()

            self.init_datagraph_frame(frame_i)

            new_frame_observation = self.tracker.next(frame_i)

            new_frame_node = self.data_graph.get_frame_data(frame_i)
            new_frame_node.frame_observation = new_frame_observation.send_to_device('cpu')
            new_frame_node.image_shape = self.tracker.get_image_size()

            start = time.time()

            self.frame_filter.filter_frames(frame_i)

            self.results_writer.write_results(frame_i=frame_i, keyframe_graph=self.frame_filter.keyframe_graph)

            print(f'Elapsed time in seconds: {time.time() - start:.3f}s, frame {frame_i} out of '
                  f'{self.config.input.input_frames - 1}')

        keyframe_graph = self.frame_filter.get_keyframe_graph()

        return keyframe_graph

    def run_reconstruction(self, images_paths, segmentation_paths, matching_pairs) -> \
            Tuple[Optional[Reconstruction], bool]:

        first_frame_data = self.data_graph.get_frame_data(0)
        camera_K = first_frame_data.gt_pinhole_K if not self.config.onboarding.use_default_colmap_K else None
        reconstruction = reconstruct_images_using_sfm(images_paths, segmentation_paths, matching_pairs,
                                                      self.config.onboarding.init_with_first_two_images,
                                                      self.config.onboarding.mapper,
                                                      self.match_provider_reconstruction,
                                                      self.config.onboarding.sample_size,
                                                      self.colmap_base_path,
                                                      self.config.onboarding.add_track_merging_matches,
                                                      camera_K, self.config.run.device)

        if reconstruction is None or self.gt_Se3_world2cam is None:
            return reconstruction, False
        if self.config.onboarding.similarity_transformation == 'depths':

            first_image_filename = str(first_frame_data.image_filename)

            gt_Se3_obj2cam = self.gt_Se3_world2cam[0]

            image_depths = {}
            for i in self.data_graph.G.nodes:
                frame_data = self.data_graph.get_frame_data(i)
                image_depths[str(frame_data.image_filename)] = frame_data.frame_observation.depth.squeeze()

            reconstruction, align_success = align_reconstruction_with_pose(reconstruction, gt_Se3_obj2cam, image_depths,
                                                                           first_image_filename)
        elif self.config.onboarding.similarity_transformation == 'kabsch':
            gt_Se3_world2cam_poses = {
                str(self.data_graph.get_frame_data(n).image_filename):
                    self.data_graph.get_frame_data(n).gt_Se3_world2cam
                for n in self.data_graph.G.nodes
            }
            reconstruction, align_success = align_with_kabsch(reconstruction, gt_Se3_world2cam_poses)
        else:
            raise ValueError(f'Unknown similarity transform method {self.config.onboarding.similarity_transformation}')

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

        frame_node.image_filename = Path(f'{frame_i}.png')

        if type(self.input_segmentations) is list:
            frame_node.segmentation_filename = Path(f'{frame_i}.png')

    def prepare_output_folder(self):
        """Wipe and recreate the output folder. Called early in __init__."""
        if os.path.exists(self.write_folder):
            shutil.rmtree(self.write_folder)
        self.write_folder.mkdir(exist_ok=True, parents=True)
        self.colmap_image_path.mkdir(exist_ok=True, parents=True)
        self.colmap_seg_path.mkdir(exist_ok=True, parents=True)

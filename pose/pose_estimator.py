import json
import pickle
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import pycolmap
import torchvision.ops as ops
import torchvision.transforms.functional as TF
import torch

import numpy as np
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from kornia.geometry import Se3
from tqdm import tqdm

from data_providers.flow_provider import RoMaFlowProviderDirect, UFMFlowProviderDirect, FlowProviderDirect
from data_providers.frame_provider import PrecomputedFrameProvider

from data_structures.view_graph import ViewGraph, load_view_graphs_by_object_id
from pose.frame_filter import compute_matching_reliability
from pose.glomap import unique_keypoints_from_matches
from tracker_config import TrackerConfig
from utils.bop_challenge import get_gop_camera_intrinsics
from utils.cnos_utils import get_default_detections_per_scene_and_image, get_detections_cnos_format
from visualizations.pose_estimation_visualizations import PoseEstimatorLogger
from repositories.cnos.segment_anything.utils.amg import rle_to_mask


class BOPChallengePosePredictor:

    def __init__(self, config: TrackerConfig):

        self.config = config
        self.flow_provider: Optional[FlowProviderDirect] = None

        self.write_folder = Path('/mnt/personal/jelint19/results/PoseEstimation')

        self._initialize_flow_provider()

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        cfg_dir = (Path(__file__).parent.parent / 'repositories' / 'cnos' / 'configs').resolve()
        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cnos_cfg = compose(config_name="run_inference")

        sys.path.append('./repositories/cnos')
        from src.model.loss import PairwiseSimilarity

        self.cnos_matching_config = instantiate(cnos_cfg.model.matching_config)
        self.cnos_postprocessing_config = instantiate(cnos_cfg.model.post_processing_config)
        self.cnos_similarity: PairwiseSimilarity = self.cnos_matching_config['metric']

    def _initialize_flow_provider(self) -> None:

        if self.config.frame_filter_matcher == 'RoMa':
            self.flow_provider = RoMaFlowProviderDirect(self.config.device, self.config.roma_config)
        elif self.config.frame_filter_matcher == 'UFM':
            self.flow_provider = UFMFlowProviderDirect(self.config.device, self.config.ufm_config)
        else:
            raise ValueError(f'Unknown dense matching option {self.config.frame_filter_matcher}')

    def predict_poses_for_bop_challenge(self, base_dataset_folder: Path, bop_targets_path: Path,
                                        view_graph_save_paths: Path, onboarding_type: str, split: str,
                                        method_name: str) -> None:

        view_graphs: Dict[Any, ViewGraph] = load_view_graphs_by_object_id(view_graph_save_paths, onboarding_type,
                                                                          self.config.device)

        rerun_folder = self.write_folder / 'rerun' / base_dataset_folder.stem
        rerun_folder.mkdir(exist_ok=True, parents=True)

        with bop_targets_path.open('r') as file:
            test_annotations = json.load(file)

        test_dataset_path = base_dataset_folder / split

        view_graph_descriptors: Dict[Any, torch.Tensor] = {
            obj_id: view_graph.get_concatenated_descriptors() for obj_id, view_graph in view_graphs.items()
        }

        json_2d_detection_results = []

        total_items = len(test_annotations)

        default_detections_file = Path('/mnt/personal/jelint19/data/bop/default_detections/h3_bop24_model_free_unseen/'
                                       'cnos-sam/onboarding_static/'
                                       'cnos-sam_hope-test_static-020a-45bd-8ec5-c95560b68011.json')

        default_detections_scene_im_dict = get_default_detections_per_scene_and_image(default_detections_file)

        for i, item in tqdm(enumerate(test_annotations), desc="Processing test annotations", total=total_items,
                            unit="items"):
            im_id = item['im_id']
            scene_id = item['scene_id']

            pose_logger = PoseEstimatorLogger(rerun_folder / f'scene-{scene_id}_im-{im_id}.rrd')

            # Construct paths
            scene_folder_name = f'{scene_id:06d}'
            image_id_str = f'{im_id:06d}'
            path_to_scene = test_dataset_path / scene_folder_name
            path_to_image = self._get_image_path(path_to_scene, image_id_str)
            path_to_camera_intrinsics = path_to_scene / 'scene_camera.json'
            path_to_cnos_detections = path_to_scene / 'cnos_sam_detections'
            path_to_detections_file = path_to_cnos_detections / f'{im_id:06d}.pkl'

            camera_intrinsics = get_gop_camera_intrinsics(path_to_camera_intrinsics, im_id)

            with open(path_to_detections_file, "rb") as detections_file:
                cnos_detections = pickle.load(detections_file)

            detections_start_time = time.time()
            detections = self.proces_custom_sam_detections(cnos_detections, view_graph_descriptors)

            detections_duration = time.time() - detections_start_time

            image = PrecomputedFrameProvider.load_and_downsample_image(
                path_to_image, self.config.image_downsample, self.config.device
            )
            image = image.squeeze()

            pose_logger.visualize_image(image)

            default_detections = get_detections_cnos_format(default_detections_scene_im_dict, scene_id, im_id,
                                                            self.config.device)

            for detection_mask_idx in tqdm(range(detections.masks.shape[0]), desc="Processing SAM mask proposals",
                                           total=detections.masks.shape[0], unit="items"):
                corresponding_obj_id: int = detections.object_ids[detection_mask_idx].item()
                corresponding_view_graph = view_graphs[corresponding_obj_id]
                proposal_mask = detections.masks[detection_mask_idx]

                # if pose_logger is not None:
                #     pose_logger.visualize_detections(proposal_mask)
                #     pose_logger.rerun_sequence_id += 1

                torchvision_bbox = ops.masks_to_boxes(proposal_mask[None].to(torch.float)).squeeze().to(torch.long)
                x0, y0, x1, y1 = torchvision_bbox.tolist()
                coco_bbox = [x0, y0, x1 - x0, y1 - y0]

                detection_result = {
                    'scene_id': scene_id,
                    'image_id': im_id,
                    'category_id': corresponding_obj_id,
                    'bbox': coco_bbox,
                    'time': detections_duration,
                    'score': detections.scores[detection_mask_idx].item(),
                }
                json_2d_detection_results.append(detection_result)

                # self.predict_poses(image, camera_intrinsics, corresponding_view_graph, self.flow_provider,
                #                    self.config.roma_sample_size,
                #                    match_min_certainty=self.config.min_roma_certainty_threshold,
                #                    match_reliability_threshold=self.config.flow_reliability_threshold,
                #                    query_img_segmentation=proposal_mask,
                #                    device=self.config.device, pose_logger=pose_logger)

        # {method}_{dataset}-{split}_{optional_id}.{ext}
        json_file_path = self.write_folder / (f'{method_name}_{base_dataset_folder.stem}-{split}_'
                                              f'{view_graph_save_paths.parent.stem}@{onboarding_type}.json')
        with open(json_file_path, 'w') as f:
            json.dump(json_2d_detection_results, f)

        print(f'Results saved to {str(json_file_path)}')

    def proces_custom_sam_detections(self, cnos_detections, view_graph_descriptors):
        from src.model.utils import Detections
        from src.model.detector import compute_templates_similarity_scores

        default_detections_descriptors = torch.from_numpy(cnos_detections['descriptors']).to(self.config.device)
        default_detections_masks = []
        for detection in cnos_detections['masks']:
            detection_mask = rle_to_mask(detection)
            detection_mask_tensor = torch.from_numpy(detection_mask).to(self.config.device)
            default_detections_masks.append(detection_mask_tensor)
        default_detections_masks = torch.stack(default_detections_masks, dim=0)

        idx_selected_proposals, selected_objects, pred_scores, pred_score_distribution = \
            compute_templates_similarity_scores(view_graph_descriptors, default_detections_descriptors,
                                                self.cnos_similarity,
                                                self.cnos_matching_config['aggregation_function'],
                                                self.cnos_matching_config['confidence_thresh'],
                                                self.cnos_matching_config['max_num_instances'])
        selected_detections_masks = default_detections_masks[idx_selected_proposals]
        detections_dict = {
            'masks': selected_detections_masks,
            'scores': pred_scores,
            'score_distribution': pred_score_distribution,
            'object_ids': selected_objects,
            'boxes': ops.masks_to_boxes(selected_detections_masks.to(torch.float)).to(torch.long)
        }
        detections = Detections(detections_dict)
        detections.apply_nms_per_object_id(
            nms_thresh=self.cnos_postprocessing_config['nms_thresh'],
        )
        return detections

    @staticmethod
    def _get_image_path(path_to_scene: Path, image_id_str: str) -> Path:

        # Try .png first
        image_filename = f'{image_id_str}.png'
        path_to_image = path_to_scene / 'rgb' / image_filename

        if not path_to_image.exists():
            image_filename = f'{image_id_str}.jpg'
            path_to_image = path_to_scene / 'rgb' / image_filename
            assert path_to_image.exists(), f"Image file not found: {path_to_image}"

        return path_to_image

    def predict_poses(self, query_img: torch.Tensor, camera_K: np.ndarray, view_graph: ViewGraph,
                      flow_provider: FlowProviderDirect, match_sample_size, match_min_certainty=0.,
                      match_reliability_threshold=0., query_img_segmentation: Optional[torch.Tensor] = None,
                      black_background: bool = True, pose_logger: PoseEstimatorLogger = None, device: str = 'cuda')\
            -> Se3 | None:
        # query_img_segmentation shape (H, W)

        path_to_colmap_db = view_graph.colmap_db_path
        path_to_reconstruction = view_graph.colmap_reconstruction_path

        path_to_cache = Path('/mnt/personal/jelint19/tmp/colmap_db_cache') / str(view_graph.object_id)
        cache_db_file = path_to_cache / path_to_colmap_db.name

        if path_to_cache.exists() and path_to_cache.is_dir():
            shutil.rmtree(path_to_cache)
        path_to_cache.mkdir(exist_ok=True, parents=True)
        shutil.copy(path_to_colmap_db, cache_db_file)

        database = pycolmap.Database(str(cache_db_file))

        h, w = query_img.shape[-2:]
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])

        new_camera_id = database.num_cameras + 1
        new_camera = pycolmap.Camera(camera_id=new_camera_id, model=pycolmap.CameraModelId.PINHOLE, width=w,
                                     height=h,
                                     params=[f_x, f_y, c_x, c_y])

        new_image_id = database.num_images + 1
        new_database_image = pycolmap.Image(image_id=new_image_id, camera_id=new_camera_id, name='tmp_target')

        database.write_camera(new_camera, use_camera_id=True)
        database.write_image(new_database_image, use_image_id=True)

        matching_edges: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        matching_edges_certainties: Dict[Tuple[int, int], torch.Tensor] = {}

        n = len(view_graph.view_graph.nodes())
        if n > 10:
            k = max(1, n // 10)
            while n // k < 10:
                k -= 1
        else:
            k = 1
        for frame_idx in list(view_graph.view_graph.nodes())[::k]:
            view_graph_node = view_graph.get_node_data(frame_idx)
            db_img_id = view_graph_node.colmap_db_image_id

            pose_graph_image = view_graph_node.observation.observed_image.to(device).squeeze()
            pose_graph_segmentation = view_graph_node.observation.observed_segmentation.to(device).squeeze()
            if black_background:
                pose_graph_image = pose_graph_image * pose_graph_segmentation

            if type(flow_provider) is FlowProviderDirect or True:
                query_img_resized = TF.resize(query_img, list(pose_graph_image.shape[-2:]))
                if query_img_segmentation is not None:
                    query_seg_resized = TF.resize(query_img_segmentation[None],
                                                  list(pose_graph_segmentation.shape[-2:])).squeeze()

                    if black_background:
                        query_img_resized = query_img_resized * query_seg_resized.to(torch.float)
                else:
                    query_seg_resized = None
                db_img_pts_xy, query_img_pts_xy, certainties = (
                    flow_provider.get_source_target_points(pose_graph_image, query_img_resized, match_sample_size,
                                                           pose_graph_segmentation, query_seg_resized,
                                                           as_int=True, zero_certainty_outside_segmentation=True,
                                                           only_foreground_matches=True))

                warp, certainty = flow_provider.compute_flow(pose_graph_image, query_img_resized,
                                                             source_image_segmentation=pose_graph_segmentation,
                                                             zero_certainty_outside_segmentation=True)
            else:
                raise NotImplementedError('So far we can only work with RoMaFlowProviderDirect')

            if db_img_pts_xy.shape[0] == 0:  # No good matches found within the segmentation
                continue

            reliability = compute_matching_reliability(db_img_pts_xy, certainties, pose_graph_segmentation,
                                                       match_min_certainty)

            if pose_logger is not None:
                pose_logger.visualize_pose_matching_rerun(db_img_pts_xy, query_img_pts_xy, certainties,
                                                          pose_graph_image, query_img_resized, reliability,
                                                          match_reliability_threshold, match_min_certainty, certainty,
                                                          viewgraph_image_segment=pose_graph_segmentation,
                                                          query_image_segment=query_seg_resized)
                pose_logger.rerun_sequence_id += 1

            print(f'Mean certainty: {certainties.mean().item()}, Reliability: {reliability}')
            if reliability >= match_reliability_threshold:
                matching_edges[(new_image_id, db_img_id)] = (query_img_pts_xy, db_img_pts_xy)
                matching_edges_certainties[(new_image_id, db_img_id)] = certainties

        if len(matching_edges) == 0:
            return None

        keypoints, edge_match_indices = \
            unique_keypoints_from_matches(matching_edges, database, eliminate_one_to_many_matches=True,
                                          matching_edges_certainties=matching_edges_certainties, device=device)

        all_image_ids = {img.image_id for img in database.read_all_images()}
        matched_images_ids = {node for edge in edge_match_indices.keys() for node in edge}
        non_matched_images_ids = all_image_ids - matched_images_ids
        non_matched_keypoints = {img_id: database.read_keypoints(img_id) for img_id in non_matched_images_ids}

        old_keypoints = {}
        for colmap_image_id in set(keypoints.keys()) - {new_image_id}:
            keypoints_np = database.read_keypoints(colmap_image_id)
            old_keypoints[colmap_image_id] = keypoints_np

        database.clear_keypoints()
        for colmap_image_id in sorted(keypoints.keys()):
            if colmap_image_id == new_image_id:
                keypoints_np = keypoints[colmap_image_id].numpy(force=True).astype(np.float32)
                database.write_keypoints(colmap_image_id, keypoints_np)
            else:
                keypoints_np = old_keypoints[colmap_image_id]
                database.write_keypoints(colmap_image_id, keypoints_np)

        for colmap_image_u, colmap_image_v in edge_match_indices.keys():
            match_indices_np = edge_match_indices[colmap_image_u, colmap_image_v].numpy(force=True)

            keypoints_u = database.read_keypoints(colmap_image_u)
            keypoints_v = database.read_keypoints(colmap_image_v)

            valid_match_mask = (match_indices_np[:, 0] < len(keypoints_u)) & (
                    match_indices_np[:, 1] < len(keypoints_v))
            filtered_match_indices = match_indices_np[valid_match_mask]

            database.write_matches(colmap_image_u, colmap_image_v, filtered_match_indices)

        for img_id, keypoints in non_matched_keypoints.items():
            database.write_keypoints(img_id, keypoints)

        database_cache = pycolmap.DatabaseCache().create(database, 0, False, set())

        reconstruction = pycolmap.Reconstruction()
        reconstruction.read(str(path_to_reconstruction))
        reconstruction.add_camera(new_camera)
        reconstruction.add_image(new_database_image)

        mapper = pycolmap.IncrementalMapper(database_cache)
        mapper.begin_reconstruction(reconstruction)
        mapper_options = pycolmap.IncrementalMapperOptions()

        # Register the new image
        print(f"Registering image #{new_image_id}")
        print(f"=> Image sees {mapper.observation_manager.num_visible_points3D(new_image_id)} / "
              f"{mapper.observation_manager.num_observations(new_image_id)} points")

        success = mapper.register_next_image(mapper_options, new_image_id)

        if success:
            print(f"Successfully registered image {new_image_id} into the reconstruction.")
            mapper.triangulate_image(
                mapper_options.triangulation, new_image_id
            )
            reconstruction.normalize()
            breakpoint()
        else:
            print(f"Failed to register image {new_image_id}.")

        return Se3.identity()


def main():
    dataset = 'hope'
    onboarding_type = 'static'
    config = 'ufm_c0975r05'
    method_name = 'FlowTemplates'

    base_dataset_folder = Path(f'/mnt/personal/jelint19/data/bop/{dataset}')
    bop_targets_path = base_dataset_folder / 'test_targets_bop24.json'
    view_graph_location = Path(f'/mnt/personal/jelint19/cache/view_graph_cache/{config}/{dataset}')

    config = TrackerConfig()
    config.device = 'cuda'
    predictor = BOPChallengePosePredictor(config)

    predictor.predict_poses_for_bop_challenge(base_dataset_folder, bop_targets_path, view_graph_location,
                                              onboarding_type, 'test', method_name)


if __name__ == '__main__':
    main()

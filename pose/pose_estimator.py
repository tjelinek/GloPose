import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torchvision.ops as ops
import torch

from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from tqdm import tqdm

from condensate_templates import get_descriptors_for_condensed_templates, TemplateBank, _l2n, _apply_whitener
from data_providers.flow_provider import RoMaFlowProviderDirect, UFMFlowProviderDirect, FlowProviderDirect
from data_providers.frame_provider import PrecomputedFrameProvider

from data_structures.view_graph import ViewGraph, load_view_graphs_by_object_id
from src.model.detector import filter_similarities_dict
from tracker_config import TrackerConfig
from utils.bop_challenge import get_gop_camera_intrinsics, group_test_targets_by_image, get_descriptors_for_templates
from utils.cnos_utils import get_default_detections_per_scene_and_image, get_detections_cnos_format
from utils.eval_bop_detection import evaluate_bop_coco, update_results_csv
from visualizations.pose_estimation_visualizations import PoseEstimatorLogger
from repositories.cnos.segment_anything.utils.amg import rle_to_mask


class BOPChallengePosePredictor:

    def __init__(self, config: TrackerConfig, base_cache_folder, matching_config_overrides: Dict[str, Any],
                 experiment_folder='default', ):

        self.config = config
        self.flow_provider: Optional[FlowProviderDirect] = None

        self.write_folder = Path('/mnt/personal/jelint19/results/PoseEstimation') / experiment_folder
        self.cache_folder = base_cache_folder

        self._initialize_flow_provider()

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        cfg_dir = (Path(__file__).parent.parent / 'repositories' / 'cnos' / 'configs').resolve()
        overrides = []

        for override_k, override_v in matching_config_overrides.items():
            if override_v is not None:
                overrides.append(f'model.matching_config.{override_k}={override_v}')

        with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
            cnos_cfg = compose(config_name="run_inference", overrides=overrides)

        self.cnos_matching_config = instantiate(cnos_cfg.model.matching_config)
        self.cnos_postprocessing_config = instantiate(cnos_cfg.model.post_processing_config)

    def _initialize_flow_provider(self) -> None:

        if self.config.frame_filter_matcher == 'RoMa':
            self.flow_provider = RoMaFlowProviderDirect(self.config.device, self.config.roma_config)
        elif self.config.frame_filter_matcher == 'UFM':
            self.flow_provider = UFMFlowProviderDirect(self.config.device, self.config.ufm_config)
        else:
            raise ValueError(f'Unknown dense matching option {self.config.frame_filter_matcher}')

    def predict_poses_for_bop_challenge(self, base_dataset_folder: Path, bop_targets_path: Path,
                                        detection_templates_save_folder, onboarding_type: str, split: str,
                                        method_name: str, experiment_name: str, view_graph_save_paths: Path = None,
                                        descriptor: str = 'dinov2', detector_name='sam',
                                        descriptor_mask_detections=True, default_detections_file: Path = None,
                                        templates_source: str = 'cnns', dry_run: bool = False) -> None:

        dataset_name = base_dataset_folder.stem
        rerun_folder = self.write_folder / experiment_name / f'rerun_{dataset_name}'
        rerun_folder.mkdir(exist_ok=True, parents=True)

        with bop_targets_path.open('r') as file:
            test_annotations = json.load(file)
            test_annotations = group_test_targets_by_image(test_annotations)

        test_dataset_path = base_dataset_folder / split

        template_cls_descriptors: Dict[int, torch.Tensor]
        template_images: Dict[int, torch.Tensor]
        template_segmentations: Dict[int, torch.Tensor]

        view_graphs: Dict[int, ViewGraph] = {}
        from src.model.dinov2 import descriptor_from_hydra
        dino_descriptor = descriptor_from_hydra(descriptor, descriptor_mask_detections)
        if templates_source == 'viewgraph':
            view_graphs: Dict[Any, ViewGraph] = load_view_graphs_by_object_id(view_graph_save_paths, onboarding_type,
                                                                              self.config.device)
            template_cls_descriptors = {
                obj_id: view_graph.compute_dino_descriptors_for_nodes(dino_descriptor)[0]
                for obj_id, view_graph in view_graphs.items()
            }
            template_images = {
                obj_id: view_graph.get_concatenated_images() for obj_id, view_graph in view_graphs.items()
            }
            template_segmentations = {
                obj_id: view_graph.get_concatenated_segmentations() for obj_id, view_graph in view_graphs.items()
            }

            template_data = TemplateBank(images=template_images, cls_desc=template_cls_descriptors,
                                         masks=template_segmentations)
        elif templates_source == 'cnns':
            template_data = get_descriptors_for_condensed_templates(
                detection_templates_save_folder,
                descriptor,
                self.cnos_matching_config['cosine_similarity_quantile'],
                self.cnos_matching_config['mahalanobis_quantile'],
                device=self.config.device
            )
        elif templates_source == 'prerendered':
            orig_split_path = base_dataset_folder / onboarding_type
            cache_split_path = self.cache_folder / f'{descriptor}_cache' / 'bop' / dataset_name / onboarding_type
            template_images, template_segmentations, template_cls_descriptors = \
                get_descriptors_for_templates(orig_split_path, cache_split_path, descriptor, self.config.device)

            template_data = TemplateBank(images=template_images, cls_desc=template_cls_descriptors,
                                         masks=template_segmentations)
        else:
            raise ValueError(f'Unknown templates_source {templates_source}')

        json_2d_detection_results = []

        total_items = len(test_annotations)

        if default_detections_file is not None:
            default_detections_scene_im_dict = get_default_detections_per_scene_and_image(default_detections_file)
        else:
            default_detections_scene_im_dict = None

        for i, item in tqdm(enumerate(test_annotations), desc="Processing test annotations", total=total_items,
                            unit="items"):
            im_id = item['im_id']
            scene_id = item['scene_id']

            downsample_factor = 0.5 if dataset_name in ['hope', 'handal'] else 1.0

            step = len(test_annotations) / 10
            if i == int(step * (i // int(step))) and i // int(step) < 10:
                pose_logger = PoseEstimatorLogger(rerun_folder / f'scene-{scene_id}_im-{im_id}.rrd', downsample_factor)
            else:
                pose_logger = None

            # Construct paths
            scene_folder_name = f'{scene_id:06d}'
            image_id_str = f'{im_id:06d}'
            path_to_scene = test_dataset_path / scene_folder_name
            path_to_scene_detection_cache = (self.cache_folder / 'detections_cache' / dataset_name / split /
                                             scene_folder_name)
            path_to_image = self._get_image_path(path_to_scene, image_id_str)
            path_to_camera_intrinsics = path_to_scene / 'scene_camera.json'
            path_to_cnos_detections = path_to_scene_detection_cache / f'cnos_{detector_name}_detections_{descriptor}'
            path_to_detections_file = path_to_cnos_detections / f'{im_id:06d}.pkl'

            # camera_intrinsics = get_gop_camera_intrinsics(path_to_camera_intrinsics, im_id)

            with open(path_to_detections_file, "rb") as detections_file:
                cnos_detections = pickle.load(detections_file)

            image = PrecomputedFrameProvider.load_and_downsample_image(
                path_to_image, self.config.image_downsample, self.config.device
            )
            image = image.squeeze()
            if pose_logger is not None:
                pose_logger.visualize_image(image)

            detections_start_time = time.time()

            detections, detections_scores = self.proces_custom_sam_detections(cnos_detections, template_data, image,
                                                                              dino_descriptor)

            detections_duration = time.time() - detections_start_time

            if default_detections_scene_im_dict is not None:
                default_detections = get_detections_cnos_format(default_detections_scene_im_dict, scene_id, im_id,
                                                                self.config.device)
                # detections = default_detections

            for detection_mask_idx in tqdm(range(detections.masks.shape[0]), desc="Processing SAM mask proposals",
                                           total=detections.masks.shape[0], unit="items", disable=True):
                corresponding_obj_id: int = detections.object_ids[detection_mask_idx].item()
                # corresponding_view_graph = view_graphs[corresponding_obj_id]
                proposal_mask = detections.masks[detection_mask_idx]

                if pose_logger is not None:
                    pose_logger.visualize_detections(detections.masks, detection_mask_idx)
                    pose_logger.visualize_nearest_neighbors(image, template_data.images, detection_mask_idx, detections,
                                                            detections_scores,
                                                            self.cnos_matching_config['similarity_metric'])
                    pose_logger.rerun_sequence_id += 1

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
        json_file_path = self.write_folder / experiment_name / (f'{method_name}_{base_dataset_folder.stem}-{split}_'
                                                                f'{onboarding_type}@{experiment_name}.json')
        with open(json_file_path, 'w') as f:
            json.dump(json_2d_detection_results, f)

        print(f'Results saved to {str(json_file_path)}')

        if dataset_name in ['hope', 'handal'] and split != 'val':
            return
        result_filename = json_file_path.name
        results_path = json_file_path.parent
        eval_path = self.write_folder / "bop_eval"
        datasets_path = "/mnt/personal/jelint19/data/bop/"
        targets_filename = "test_targets_bop19.json"
        if dataset_name in ['hope', 'handal'] and split == 'val':
            targets_filename = "val_targets_bop24.json"
            # Run evaluation

        try:
            metrics = evaluate_bop_coco(
                result_filename=result_filename,
                results_path=results_path,
                datasets_path=datasets_path,
                eval_path=eval_path,
                ann_type="bbox",
                targets_filename=targets_filename
            )
        except ValueError:
            return  # Empty detection results

        results_csv_path = self.write_folder / 'detection_results.csv'
        if not dry_run:
            update_results_csv(metrics, experiment_name, dataset_name, split, results_csv_path)

    def proces_custom_sam_detections(self, cnos_detections, template_data: TemplateBank, image, dino_descriptor,
                                     recompute_default_descriptors=True):
        from src.model.utils import Detections
        from src.model.detector import compute_templates_similarity_scores

        default_detections_masks = []
        for detection in cnos_detections['masks']:
            detection_mask = rle_to_mask(detection)
            detection_mask_tensor = torch.from_numpy(detection_mask).to(self.config.device)
            default_detections_masks.append(detection_mask_tensor)
        default_detections_masks = torch.stack(default_detections_masks, dim=0)

        if recompute_default_descriptors:
            detections_dict = {
                'masks': default_detections_masks,
                'boxes': ops.masks_to_boxes(default_detections_masks.to(torch.float)).to(torch.long)
            }
            cnos_detections_class_format = Detections(detections_dict)
            image_np = image.permute(1, 2, 0).numpy(force=True)
            default_detections_cls_descriptors, _ = dino_descriptor(image_np, cnos_detections_class_format)
        else:
            default_detections_cls_descriptors = torch.from_numpy(cnos_detections['descriptors']).to(self.config.device)

        if template_data.whitening_mean is not None and template_data.whitening_W is not None:
            mu_w = template_data.whitening_mean
            W_w = template_data.whitening_W
            default_detections_cls_descriptors = _apply_whitener(default_detections_cls_descriptors, mu_w, W_w)

        default_detections_cls_descriptors = _l2n(default_detections_cls_descriptors)

        idx_selected_proposals, selected_objects, pred_scores, pred_score_distribution, detections_scores = \
            compute_templates_similarity_scores(template_data, default_detections_cls_descriptors,
                                                self.cnos_matching_config['similarity_metric'],
                                                self.cnos_matching_config['aggregation_function'],
                                                self.cnos_matching_config['max_num_instances'],
                                                self.cnos_matching_config['confidence_thresh'],
                                                self.cnos_matching_config['lowe_ratio_threshold'],
                                                self.cnos_matching_config['ood_detection_method'],
                                                )
        selected_detections_masks = default_detections_masks[idx_selected_proposals]
        detections_dict = {
            'masks': selected_detections_masks,
            'scores': pred_scores,
            'score_distribution': pred_score_distribution,
            'object_ids': selected_objects,
            'boxes': ops.masks_to_boxes(selected_detections_masks.to(torch.float)).to(torch.long),
        }
        detections = Detections(detections_dict)
        keep_indices = detections.apply_nms_per_object_id(nms_thresh=self.cnos_postprocessing_config['nms_thresh'])
        filter_similarities_dict(detections_scores, keep_indices)
        keep_indices = detections.apply_nms_for_masks_inside_masks()
        filter_similarities_dict(detections_scores, keep_indices)
        return detections, detections_scores

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


def main():
    parser = argparse.ArgumentParser(description='Run BOP Challenge pose prediction')

    parser.add_argument('--descriptor', choices=['dinov2', 'dinov3'], default='dinov2')
    parser.add_argument('--templates_source', choices=['viewgraph', 'cnns', 'prerendered'], default='cnns')
    parser.add_argument('--condensation_source', default='1nn-hart')
    parser.add_argument('--whitening_dim', type=int, default=0)
    parser.add_argument('--detector', default='sam')
    parser.add_argument('--aggregation_function', default=None)
    parser.add_argument('--confidence_thresh', type=float, default=None)
    parser.add_argument('--experiment_folder', default='default')
    parser.add_argument('--use_enhanced_nms', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--descriptor_mask_detections', type=lambda x: bool(int(x)), default=True)
    parser.add_argument('--similarity_metric', default='cosine')
    parser.add_argument('--ood_detection_method', default=None)
    parser.add_argument('--cosine_similarity_quantile', type=float, default=None)
    parser.add_argument('--mahalanobis_quantile', type=float, default=None)
    parser.add_argument('--lowe_ratio_threshold', type=float, default=None)
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    bop_base = Path('/mnt/personal/jelint19/data/bop')

    sequences_to_run = [
        (
            'hot3d', 'test', 'object_ref_aria_static_scenewise', None
        ),
        (
            'hot3d', 'test', 'object_ref_aria_quest3_scenewise', None
        ),
        (
            'hope', 'test', 'onboarding_static',
            bop_base / 'default_detections/h3_bop24_model_free_unseen/cnos-sam/onboarding_static/'
                       'cnos-sam_hope-test_static-020a-45bd-8ec5-c95560b68011.json'
        ),
        ('hope', 'val', 'onboarding_static', None),
        (
            'tless', 'test', 'train_primesense',
            bop_base / 'default_detections/classic_bop23_model_based_unseen/cnos-fastsam/'
                       'cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json'
        ),
        (
            'lmo', 'test', 'train',
            bop_base / 'default_detections/classic_bop23_model_based_unseen/cnos-fastsam/'
                       'cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json'
        ),
        (
            'icbin', 'test', 'train',
            bop_base / 'default_detections/classic_bop23_model_based_unseen/cnos-fastsam/'
                       'cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json'
        ),
        ('handal', 'test', 'onboarding_static', None),
        ('handal', 'val', 'onboarding_static', None),
    ]

    method_name = 'MyCNOS'
    cache_path = Path('/mnt/personal/jelint19/cache')

    for dataset, split, detections_split, default_detections_file in sequences_to_run:
        data_type = ''

        if dataset in ['hope', 'handal']:
            targets_year = 'bop24'
        elif dataset in 'hot3d':
            targets_year = 'bop24'
            if 'aria' in detections_split:
                data_type = '_aria_scenewise'
            elif 'quest3' in detections_split:
                data_type = '_quest3_scenewise'
            else:
                raise ValueError(f'Detections split may only be "aria" or "quest3", but is {detections_split}')
        elif dataset in ['tless', 'lmo', 'icbin']:
            targets_year = 'bop19'
            if dataset == 'tless':
                data_type = '_primesense'
            assert split != 'val'
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        split_folder = f'{split}{data_type}'
        base_dataset_folder = bop_base / dataset
        bop_targets_path = base_dataset_folder / f'{split}_targets_{targets_year}.json'

        # Set up paths based on templates_source
        if args.templates_source == 'viewgraph':
            config_name = 'ufm_c0975r05'
            view_graph_location = cache_path / 'view_graph_cache' / config_name / dataset
            condensed_templates_base = None
            experiment = f'viewgraph-templates-{args.descriptor}'
        elif args.templates_source == 'cnns':
            view_graph_location = None
            if not args.condensation_source:
                parser.error("--condensation_source is required when --templates_source is 'cnns'")

            whitening_suffix = f'-whitening_{args.whitening_dim}' if args.whitening_dim > 0 else ''
            condensation_source = f"{args.condensation_source}-{args.descriptor}{whitening_suffix}"
            condensed_templates_base = (cache_path / 'detections_templates_cache' / condensation_source /
                                        dataset / detections_split)
            experiment = f'cnns@{condensation_source}'
        else:  # pre-rendered
            view_graph_location = None
            condensed_templates_base = None
            experiment = f'onboarding-templates@{args.descriptor}'

        config = TrackerConfig()
        config.device = 'cuda'

        matching_config_overrides = {
            'aggregation_function': args.aggregation_function,
            'ood_detection_method': args.ood_detection_method,
            'confidence_thresh': args.confidence_thresh,
            'cosine_similarity_quantile': args.cosine_similarity_quantile,
            'mahalanobis_quantile': args.mahalanobis_quantile,
            'lowe_ratio_threshold': args.lowe_ratio_threshold,
            'similarity_metric': args.similarity_metric,
        }
        predictor = BOPChallengePosePredictor(config, cache_path, matching_config_overrides, args.experiment_folder)
        match_cfg = predictor.cnos_matching_config
        experiment = (f'{experiment}@mask_{args.descriptor_mask_detections}@aggr_{match_cfg.aggregation_function}@'
                      f'sim_{match_cfg.similarity_metric}@detector_{args.detector}@nms{args.use_enhanced_nms}@OOD_')

        if args.ood_detection_method == 'global_threshold':
            experiment += f'conf_{match_cfg.confidence_thresh}'
        elif args.ood_detection_method == 'lowe_test':
            experiment += f'lowe_{match_cfg.lowe_ratio_threshold}'
        elif args.ood_detection_method == 'cosine_similarity_quantiles':
            experiment += f'cosQuantiles_{match_cfg.cosine_similarity_quantile}'
        elif args.ood_detection_method == 'mahalanobis_ood_detection':
            experiment += f'mahaTau_{match_cfg.mahalanobis_quantile}'
        elif args.ood_detection_method == 'none':
            experiment += f'none'

        predictor.predict_poses_for_bop_challenge(base_dataset_folder, bop_targets_path, condensed_templates_base,
                                                  detections_split, split_folder, method_name, experiment,
                                                  view_graph_location, descriptor=args.descriptor,
                                                  detector_name=args.detector,
                                                  descriptor_mask_detections=args.descriptor_mask_detections,
                                                  default_detections_file=default_detections_file,
                                                  templates_source=args.templates_source,
                                                  dry_run=args.dry_run)


if __name__ == '__main__':
    main()

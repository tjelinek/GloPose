"""Standalone pose estimation runner using default CNOS detections and cached ViewGraphs.

Usage:
    python run_pose_estimation.py --dataset hope --split test \
        --default_detections /home/tom/rci_data/data/bop/default_detections/h3_bop24_model_free_unseen/cnos-sam/onboarding_static/cnos-sam_hope-test_static-020a-45bd-8ec5-c95560b68011.json \
        --view_graph_config ufm_c0975r05 \
        --device cpu --max_images 2
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from configs.glopose_config import GloPoseConfig, PoseEstimationConfig, OnboardingConfig
from data_providers.flow_provider import create_matching_provider
from data_providers.frame_provider import PrecomputedFrameProvider
from data_structures.types import Detection
from data_structures.view_graph import load_view_graphs_by_object_id
from pose_estimation.estimator import PoseEstimator
from utils.bop_data import (get_scene_folder, get_image_path, get_camera_json_path,
                             load_camera_intrinsics)
from utils.bop_io import pose_to_bop_record, write_bop_pose_csv
from utils.cnos_utils import get_default_detections_per_scene_and_image
from utils.mask_utils import rle_to_mask


# Path remapping: ViewGraph pickles store RCI paths; remap to local sshfs/symlink paths.
RCI_PATH_REMAP = {
    '/mnt/personal/jelint19/': '/home/tom/rci_data/',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Run pose estimation with default CNOS detections')

    parser.add_argument('--dataset', required=True, help='BOP dataset name (e.g. hope, handal, tless, lmo)')
    parser.add_argument('--split', default='test', help='BOP split (test, val)')
    parser.add_argument('--onboarding_type', default='onboarding_static',
                        help='Onboarding type for ViewGraph filtering (onboarding_static, onboarding_dynamic)')
    parser.add_argument('--default_detections', required=True, type=Path,
                        help='Path to default CNOS detections JSON file')
    parser.add_argument('--view_graph_config', default='ufm_c0975r05',
                        help='ViewGraph cache config name')
    parser.add_argument('--bop_data_folder', default='/mnt/data/vrg/public_datasets/bop/',
                        type=Path, help='Root BOP data folder')
    parser.add_argument('--cache_folder', default='/home/tom/rci_data/cache/',
                        type=Path, help='Cache folder root')
    parser.add_argument('--output_folder', default=None, type=Path,
                        help='Output folder (default: results/pose_estimation/)')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--max_images', type=int, default=None,
                        help='Limit number of test images (for debugging)')
    parser.add_argument('--object_ids', type=int, nargs='+', default=None,
                        help='Only load ViewGraphs for these object IDs (1-indexed, e.g. 1 2 3)')
    parser.add_argument('--max_detections_per_image', type=int, default=10,
                        help='Max detections to process per image')
    parser.add_argument('--min_detection_score', type=float, default=0.3,
                        help='Minimum detection score to consider')
    parser.add_argument('--image_downsample', type=float, default=0.25,
                        help='Image downsample factor (0.25 for hope/handal)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only load data, do not run pose estimation')
    parser.add_argument('--flow_reliability_threshold', type=float, default=0.15,
                        help='Min flow reliability for SIFT matches (lower than dense methods)')
    parser.add_argument('--min_certainty_threshold', type=float, default=0.5,
                        help='Min match certainty threshold')
    parser.add_argument('--max_templates', type=int, default=10,
                        help='Max ViewGraph templates to match against')

    return parser.parse_args()


def detections_for_image(detections_data: list[dict], device: str,
                          min_score: float, max_detections: int) -> list[Detection]:
    """Convert raw CNOS detection dicts to Detection objects.

    category_id in CNOS JSON is 1-indexed and matches ViewGraph object_id directly.
    """
    # Filter by score
    filtered = [d for d in detections_data if d['score'] >= min_score]

    # Sort by score descending, take top N
    filtered.sort(key=lambda d: d['score'], reverse=True)
    filtered = filtered[:max_detections]

    detections = []
    for det in filtered:
        mask = torch.tensor(rle_to_mask(det['segmentation']), dtype=torch.float32, device=device)
        bbox = det['bbox']  # xywh format

        detections.append(Detection(
            object_id=det['category_id'],  # 1-indexed, matches ViewGraph object_id
            score=det['score'],
            bbox_xywh=bbox,
            mask=mask,
        ))

    return detections


def main():
    args = parse_args()

    # Resolve paths
    bop_dataset_folder = args.bop_data_folder / args.dataset
    test_dataset_path = bop_dataset_folder / args.split
    view_graph_cache_dir = args.cache_folder / 'view_graph_cache' / args.view_graph_config / args.dataset

    output_folder = args.output_folder or Path('results/pose_estimation')
    output_folder.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}/{args.split}")
    print(f"BOP data: {bop_dataset_folder}")
    print(f"ViewGraph cache: {view_graph_cache_dir}")
    print(f"Detections: {args.default_detections}")
    print(f"Device: {args.device}")

    # --- Load default detections ---
    print("\nLoading default detections...")
    detections_dict = get_default_detections_per_scene_and_image(args.default_detections)
    unique_images = sorted(detections_dict.keys())
    print(f"  Loaded {sum(len(v) for v in detections_dict.values())} detections "
          f"across {len(unique_images)} images")

    if args.max_images:
        unique_images = unique_images[:args.max_images]
        print(f"  Limited to {len(unique_images)} images (--max_images)")

    # --- Load ViewGraphs ---
    print("\nLoading ViewGraphs...")
    if not view_graph_cache_dir.exists():
        raise FileNotFoundError(f"ViewGraph cache not found: {view_graph_cache_dir}")

    if args.object_ids:
        # Load only specific ViewGraphs by object_id (much faster for debugging)
        from data_structures.view_graph import load_view_graph
        view_graphs = {}
        suffix_map = {'onboarding_static': '_both', 'onboarding_dynamic': '_dynamic'}
        suffix = suffix_map.get(args.onboarding_type, '_both')
        for obj_id in args.object_ids:
            vg_dir = view_graph_cache_dir / f'obj_{obj_id:06d}{suffix}'
            if vg_dir.exists():
                vg = load_view_graph(vg_dir, device=args.device, remap_paths=RCI_PATH_REMAP)
                view_graphs[vg.object_id] = vg
                print(f"  Loaded ViewGraph for object {obj_id}")
            else:
                print(f"  ViewGraph not found: {vg_dir}")
    else:
        view_graphs = load_view_graphs_by_object_id(
            view_graph_cache_dir, args.onboarding_type, args.device,
            remap_paths=RCI_PATH_REMAP
        )
    print(f"  Loaded {len(view_graphs)} ViewGraphs: {sorted(view_graphs.keys())}")

    successful_vgs = {k: v for k, v in view_graphs.items()
                      if getattr(v, 'reconstruction_success', True)}
    print(f"  {len(successful_vgs)}/{len(view_graphs)} have successful reconstructions")

    if args.dry_run:
        print("\n--- DRY RUN: skipping pose estimation ---")
        # Print a summary of what would be processed
        for im_id, scene_id in unique_images[:5]:
            dets = detections_dict[(im_id, scene_id)]
            cats = [d['category_id'] for d in dets]
            matched = [c for c in cats if c in successful_vgs]
            print(f"  Scene {scene_id}, Image {im_id}: "
                  f"{len(dets)} detections, categories {sorted(set(cats))}, "
                  f"{len(matched)} matchable to ViewGraphs")
        return

    # --- Create matching provider and pose estimator ---
    print("\nInitializing SIFT+LightGlue matching provider...")
    onboarding_config = OnboardingConfig(filter_matcher='SIFT')
    matching_provider = create_matching_provider('SIFT', onboarding_config, args.device)

    pose_config = PoseEstimationConfig(
        matcher='SIFT',
        sample_size=10000,
        min_certainty_threshold=args.min_certainty_threshold,
        flow_reliability_threshold=args.flow_reliability_threshold,
        black_background=True,
        max_templates_to_match=args.max_templates,
    )

    pose_cache = output_folder / 'cache'
    pose_cache.mkdir(parents=True, exist_ok=True)

    pose_estimator = PoseEstimator(
        matching_provider=matching_provider,
        config=pose_config,
        cache_folder=pose_cache,
        device=args.device,
    )

    # --- Run pose estimation ---
    print(f"\nRunning pose estimation on {len(unique_images)} images...")
    bop_pose_results = []
    stats = defaultdict(int)

    for im_id, scene_id in tqdm(unique_images, desc="Pose estimation"):
        raw_dets = detections_dict[(im_id, scene_id)]

        # Load image
        scene_folder = get_scene_folder(test_dataset_path, scene_id)
        try:
            image_path = get_image_path(scene_folder, im_id, args.dataset)
        except FileNotFoundError:
            stats['images_not_found'] += 1
            continue

        image = PrecomputedFrameProvider.load_and_downsample_image(
            image_path, args.image_downsample, args.device
        ).squeeze()

        # Load camera intrinsics (scaled by downsample factor)
        camera_json = get_camera_json_path(scene_folder)
        camera_K = load_camera_intrinsics(camera_json, im_id)
        camera_K[0, :] *= args.image_downsample
        camera_K[1, :] *= args.image_downsample

        # Convert detections
        image_detections = detections_for_image(
            raw_dets, args.device,
            min_score=args.min_detection_score,
            max_detections=args.max_detections_per_image
        )

        # Downsample detection masks to match image
        if args.image_downsample != 1.0:
            h, w = image.shape[-2:]
            for det in image_detections:
                if det.mask is not None:
                    det.mask = torch.nn.functional.interpolate(
                        det.mask[None, None].float(), size=(h, w), mode='nearest'
                    ).squeeze().bool().float()

        stats['total_detections'] += len(image_detections)

        # Filter to detections with available ViewGraphs
        matchable = [d for d in image_detections if d.object_id in successful_vgs]
        stats['matchable_detections'] += len(matchable)

        if not matchable:
            continue

        # Run pose estimation
        pose_start = time.time()
        try:
            pose_estimates = pose_estimator.estimate_poses(
                matchable, successful_vgs, image, camera_K, pose_logger=None
            )
        except Exception as e:
            import traceback as tb
            print(f"\n  Error on scene {scene_id}, image {im_id}: {e}")
            tb.print_exc()
            stats['errors'] += 1
            continue
        pose_duration = time.time() - pose_start

        stats['successful_poses'] += len(pose_estimates)
        stats['images_processed'] += 1

        for pose_est in pose_estimates:
            bop_pose_results.append(
                pose_to_bop_record(pose_est, scene_id, im_id, pose_duration)
            )

    # --- Write results ---
    print(f"\n--- Results ---")
    print(f"  Images processed: {stats['images_processed']}")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Matchable (have ViewGraph): {stats['matchable_detections']}")
    print(f"  Successful poses: {stats['successful_poses']}")
    print(f"  Errors: {stats['errors']}")

    if bop_pose_results:
        csv_path = output_folder / f'poses_{args.dataset}-{args.split}_{args.view_graph_config}.csv'
        write_bop_pose_csv(bop_pose_results, csv_path)
        print(f"\n  Pose results written to: {csv_path}")
    else:
        print("\n  No poses estimated.")


if __name__ == '__main__':
    main()

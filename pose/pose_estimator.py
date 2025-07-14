import json
from pathlib import Path
from typing import List

import numpy as np

from data_providers.flow_provider import RoMaFlowProviderDirect
from data_providers.frame_provider import PrecomputedFrameProvider, PrecomputedSegmentationProvider

from data_structures.view_graph import ViewGraph, load_view_graph
from pose.glomap import predict_poses
from tracker_config import TrackerConfig
from utils.bop_challenge import get_gop_camera_intrinsics
from utils.image_utils import get_target_shape


def predict_poses_for_bop_challenge(bop_targets_path: Path, view_graph_save_paths: Path, config: TrackerConfig) -> None:
    with bop_targets_path.open('r') as file:
        test_annotations = json.load(file)

    test_dataset_path = bop_targets_path.parent.parent / 'test'

    for item in test_annotations:
        im_id = item['im_id']
        scene_id = item['scene_id']

        scene_folder_name = f'{scene_id:06d}'
        image_id_str = f'{im_id:06d}'
        image_filename = f'{image_id_str}.png'

        path_to_scene = test_dataset_path / scene_folder_name
        path_to_image = path_to_scene / 'rgb' / image_filename
        if not path_to_image.exists():
            image_filename = f'{image_id_str}.jpg'
            path_to_image = path_to_scene / 'rgb' / image_filename
            assert path_to_image.exists()

        path_to_camera_intrinsics = path_to_scene / 'scene_camera.json'
        segmentation_paths = path_to_scene / 'mask_visib'

        segmentation_files = sorted(segmentation_paths.glob(f"{image_id_str}_*.png"))
        camera_intrinsics = get_gop_camera_intrinsics(path_to_camera_intrinsics, im_id)

        view_graphs: List[ViewGraph] = []
        for view_graph_dir in view_graph_save_paths.iterdir():
            if view_graph_dir.is_dir():
                view_graph = load_view_graph(view_graph_dir, device=config.device)
                view_graphs.append(view_graph)

        predict_all_poses_in_image(path_to_image, segmentation_files, camera_intrinsics, view_graphs, config)


def predict_all_poses_in_image(image_path: Path, segmentation_paths: List[Path], camera_K: np.ndarray,
                               view_graphs: List[ViewGraph],
                               config: TrackerConfig) -> None:
    target_shape = get_target_shape(image_path, config.image_downsample)
    image = PrecomputedFrameProvider.load_and_downsample_image(image_path, config.image_downsample, config.device)
    image = image.squeeze()

    config.device = 'cuda'
    global FLOW_PROVIDER_GLOBAL

    if FLOW_PROVIDER_GLOBAL is None:
        FLOW_PROVIDER_GLOBAL = RoMaFlowProviderDirect(config.device, config.roma_config)

    for segmentation_paths in segmentation_paths:
        segmentation = PrecomputedSegmentationProvider.load_and_downsample_segmentation(segmentation_paths,
                                                                                        target_shape,
                                                                                        config.device)
        segmentation = segmentation.squeeze()

        # TODO iterate over all view graphs
        predict_poses(image, segmentation, camera_K=camera_K, view_graph=view_graphs[0],
                      flow_provider=FLOW_PROVIDER_GLOBAL, config=config)


if __name__ == '__main__':
    _bop_targets_path = Path('/mnt/personal/jelint19/data/bop/handal/handal_base/test_targets_bop24.json')
    _view_graph_location = Path('/mnt/personal/jelint19/cache/view_graph_cache/handal')

    _config = TrackerConfig()
    predict_poses_for_bop_challenge(_bop_targets_path, _view_graph_location, _config)

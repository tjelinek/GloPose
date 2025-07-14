import json
from pathlib import Path
from typing import List, Optional

import numpy as np

from data_providers.flow_provider import RoMaFlowProviderDirect, UFMFlowProviderDirect, FlowProviderDirect
from data_providers.frame_provider import PrecomputedFrameProvider, PrecomputedSegmentationProvider

from data_structures.view_graph import ViewGraph, load_view_graph
from pose.glomap import predict_poses
from tracker_config import TrackerConfig
from utils.bop_challenge import get_gop_camera_intrinsics
from utils.image_utils import get_target_shape


class BOPChallengePosePredictor:

    def __init__(self, config: TrackerConfig, view_graph_save_paths: Path, onboarding_type: str):

        self.config = config
        self.flow_provider: Optional[FlowProviderDirect] = None
        self.view_graphs: List[ViewGraph] = []

        self._initialize_flow_provider()
        self.load_view_graphs(view_graph_save_paths, onboarding_type)
        self.onboarding_type: str = onboarding_type

    def _initialize_flow_provider(self) -> None:

        if self.config.dense_matching == 'RoMa':
            self.flow_provider = RoMaFlowProviderDirect(self.config.device, self.config.roma_config)
        elif self.config.dense_matching == 'UFM':
            self.flow_provider = UFMFlowProviderDirect(self.config.device, self.config.ufm_config)
        else:
            raise ValueError(f'Unknown dense matching option {self.config.dense_matching}')

    def load_view_graphs(self, view_graph_save_paths: Path, onboarding_type: str) -> None:
        self.view_graphs = []
        for view_graph_dir in view_graph_save_paths.iterdir():
            if view_graph_dir.is_dir():
                if onboarding_type == 'static':
                    if not view_graph_dir.stem.endswith('_up') and not view_graph_dir.stem.endswith('_up'):
                        continue
                elif onboarding_type == 'dynamic':
                    if not view_graph_dir.stem.endswith('_dynamic'):
                        continue
                else:
                    raise ValueError(f"Unknown onboarding type {onboarding_type}")

                view_graph = load_view_graph(view_graph_dir, device=self.config.device)
                self.view_graphs.append(view_graph)

    def predict_poses_for_bop_challenge(self, base_dataset_folder: Path, bop_targets_path: Path, split: str) -> None:

        with bop_targets_path.open('r') as file:
            test_annotations = json.load(file)

        test_dataset_path = base_dataset_folder / split

        for item in test_annotations:
            im_id = item['im_id']
            scene_id = item['scene_id']

            # Construct paths
            scene_folder_name = f'{scene_id:06d}'
            image_id_str = f'{im_id:06d}'
            path_to_scene = test_dataset_path / scene_folder_name
            path_to_image = self._get_image_path(path_to_scene, image_id_str)
            path_to_camera_intrinsics = path_to_scene / 'scene_camera.json'
            segmentation_paths = path_to_scene / 'mask_visib'

            # Get segmentation files and camera intrinsics
            segmentation_files = sorted(segmentation_paths.glob(f"{image_id_str}_*.png"))
            camera_intrinsics = get_gop_camera_intrinsics(path_to_camera_intrinsics, im_id)
            # Predict poses for this image
            self.predict_all_poses_in_image(path_to_image, segmentation_files, camera_intrinsics)

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

    def predict_all_poses_in_image(self, image_path: Path, segmentation_paths: List[Path],
                                   camera_K: np.ndarray) -> None:

        # Load and preprocess image
        target_shape = get_target_shape(image_path, self.config.image_downsample)
        image = PrecomputedFrameProvider.load_and_downsample_image(
            image_path, self.config.image_downsample, self.config.device
        )
        image = image.squeeze()

        for segmentation_path in segmentation_paths:
            segmentation = PrecomputedSegmentationProvider.load_and_downsample_segmentation(
                segmentation_path, target_shape, self.config.device
            )
            segmentation = segmentation.squeeze()
            if self.view_graphs:
                predict_poses(
                    image,
                    segmentation,
                    camera_K=camera_K,
                    view_graph=self.view_graphs[0],
                    flow_provider=self.flow_provider,
                    config=self.config
                )


def main():
    """Example usage of the BOPChallengePosePredictor class."""

    dataset = 'hope'
    onboarding_type = 'static'

    base_dataset_folder = Path(f'/mnt/personal/jelint19/data/bop/{dataset}')
    bop_targets_path = base_dataset_folder / 'test_targets_bop24.json'
    view_graph_location = Path(f'/mnt/personal/jelint19/cache/view_graph_cache/default/{dataset}')

    default_detections_dir = (base_dataset_folder / 'h3_bop24_model_free_unseen' / 'cnos-sam' /
                              f'onboarding_{onboarding_type}')
    default_detections_file = list(default_detections_dir.glob(f"*{dataset}*.json"))[0]

    config = TrackerConfig()
    config.device = 'cpu'
    predictor = BOPChallengePosePredictor(config, view_graph_location, onboarding_type)

    predictor.predict_poses_for_bop_challenge(base_dataset_folder, bop_targets_path, 'test',
                                              default_detections_file)


if __name__ == '__main__':
    main()
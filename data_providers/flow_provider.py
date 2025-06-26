import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Tuple, Optional

import torch
import torchvision
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model

from configs.matching_configs.roma_configs.base_roma_config import BaseRomaConfig
from data_structures.data_graph import DataGraph
from flow import roma_warp_to_pixel_coordinates


class FlowProviderDirect(ABC):

    @abstractmethod
    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None, zero_certainty_outside_segmentation: bool = False):
        pass

    @staticmethod
    def keypoints_to_int(src_pts_xy_roma, dst_pts_xy_roma, source_image_tensor, target_image_tensor):
        h1 = source_image_tensor.shape[-2]
        w1 = source_image_tensor.shape[-1]
        h2 = target_image_tensor.shape[-2]
        w2 = target_image_tensor.shape[-1]
        src_pts_xy_roma = src_pts_xy_roma.to(torch.int)
        dst_pts_xy_roma = dst_pts_xy_roma.to(torch.int)
        src_pts_xy_roma[:, 0] = torch.clamp(src_pts_xy_roma[:, 0], 0, w1 - 1)
        src_pts_xy_roma[:, 1] = torch.clamp(src_pts_xy_roma[:, 1], 0, h1 - 1)
        dst_pts_xy_roma[:, 0] = torch.clamp(dst_pts_xy_roma[:, 0], 0, w2 - 1)
        dst_pts_xy_roma[:, 1] = torch.clamp(dst_pts_xy_roma[:, 1], 0, h2 - 1)
        return src_pts_xy_roma, dst_pts_xy_roma


class PrecomputedFlowProviderDirect(ABC):

    pass


class RoMaFlowProviderDirect(FlowProviderDirect):

    def __init__(self, device, roma_config: BaseRomaConfig):
        self.device = device

        if roma_config.use_custom_weights:
            weights = torch.load('/mnt/personal/jelint19/weights/RoMa/checkpointstrain_roma_outdoor_latest.pth',
                                 map_location=device, weights_only=True)
            if "model" in weights:
                weights = weights["model"]
        else:
            weights = torch.load('/mnt/personal/jelint19/weights/RoMa/roma_outdoor.pth',
                                 map_location=device, weights_only=True)

        self.flow_model: roma_model = roma_outdoor(device=self.device, weights=weights)
        self.flow_model.sample_mode = 'balanced'  # This ensures that the matches are sampled ~ certainties
        self.roma_size_hw = (864, 864)

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None, zero_certainty_outside_segmentation: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image_tensor.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image_tensor.squeeze())

        warp, certainty = self.flow_model.match(source_image_roma, target_image_roma, device=self.device)

        if zero_certainty_outside_segmentation:
            certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                                 target_image_segmentation)

        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def zero_certainty_outside_segmentation(self, certainty: torch.Tensor,
                                            source_image_segmentation: torch.Tensor = None,
                                            target_image_segmentation: torch.Tensor = None) -> \
            torch.Tensor:
        roma_h, roma_w = self.roma_size_hw
        assert source_image_segmentation is not None or target_image_segmentation is not None

        certainty = certainty.clone()
        if source_image_segmentation is not None:
            source_image_segment_roma_size = torchvision.transforms.functional.resize(source_image_segmentation[None],
                                                                                      size=self.roma_size_hw)
            certainty[:, :roma_w] *= source_image_segment_roma_size.squeeze().bool().float()
        if target_image_segmentation is not None:
            target_image_segment_roma_size = torchvision.transforms.functional.resize(target_image_segmentation[None],
                                                                                      size=self.roma_size_hw)
            certainty[:, roma_w:2 * roma_w] *= target_image_segment_roma_size.squeeze().bool().float()
        return certainty

    def get_source_target_points(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        warp, certainty = self.compute_flow(source_image_tensor, target_image_tensor, sample,
                                            source_image_segmentation, target_image_segmentation, source_image_name,
                                            target_image_name, source_image_index, target_image_index,
                                            zero_certainty_outside_segmentation)

        h1 = source_image_tensor.shape[-2]
        w1 = source_image_tensor.shape[-1]
        h2 = target_image_tensor.shape[-2]
        w2 = target_image_tensor.shape[-1]
        src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

        if len(src_pts_xy_roma.shape) == 3:
            src_pts_xy_roma = src_pts_xy_roma.flatten(0, 1)
        if len(dst_pts_xy_roma.shape) == 3:
            dst_pts_xy_roma = dst_pts_xy_roma.flatten(0, 1)

        src_pts_xy_roma_int, dst_pts_xy_roma_int = self.keypoints_to_int(src_pts_xy_roma, dst_pts_xy_roma,
                                                                         source_image_tensor, target_image_tensor)
        if as_int:
            src_pts_xy_roma, dst_pts_xy_roma = src_pts_xy_roma_int, dst_pts_xy_roma_int

        if only_foreground_matches:
            assert source_image_segmentation is not None and target_image_segmentation is not None
            assert len(source_image_segmentation.shape) == 2 and len(target_image_segmentation.shape) == 2
            in_segment_mask_src = source_image_segmentation[src_pts_xy_roma_int[:, 1], src_pts_xy_roma_int[:, 0]].bool()
            in_segment_mask_tgt = target_image_segmentation[dst_pts_xy_roma_int[:, 1], dst_pts_xy_roma_int[:, 0]].bool()

            fg_matches = in_segment_mask_src * in_segment_mask_tgt

            src_pts_xy_roma = src_pts_xy_roma[fg_matches]
            dst_pts_xy_roma = dst_pts_xy_roma[fg_matches]
            certainty = certainty[fg_matches]

        return src_pts_xy_roma, dst_pts_xy_roma, certainty


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect):

    def __init__(self, device, roma_config: BaseRomaConfig, cache_dir: Path, data_graph: DataGraph = None, allow_missing: bool = True,
                 allow_disk_cache=True, purge_cache: bool = False):
        super().__init__(device, roma_config)

        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'
        self.data_graph: Optional[DataGraph] = data_graph

        if purge_cache and self.warps_path.exists():
            shutil.rmtree(self.warps_path)
        if purge_cache and self.certainties_path.exists():
            shutil.rmtree(self.certainties_path)

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.allow_missing: bool = allow_missing
        self.allow_disk_cache: bool = allow_disk_cache

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None,
                     zero_certainty_outside_segmentation=False) -> Tuple[torch.Tensor, torch.Tensor]:

        if source_image_name is not None and target_image_name is not None:
            saved_filename = f'{source_image_name.stem}___{target_image_name.stem}.pt'
            warp_filename = self.warps_path / saved_filename
            certainty_filename = self.certainties_path / saved_filename
        elif (source_image_index is not None and target_image_index is not None and self.data_graph is not None and
              self.data_graph.G.has_node(source_image_index) and self.data_graph.G.has_node(source_image_index)):
            source_data = self.data_graph.get_frame_data(source_image_index)
            target_data = self.data_graph.get_frame_data(target_image_index)

            saved_filename = f'{source_data.image_filename.stem}___{target_data.image_filename.stem}.pt'
            warp_filename = self.warps_path / saved_filename
            certainty_filename = self.certainties_path / saved_filename
        else:
            warp_filename = None
            certainty_filename = None

        warp, certainty = None, None
        if self._datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
            if edge_data.roma_flow_warp is not None and edge_data.roma_flow_warp_certainty is not None:
                warp, certainty = edge_data.roma_flow_warp, edge_data.roma_flow_warp_certainty

        if (warp is None or certainty is None) and warp_filename is not None and certainty_filename is not None:
            if warp_filename.exists() and certainty_filename.exists():
                warp = torch.load(warp_filename, weights_only=True, map_location=self.device)
                certainty = torch.load(certainty_filename, weights_only=True, map_location=self.device)

        if warp is None or certainty is None:
            warp, certainty = super().compute_flow(source_image_tensor, target_image_tensor)

            if source_image_name and target_image_name and self.allow_missing and self.allow_disk_cache:
                torch.save(warp, warp_filename)
                torch.save(certainty, certainty_filename)

        if zero_certainty_outside_segmentation:
            certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                                 target_image_segmentation)

        if self.data_graph is not None:
            if source_image_index is not None and target_image_index is not None:
                if not self.data_graph.G.has_edge(source_image_index, target_image_index):
                    self.data_graph.add_new_arc(source_image_index, target_image_index)

                edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
                if edge_data.roma_flow_warp is None:
                    edge_data.roma_flow_warp = warp
                if edge_data.roma_flow_warp_certainty is None:
                    edge_data.roma_flow_warp_certainty = certainty

        if sample:
            if ((source_image_segmentation is not None and source_image_segmentation.sum() <= 5) or
                    (target_image_segmentation is not None and target_image_segmentation.sum() <= 5) and
                    zero_certainty_outside_segmentation):  # This should prevent the sampling to fail due to CUDA error
                warp = torch.zeros(0, 4).to(warp.device).to(warp.dtype)
                certainty = torch.zeros(0, ).to(certainty.device).to(certainty.dtype)
            else:
                warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def get_source_target_points(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor,
                                 sample: int = None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation=False, only_foreground_matches=False) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_pts_xy_roma = None
        dst_pts_xy_roma = None
        certainty = None

        if self._datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)

            if edge_data.src_pts_xy_roma is not None and edge_data.dst_pts_xy_roma is not None:
                src_pts_xy_roma, dst_pts_xy_roma = edge_data.src_pts_xy_roma, edge_data.dst_pts_xy_roma

        if src_pts_xy_roma is None or dst_pts_xy_roma is None or certainty is None:
            src_pts_xy_roma, dst_pts_xy_roma, certainty = (
                super().get_source_target_points(source_image_tensor, target_image_tensor, sample,
                                                 source_image_segmentation, target_image_segmentation,
                                                 source_image_name, target_image_name, source_image_index,
                                                 target_image_index, as_int, zero_certainty_outside_segmentation,
                                                 only_foreground_matches))

        if (self.data_graph is not None and source_image_index is not None and target_image_index is not None and
                not self.data_graph.G.has_edge(source_image_index, target_image_index)):
            self.data_graph.add_new_arc(source_image_index, target_image_index)

        if self._datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)

            if edge_data.src_pts_xy_roma is None:
                edge_data.src_pts_xy_roma = src_pts_xy_roma
            if edge_data.dst_pts_xy_roma is None:
                edge_data.dst_pts_xy_roma = dst_pts_xy_roma
            if edge_data.src_dst_certainty_roma is None:
                edge_data.src_dst_certainty_roma = certainty

        if as_int:
            src_pts_xy_roma, dst_pts_xy_roma = self.keypoints_to_int(src_pts_xy_roma, dst_pts_xy_roma,
                                                                     source_image_tensor, target_image_tensor)

        return src_pts_xy_roma, dst_pts_xy_roma, certainty

    def get_source_target_points_datagraph(self, source_image_index: int, target_image_index: int,
                                           sample: int = None, as_int: bool = False,
                                           zero_certainty_outside_segmentation: bool = False,
                                           only_foreground_matches: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_data = self.data_graph.get_frame_data(source_image_index)
        target_data = self.data_graph.get_frame_data(target_image_index)

        return self.get_source_target_points(source_data.frame_observation.observed_image.squeeze(),
                                             target_data.frame_observation.observed_image.squeeze(), sample,
                                             source_data.frame_observation.observed_segmentation.squeeze(),
                                             target_data.frame_observation.observed_segmentation.squeeze(),
                                             source_data.image_filename, target_data.image_filename,
                                             source_image_index, target_image_index, as_int,
                                             zero_certainty_outside_segmentation, only_foreground_matches)

    def _datagraph_edge_exists(self, source_image_index, target_image_index):
        return (source_image_index is not None and target_image_index is not None and self.data_graph is not None and
                self.data_graph.G.has_edge(source_image_index, target_image_index))

    def cached_flow_from_filenames(self, src_image_name: Union[str, Path], target_image_name: Union[str, Path]):

        src_image_name = Path(src_image_name)
        target_image_name = Path(target_image_name)
        saved_filename = f'{src_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if warp_filename.exists() and certainty_filename.exists():
            warp = torch.load(warp_filename, weights_only=True).to(self.device)
            certainty = torch.load(certainty_filename, weights_only=True).to(self.device)
            return warp, certainty

        return None

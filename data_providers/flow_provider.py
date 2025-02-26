from pathlib import Path
from typing import Union, Tuple

import torch
import torchvision
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model

from data_structures.data_graph import DataGraph
from flow import roma_warp_to_pixel_coordinates


class RoMaFlowProviderDirect:

    def __init__(self, device):
        self.device = device
        self.flow_model: roma_model = None

    def _init_flow_model(self):
        self.flow_model: roma_model = roma_outdoor(device=self.device)

    def next_flow_roma(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None)\
            -> Tuple[torch.Tensor, torch.Tensor]:
        if self.flow_model is None:
            self._init_flow_model()

        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image_tensor.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image_tensor.squeeze())

        warp, certainty = self.flow_model.match(source_image_roma, target_image_roma, device=self.device)
        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def get_source_target_points_roma(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor,
                                      sample=None) -> Tuple[torch.Tensor, torch.Tensor]:
        warp, certainty = self.next_flow_roma(source_image_tensor, target_image_tensor, sample)

        h1 = source_image_tensor.shape[-2]
        w1 = source_image_tensor.shape[-1]
        h2 = target_image_tensor.shape[-2]
        w2 = target_image_tensor.shape[-1]

        src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

        return src_pts_xy_roma, dst_pts_xy_roma


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect):

    def __init__(self, data_graph: DataGraph, device, cache_dir: Path, allow_missing: bool = True):
        super().__init__(device)

        self.data_graph = data_graph
        self.saved_flow_paths = cache_dir
        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.allow_missing: bool = allow_missing

    def next_cache_flow_roma(self, source_image_idx: int, target_image_idx: int, sample=None):

        src_image_name = self.data_graph.get_frame_data(source_image_idx).image_filename
        target_image_name = self.data_graph.get_frame_data(target_image_idx).image_filename
        saved_filename = f'{src_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if (not warp_filename.exists() or not certainty_filename.exists()) and self.allow_missing:
            source_image_tensor = self.data_graph.get_frame_data(source_image_idx).frame_observation.observed_image
            target_image_tensor = self.data_graph.get_frame_data(target_image_idx).frame_observation.observed_image
            warp, certainty = super().next_flow_roma(source_image_tensor, target_image_tensor)

            torch.save(warp, warp_filename)
            torch.save(certainty, certainty_filename)
        else:
            warp = torch.load(warp_filename, weights_only=True).to(self.device)
            certainty = torch.load(certainty_filename, weights_only=True).to(self.device)

        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

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

    def add_flows_into_datagraph(self, flow_source_frame: int, flow_target_frame: int):
        edge_data = self.data_graph.get_edge_observations(flow_source_frame, flow_target_frame)
        if edge_data.roma_flow_warp is None or edge_data.roma_flow_certainty is None:
            warp, certainty = self.next_cache_flow_roma(flow_source_frame, flow_target_frame, sample=10000)
        else:
            warp, certainty = edge_data.roma_flow_warp, edge_data.roma_flow_certainty

        data_graph = self.data_graph

        edge_data.roma_flow_warp = warp
        edge_data.roma_flow_certainty = certainty

        source_frame_data = data_graph.get_frame_data(flow_source_frame)
        target_frame_data = data_graph.get_frame_data(flow_target_frame)

        h1 = source_frame_data.image_shape.height
        w1 = source_frame_data.image_shape.width
        h2 = target_frame_data.image_shape.height
        w2 = target_frame_data.image_shape.width

        src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

        edge_data.src_pts_xy_roma = src_pts_xy_roma
        edge_data.dst_pts_xy_roma = dst_pts_xy_roma


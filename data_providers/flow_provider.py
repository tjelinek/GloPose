from pathlib import Path
from typing import List, Union

import torch
import torchvision
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model

from data_structures.data_graph import DataGraph


class RoMaFlowProviderDirect:

    def __init__(self, data_graph: DataGraph, device):
        self.device = device
        self.data_graph = data_graph
        self.flow_model: roma_model = None

    def _init_flow_model(self):
        self.flow_model: roma_model = roma_outdoor(device=self.device)

    def next_flow_roma(self, source_image_idx: int, target_image_idx: int, sample=None):
        if self.flow_model is None:
            self._init_flow_model()

        source_image_tensor = self.data_graph.get_frame_data(source_image_idx).frame_observation.observed_image
        target_image_tensor = self.data_graph.get_frame_data(target_image_idx).frame_observation.observed_image

        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image_tensor.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image_tensor.squeeze())

        warp, certainty = self.flow_model.match(source_image_roma, target_image_roma, device=self.device)
        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def add_flows_into_datagraph(self, flow_source_frame, flow_target_frame):
        edge_data = self.data_graph.get_edge_observations(flow_source_frame, flow_target_frame)
        if edge_data.flow_warp is None or edge_data.flow_certainty is None:
            warp, certainty = self.next_flow_roma(flow_source_frame, flow_target_frame, sample=10000)
        else:
            warp, certainty = edge_data.flow_warp, edge_data.flow_certainty
            # TODO handle sampling when reading from datagraph

        edge_data.flow_warp = warp
        edge_data.flow_certainty = certainty


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect):

    def __init__(self, data_graph: DataGraph, device, cache_dir: Path, image_files_paths: List,
                 allow_missing: bool = True):
        super().__init__(data_graph, device)
        self.flow_model: roma_model = roma_outdoor(device=device)

        self.saved_flow_paths = cache_dir
        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.image_names_sorted = sorted(Path(p) for p in image_files_paths)

        self.allow_missing: bool = allow_missing

    def next_flow_roma(self, source_image_idx: int, target_image_idx: int, sample=None):

        assert source_image_idx < len(self.image_names_sorted)
        assert target_image_idx < len(self.image_names_sorted)

        src_image_name = Path(self.image_names_sorted[source_image_idx])
        target_image_name = Path(self.image_names_sorted[target_image_idx])
        saved_filename = f'{src_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if (not warp_filename.exists() or not certainty_filename.exists()) and self.allow_missing:
            warp, certainty = super().next_flow_roma(source_image_idx, target_image_idx)

            torch.save(warp, warp_filename)
            torch.save(certainty, certainty_filename)
        else:
            warp = torch.load(warp_filename).to(self.device)
            certainty = torch.load(certainty_filename).to(self.device)

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
            warp = torch.load(warp_filename).to(self.device)
            certainty = torch.load(certainty_filename).to(self.device)
            return warp, certainty

        return None


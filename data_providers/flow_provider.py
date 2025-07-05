import shutil
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Union, Tuple, Optional

import torch
import torchvision
from einops import rearrange
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model
from romatch.utils.kde import kde

from configs.matching_configs.roma_configs.base_roma_config import BaseRomaConfig
from configs.matching_configs.ufm_configs.base_ufm_config import BaseUFMConfig
from data_structures.data_graph import DataGraph
from flow import roma_warp_to_pixel_coordinates, convert_to_roma_warp, convert_certainty_to_roma_format


class FlowProviderDirect(ABC):

    def __init__(self, device: str):
        self.device = device

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

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample=None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation: bool = False, only_foreground_matches=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        warp, certainty = self.compute_flow(source_image, target_image, sample,
                                            source_image_segmentation, target_image_segmentation, source_image_name,
                                            target_image_name, source_image_index, target_image_index,
                                            zero_certainty_outside_segmentation)

        h1 = source_image.shape[-2]
        w1 = source_image.shape[-1]
        h2 = target_image.shape[-2]
        w2 = target_image.shape[-1]
        src_pts_xy, dst_pts_xy = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)

        if len(src_pts_xy.shape) == 3:
            src_pts_xy = src_pts_xy.flatten(0, 1)
        if len(dst_pts_xy.shape) == 3:
            dst_pts_xy = dst_pts_xy.flatten(0, 1)

        src_pts_xy_int, dst_pts_xy_int = self.keypoints_to_int(src_pts_xy, dst_pts_xy, source_image, target_image)
        if as_int:
            src_pts_xy, dst_pts_xy = src_pts_xy_int, dst_pts_xy_int

        if only_foreground_matches:
            assert source_image_segmentation is not None and target_image_segmentation is not None
            assert len(source_image_segmentation.shape) == 2 and len(target_image_segmentation.shape) == 2
            in_segment_mask_src = source_image_segmentation[src_pts_xy_int[:, 1], src_pts_xy_int[:, 0]].bool()
            in_segment_mask_tgt = target_image_segmentation[dst_pts_xy_int[:, 1], dst_pts_xy_int[:, 0]].bool()

            fg_matches = in_segment_mask_src * in_segment_mask_tgt

            src_pts_xy = src_pts_xy[fg_matches]
            dst_pts_xy = dst_pts_xy[fg_matches]
            certainty = certainty[fg_matches]

        return src_pts_xy, dst_pts_xy, certainty

    @abstractmethod
    def sample(self, warp: torch.Tensor, certainty: torch.Tensor, sample: int) -> Tuple[torch.Tensor, torch.Tensor]:

        pass

    def zero_certainty_outside_segmentation(self, certainty: torch.Tensor,
                                            source_image_segmentation: torch.Tensor = None,
                                            target_image_segmentation: torch.Tensor = None) -> torch.Tensor:

        assert source_image_segmentation is not None or target_image_segmentation is not None

        certainty = certainty.clone()

        h, w = certainty.shape
        w //= 2
        if source_image_segmentation is not None:
            certainty[:, :w] *= source_image_segmentation.squeeze().bool().float()
        if target_image_segmentation is not None:
            certainty[:, w:2 * w] *= target_image_segmentation.squeeze().bool().float()

        return certainty


class PrecomputedFlowProviderDirect(FlowProviderDirect, ABC):

    def __init__(self, device, cache_dir: Path, data_graph: DataGraph = None,
                 allow_missing: bool = True, allow_disk_cache=True, purge_cache: bool = False):
        FlowProviderDirect.__init__(self, device)

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
        self.allow_disk_cache: bool = False

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

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None,
                     zero_certainty_outside_segmentation=False) -> Tuple[torch.Tensor, torch.Tensor]:

        certainty_filename, warp_filename = self._get_cache_filenames(source_image_index, source_image_name,
                                                                      target_image_index, target_image_name)

        warp, certainty = self._try_to_load_data(source_image_index, target_image_index, warp_filename,
                                                 certainty_filename)

        if warp is None or certainty is None:
            warp, certainty = super().compute_flow(source_image_tensor, target_image_tensor)

            if source_image_name and target_image_name and self.allow_missing and self.allow_disk_cache:
                torch.save(warp, warp_filename)
                torch.save(certainty, certainty_filename)

        if zero_certainty_outside_segmentation:
            certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                                 target_image_segmentation)

        self._save_to_datagraph(source_image_index, target_image_index, warp, certainty)

        if sample:
            if ((source_image_segmentation is not None and source_image_segmentation.sum() <= 5) or
                    (target_image_segmentation is not None and target_image_segmentation.sum() <= 5) and
                    zero_certainty_outside_segmentation):  # This should prevent the sampling to fail due to CUDA error
                warp = torch.zeros(0, 4).to(warp.device).to(warp.dtype)
                certainty = torch.zeros(0, ).to(certainty.device).to(certainty.dtype)
            else:
                warp, certainty = self.sample(warp, certainty, sample)

        return warp, certainty

    def _save_to_datagraph(self, source_image_index, target_image_index, warp, certainty):
        if self.data_graph is not None:
            if source_image_index is not None and target_image_index is not None:
                if not self.data_graph.G.has_edge(source_image_index, target_image_index):
                    self.data_graph.add_new_arc(source_image_index, target_image_index)

                edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
                if edge_data.roma_flow_warp is None:
                    edge_data.roma_flow_warp = warp
                if edge_data.roma_flow_warp_certainty is None:
                    edge_data.roma_flow_warp_certainty = certainty

    def _get_cache_filenames(self, source_image_index, source_image_name, target_image_index, target_image_name):
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
        return certainty_filename, warp_filename

    def _try_to_load_data(self, source_image_index, target_image_index, warp_filename, certainty_filename):
        warp, certainty = None, None
        if self._datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)
            if edge_data.roma_flow_warp is not None and edge_data.roma_flow_warp_certainty is not None:
                warp, certainty = edge_data.roma_flow_warp, edge_data.roma_flow_warp_certainty
        if (warp is None or certainty is None) and warp_filename is not None and certainty_filename is not None:
            if warp_filename.exists() and certainty_filename.exists() and self.allow_disk_cache:
                warp = torch.load(warp_filename, weights_only=True, map_location=self.device)
                certainty = torch.load(certainty_filename, weights_only=True, map_location=self.device)
        return warp, certainty

    def get_source_target_points(self, source_image: torch.Tensor, target_image: torch.Tensor,
                                 sample: int = None, source_image_segmentation: torch.Tensor = None,
                                 target_image_segmentation: torch.Tensor = None, source_image_name: Path = None,
                                 target_image_name: Path = None, source_image_index: int = None,
                                 target_image_index: int = None, as_int: bool = False,
                                 zero_certainty_outside_segmentation=False, only_foreground_matches=False) -> \
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        src_pts_xy = None
        dst_pts_xy = None
        certainty = None

        if self._datagraph_edge_exists(source_image_index, target_image_index):
            edge_data = self.data_graph.get_edge_observations(source_image_index, target_image_index)

            if edge_data.src_pts_xy_roma is not None and edge_data.dst_pts_xy_roma is not None:
                src_pts_xy, dst_pts_xy = edge_data.src_pts_xy_roma, edge_data.dst_pts_xy_roma

        if src_pts_xy is None or dst_pts_xy is None or certainty is None:
            src_pts_xy, dst_pts_xy, certainty = (
                super().get_source_target_points(source_image, target_image, sample,
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
                edge_data.src_pts_xy_roma = src_pts_xy
            if edge_data.dst_pts_xy_roma is None:
                edge_data.dst_pts_xy_roma = dst_pts_xy
            if edge_data.src_dst_certainty_roma is None:
                edge_data.src_dst_certainty_roma = certainty

        if as_int:
            src_pts_xy, dst_pts_xy = self.keypoints_to_int(src_pts_xy, dst_pts_xy, source_image, target_image)

        return src_pts_xy, dst_pts_xy, certainty

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


class RoMaFlowProviderDirect(FlowProviderDirect):

    def __init__(self, device, roma_config: BaseRomaConfig):
        FlowProviderDirect.__init__(self, device)
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
            warp, certainty = self.sample(warp, certainty, sample)

        return warp, certainty

    def sample(self, warp, certainty, sample):
        return self.flow_model.sample(warp, certainty, sample)

    def zero_certainty_outside_segmentation(self, certainty: torch.Tensor,
                                            source_image_segmentation: torch.Tensor = None,
                                            target_image_segmentation: torch.Tensor = None) -> torch.Tensor:
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


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect, PrecomputedFlowProviderDirect):

    def __init__(self, device, roma_config: BaseRomaConfig, cache_dir: Path, data_graph: DataGraph = None,
                 allow_missing: bool = True, allow_disk_cache=True, purge_cache: bool = False):
        RoMaFlowProviderDirect.__init__(self, device, roma_config)

        # Then initialize PrecomputedFlowProviderDirect with its parameters
        PrecomputedFlowProviderDirect.__init__(self, device, cache_dir, data_graph,
                                               allow_missing, allow_disk_cache, purge_cache)


class UFMFlowProviderDirect(FlowProviderDirect):

    def __init__(self, device, ufm_config: BaseUFMConfig):
        FlowProviderDirect.__init__(self, device)
        self.device = device
        self.ufm_config = ufm_config

        from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement
        self.model = UniFlowMatchClassificationRefinement.from_pretrained("infinity1096/UFM-Refine").to(self.device)

        self.model.eval()

        self.sample_mode = 'balanced'
        self.sample_thresh = 0.5

    @torch.no_grad()
    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None, zero_certainty_outside_segmentation: bool = False) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        h, w = source_image_tensor.shape[-2:]
        assert len(source_image_tensor.shape) == 3
        assert len(target_image_tensor.shape) == 3

        source_image_tensor_bhwc = rearrange(source_image_tensor[None], 'b c h w -> b h w c').to(torch.float)
        target_image_tensor_bhwc = rearrange(target_image_tensor[None], 'b c h w -> b h w c').to(torch.float)

        if self.ufm_config.backward:
            source_tensor_bhwc = torch.cat([source_image_tensor_bhwc, target_image_tensor_bhwc], dim=0)
            target_tensor_bhwc = torch.cat([target_image_tensor_bhwc, source_image_tensor_bhwc], dim=0)
        else:
            source_tensor_bhwc = source_image_tensor_bhwc
            target_tensor_bhwc = target_image_tensor_bhwc

        result = self.model.predict_correspondences_batched(source_image=source_tensor_bhwc,
                                                            target_image=target_tensor_bhwc,
                                                            data_norm_type='identity',)

        flow_forward = result.flow.flow_output[0]
        covisibility_forward = result.covisibility.mask[0]
        flow_backward = None
        covisibility_backward = None
        if self.ufm_config.backward:
            flow_backward = result.flow.flow_output[1]
            covisibility_backward = result.covisibility.mask[1]

        dst_pts_xy_forward = self.get_dst_pts(flow_forward, h, w)

        dst_pts_xy_roma = convert_to_roma_warp(dst_pts_xy_forward, flow_backward)
        covisibility = convert_certainty_to_roma_format(covisibility_forward, covisibility_backward)

        if zero_certainty_outside_segmentation and False:
            covisibility = self.zero_certainty_outside_segmentation(covisibility, source_image_segmentation,
                                                                    target_image_segmentation)

        if sample:
            dst_pts_xy_roma, covisibility = self.sample(dst_pts_xy_roma, covisibility, sample)

        return dst_pts_xy_roma, covisibility

    def get_dst_pts(self, flow_forward, h, w):
        y, x = torch.meshgrid(
            torch.arange(h, dtype=torch.float32, device=self.device),
            torch.arange(w, dtype=torch.float32, device=self.device),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=0)
        dst_pts_xy = coords + flow_forward
        return dst_pts_xy

    def sample(self, warp: torch.Tensor, certainty: torch.Tensor, sample: int) -> Tuple[torch.Tensor, torch.Tensor]:

        # Taken from RoMa implementation

        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            warp.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty,
                                         num_samples=min(expansion_factor * sample, len(certainty)),
                                         replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = 1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p,
                                             num_samples=min(sample, len(good_certainty)),
                                             replacement=False)

        match_samples = good_matches[balanced_samples]
        certainty_samples = good_certainty[balanced_samples]

        return match_samples, certainty_samples


class PrecomputedUFMFlowProviderDirect(UFMFlowProviderDirect, PrecomputedFlowProviderDirect):

    def __init__(self, device, umf_config: BaseUFMConfig, cache_dir: Path, data_graph: DataGraph = None,
                 allow_missing: bool = True, allow_disk_cache=True, purge_cache: bool = False):

        UFMFlowProviderDirect.__init__(self, device, umf_config)

        # Then initialize PrecomputedFlowProviderDirect with its parameters
        PrecomputedFlowProviderDirect.__init__(self, device, cache_dir, data_graph,
                                               allow_missing, allow_disk_cache, purge_cache)

    def compute_flow(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                     source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                     source_image_name: Path = None, target_image_name: Path = None, source_image_index: int = None,
                     target_image_index: int = None,
                     zero_certainty_outside_segmentation=False) -> Tuple[torch.Tensor, torch.Tensor]:

        certainty_filename, warp_filename = self._get_cache_filenames(source_image_index, source_image_name,
                                                                      target_image_index, target_image_name)

        warp, certainty = self._try_to_load_data(source_image_index, target_image_index, warp_filename,
                                                 certainty_filename)

        if warp is None or certainty is None:
            warp, certainty = UFMFlowProviderDirect.compute_flow(self, source_image_tensor, target_image_tensor)

            # if source_image_name and target_image_name and self.allow_missing and self.allow_disk_cache:
            #     torch.save(warp, warp_filename)
            #     torch.save(certainty, certainty_filename)

        if zero_certainty_outside_segmentation:
            certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                                 target_image_segmentation)

        self._save_to_datagraph(source_image_index, target_image_index, warp, certainty)

        if sample:
            if ((source_image_segmentation is not None and source_image_segmentation.sum() <= 5) or
                    (target_image_segmentation is not None and target_image_segmentation.sum() <= 5) and
                    zero_certainty_outside_segmentation):  # This should prevent the sampling to fail due to CUDA error
                warp = torch.zeros(0, 4).to(warp.device).to(warp.dtype)
                certainty = torch.zeros(0, ).to(certainty.device).to(certainty.dtype)
            else:
                warp, certainty = self.sample(warp, certainty, sample)

        return warp, certainty

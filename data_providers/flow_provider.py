import shutil
from pathlib import Path
from typing import Union, Tuple

import torch
import torchvision
from romatch import roma_outdoor
from romatch.models.model_zoo import roma_model

from flow import roma_warp_to_pixel_coordinates


class RoMaFlowProviderDirect:

    def __init__(self, device):
        self.device = device
        self.flow_model: roma_model = roma_outdoor(device=self.device)
        self.roma_size_hw = (864, 864)

    def next_flow_roma(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                       source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                       source_image_name: str = None, target_image_name: str = None) \
            -> Tuple[torch.Tensor, torch.Tensor]:

        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image_tensor.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image_tensor.squeeze())

        warp, certainty = self.flow_model.match(source_image_roma, target_image_roma, device=self.device)

        certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                             target_image_segmentation)

        if sample:
            warp, certainty = self.flow_model.sample(warp, certainty, sample)

        return warp, certainty

    def zero_certainty_outside_segmentation(self, certainty, source_image_segmentation, target_image_segmentation) ->\
            torch.Tensor:
        roma_h, roma_w = self.roma_size_hw

        certainty = certainty.clone()
        if source_image_segmentation is not None:
            source_image_segment_roma_size = torchvision.transforms.functional.resize(source_image_segmentation,
                                                                                      size=self.roma_size_hw)
            certainty[:, :roma_w] *= source_image_segment_roma_size.mT.squeeze().bool().float()
        if target_image_segmentation is not None:
            target_image_segment_roma_size = torchvision.transforms.functional.resize(target_image_segmentation,
                                                                                      size=self.roma_size_hw)
            certainty[:, roma_w:2 * roma_w] *= target_image_segment_roma_size.mT.squeeze().bool().float()
        return certainty

    def get_source_target_points_roma(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor,
                                      sample=None, source_image_segmentation: torch.Tensor = None,
                                      target_image_segmentation: torch.Tensor = None,
                                      source_image_name: Path = None, target_image_name: Path = None,
                                      as_int: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        warp, certainty = self.next_flow_roma(source_image_tensor, target_image_tensor, sample,
                                              source_image_segmentation, target_image_segmentation,
                                              source_image_name, target_image_name)

        h1 = source_image_tensor.shape[-2]
        w1 = source_image_tensor.shape[-1]
        h2 = target_image_tensor.shape[-2]
        w2 = target_image_tensor.shape[-1]
        src_pts_xy_roma, dst_pts_xy_roma = roma_warp_to_pixel_coordinates(warp, h1, w1, h2, w2)
        if as_int:
            src_pts_xy_roma = src_pts_xy_roma.to(torch.int)
            dst_pts_xy_roma = dst_pts_xy_roma.to(torch.int)

        if len(src_pts_xy_roma.shape) == 3:
            src_pts_xy_roma = src_pts_xy_roma.flatten(0, 1)
        if len(dst_pts_xy_roma.shape) == 3:
            dst_pts_xy_roma = dst_pts_xy_roma.flatten(0, 1)

        return src_pts_xy_roma, dst_pts_xy_roma, certainty


class PrecomputedRoMaFlowProviderDirect(RoMaFlowProviderDirect):

    def __init__(self, device, cache_dir: Path, allow_missing: bool = True,
                 purge_cache: bool = False):
        super().__init__(device)

        self.saved_flow_paths = cache_dir
        self.warps_path = cache_dir / 'warps'
        self.certainties_path = cache_dir / 'certainties'

        if purge_cache and self.warps_path.exists():
            shutil.rmtree(self.warps_path)
        if purge_cache and self.certainties_path.exists():
            shutil.rmtree(self.certainties_path)

        self.warps_path.mkdir(exist_ok=True, parents=True)
        self.certainties_path.mkdir(exist_ok=True, parents=True)

        self.allow_missing: bool = allow_missing

    def next_flow_roma(self, source_image_tensor: torch.Tensor, target_image_tensor: torch.Tensor, sample=None,
                       source_image_segmentation: torch.Tensor = None, target_image_segmentation: torch.Tensor = None,
                       source_image_name: Path = None, target_image_name: Path = None) -> (
            Tuple)[torch.Tensor, torch.Tensor]:

        if source_image_name is None or target_image_name is None:
            return super().next_flow_roma(source_image_tensor, target_image_tensor, sample, source_image_segmentation,
                                          target_image_segmentation)

        saved_filename = f'{source_image_name.stem}___{target_image_name.stem}.pt'

        warp_filename = self.warps_path / saved_filename
        certainty_filename = self.certainties_path / saved_filename

        if (not warp_filename.exists() or not certainty_filename.exists()) and self.allow_missing:
            warp, certainty = super().next_flow_roma(source_image_tensor, target_image_tensor)

            torch.save(warp, warp_filename)
            torch.save(certainty, certainty_filename)
        else:
            warp = torch.load(warp_filename, weights_only=True).to(self.device)
            certainty = torch.load(certainty_filename, weights_only=True).to(self.device)

        certainty = self.zero_certainty_outside_segmentation(certainty, source_image_segmentation,
                                                             target_image_segmentation)

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

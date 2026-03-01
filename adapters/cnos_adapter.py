"""Adapter for the cnos external repository.

This is the SOLE location in GloPose that manipulates sys.path for cnos
and imports cnos internals. All other modules import from here.

The module-level sys.path.append ensures pickle compatibility: old ViewGraph
pickles may reference cnos types (e.g. src.model.utils.Detections) and need
cnos on the path at deserialization time.
"""

import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

# ── Ensure cnos is on sys.path (needed for pickle compat and all cnos imports) ──
_cnos_path = './repositories/cnos'
if _cnos_path not in sys.path:
    sys.path.append(_cnos_path)


# ---------------------------------------------------------------------------
# DescriptorExtractor protocol — consumers program against this
# ---------------------------------------------------------------------------

@runtime_checkable
class DescriptorExtractor(Protocol):
    """Protocol for DINOv2/v3 descriptor extraction.

    Callers pass raw numpy images, mask tensors, and box tensors.
    The adapter handles constructing any cnos-internal types.
    """

    def extract_descriptors(self, image_np, masks: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        """Extract (cls_descriptors, patch_descriptors) for the given proposals.

        Args:
            image_np: HxWx3 uint8 numpy array.
            masks: [N, H, W] boolean/float mask tensor.
            boxes: [N, 4] long tensor in xyxy format.

        Returns:
            (cls_descriptors, patch_descriptors) tensors.
        """
        ...

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path) -> tuple[Tensor, Tensor]:
        """Extract descriptors from image and segmentation file paths.

        Returns:
            (cls_descriptors, patch_descriptors) tensors.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete implementation wrapping cnos CustomDINOv2
# ---------------------------------------------------------------------------

class CnosDescriptorExtractor:
    """Wraps cnos CustomDINOv2 behind the DescriptorExtractor interface.

    Callers never need to construct cnos Detections objects — this adapter
    does it internally.
    """

    def __init__(self, model: str = 'dinov3', mask_detections: bool = True, device: str = 'cuda'):
        from src.model.dinov2 import descriptor_from_hydra
        self._dino_model = descriptor_from_hydra(model=model, mask_detections=mask_detections, device=device)

    def extract_descriptors(self, image_np, masks: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
        from src.model.utils import Detections
        detections = Detections({'masks': masks, 'boxes': boxes})
        return self._dino_model(image_np, detections)

    def get_detections_from_files(self, image_path: Path, segmentation_path: Path) -> tuple[Tensor, Tensor]:
        return self._dino_model.get_detections_from_files(image_path, segmentation_path)


def create_descriptor_extractor(model: str = 'dinov3', mask_detections: bool = True,
                                device: str = 'cuda') -> CnosDescriptorExtractor:
    """Factory: primary entry point for creating a descriptor extractor."""
    return CnosDescriptorExtractor(model=model, mask_detections=mask_detections, device=device)


# ---------------------------------------------------------------------------
# Hydra config loading for detection matching/postprocessing
# ---------------------------------------------------------------------------

def load_cnos_matching_config(overrides: dict[str, object]) -> tuple[dict, dict]:
    """Load cnos matching and post-processing configs via Hydra.

    Args:
        overrides: Dict of matching_config overrides (key -> value).
            Keys with None values are skipped.

    Returns:
        (matching_config, postprocessing_config) as plain dicts.
    """
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    cfg_dir = (Path(__file__).parent.parent / 'repositories' / 'cnos' / 'configs').resolve()
    hydra_overrides = []
    for k, v in overrides.items():
        if v is not None:
            hydra_overrides.append(f'model.matching_config.{k}={v}')

    with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
        cnos_cfg = compose(config_name="run_inference", overrides=hydra_overrides)

    matching_config = instantiate(cnos_cfg.model.matching_config)
    postprocessing_config = instantiate(cnos_cfg.model.post_processing_config)
    return matching_config, postprocessing_config


# ---------------------------------------------------------------------------
# Escape hatch: cnos Detections container (for NMS methods)
# ---------------------------------------------------------------------------

def make_cnos_detections(data: dict) -> 'CnosDetections':
    """Construct a cnos Detections object from a dict.

    Use this when you need cnos NMS methods (apply_nms_per_object_id,
    apply_nms_for_masks_inside_masks, filter). Prefer extract_descriptors
    for descriptor computation.
    """
    from src.model.utils import Detections
    return Detections(data)


# Type alias for documentation
CnosDetections = object  # Runtime type is src.model.utils.Detections

"""NMS algorithms for batched detections."""

import torch
import torchvision.ops

from data_structures.types import DetectionSet
from utils.image_utils import compute_overlap_ratio


def nms_per_object_id(detections: DetectionSet, nms_thresh: float = 0.5) -> torch.Tensor:
    """Apply NMS separately per object ID.

    Mutates *detections* in-place via ``detections.filter()``.

    Returns:
        Indices of kept detections.
    """
    from adapters.dino_utils import BatchedData

    keep_idxs = BatchedData(None)
    all_indexes = torch.arange(len(detections.object_ids), device=detections.boxes.device)
    for object_id in torch.unique(detections.object_ids):
        this_object_detections_mask = detections.object_ids == object_id
        this_object_detections_indices = all_indexes[this_object_detections_mask]
        this_obj_keep_indices = torchvision.ops.nms(
            detections.boxes[this_object_detections_mask].float(),
            detections.scores[this_object_detections_mask].float(),
            nms_thresh,
        )
        keep_idxs.cat(this_object_detections_indices[this_obj_keep_indices])

    keep_idxs = keep_idxs.data
    detections.filter(keep_idxs)
    return keep_idxs


def nms_masks_inside_masks(detections: DetectionSet) -> torch.Tensor:
    """Suppress detections whose masks are mostly inside higher-scoring detections.

    Mutates *detections* in-place via ``detections.filter()``.

    Returns:
        Indices of kept detections.
    """
    all_masks = detections.masks
    overlap_ratio = compute_overlap_ratio(all_masks, all_masks)  # [N, N]

    higher_score = detections.scores.unsqueeze(1) > detections.scores.unsqueeze(0)  # [N, N]
    is_enclosed = overlap_ratio > 0.8  # [N, N]

    should_suppress = is_enclosed & higher_score  # [N, N]
    keep_detections = ~should_suppress.any(dim=1)  # [N]

    keep_detections_indices = torch.nonzero(keep_detections).squeeze(1)
    detections.filter(keep_detections_indices)
    return keep_detections_indices

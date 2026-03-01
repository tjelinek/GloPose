"""Bounding box utility functions.

Vendored from repositories/cnos/src/utils/bbox_utils.py to avoid
runtime dependency on the cnos repository.
"""

import numpy as np


def xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] box format to [x1, y1, x2, y2] format."""
    if len(bbox.shape) == 1:
        x, y, w, h = bbox
        return [x, y, x + w - 1, y + h - 1]
    elif len(bbox.shape) == 2:
        x, y, w, h = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        return np.stack([x, y, x + w, y + h], axis=1)
    else:
        raise ValueError("bbox must be a numpy array of shape (4,) or (N, 4)")

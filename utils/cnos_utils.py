import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from segment_anything.utils.amg import rle_to_mask

if './repositories/cnos' not in sys.path:
    sys.path.append('./repositories/cnos')

from src.utils.bbox_utils import xywh_to_xyxy
from src.model.utils import Detections


def get_default_detections_per_scene_and_image(default_detections_file: Path) -> Dict[Tuple[int, int], List]:
    with open(default_detections_file, 'r') as f:
        default_detections_data = json.load(f)
        default_detections_scene_im_dict = defaultdict(list)
        for i, item in enumerate(default_detections_data):
            im_id: int = item['image_id']
            scene_id: int = item['scene_id']
            default_detections_scene_im_dict[(im_id, scene_id)].append(item)

    return default_detections_scene_im_dict


def get_detections_cnos_format(default_detections_scene_im: Dict[Tuple[int, int], List], scene_id: int,
                               im_id: int, device: str = 'cuda') -> Detections:
    """
    Takes output of get_default_detections_per_scene_and_image
    """
    item = default_detections_scene_im[(im_id, scene_id)]
    N = len(item)
    masks = [torch.tensor(rle_to_mask(item[i]['segmentation'])).to(device)
             for i in range(N)]
    masks = torch.stack(masks)
    bboxes = torch.tensor([xywh_to_xyxy(np.array(item[i]['bbox'])) for i in range(N)]).to(device)
    scores = torch.tensor([item[i]['score'] for i in range(N)]).to(device)
    obj_ids = torch.tensor([item[i]['category_id'] - 1 for i in range(N)]).to(device)

    detections_dict = {
        'object_ids': obj_ids,
        'bbox': bboxes,
        'scores': scores,
        'masks': masks,
    }

    detections = Detections(detections_dict)

    return detections

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from torchvision.ops import masks_to_boxes
from tqdm import tqdm

from utils.bop_challenge import extract_object_id

sys.path.append('./repositories/cnos')
from src.model.utils import Detections
from src.model.dinov2 import CustomDINOv2

device = 'cuda'

bop_base = Path('/mnt/personal/jelint19/data/bop')
dataset = 'hope'
split = 'onboarding_static'

path_to_dataset = bop_base / dataset
path_to_split = path_to_dataset / split

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
cfg_dir = (Path(__file__).parent / 'repositories' / 'cnos' / 'configs').resolve()
with initialize_config_dir(config_dir=str(cfg_dir), version_base=None):
    cnos_cfg = compose(config_name="run_inference")


dino_descriptor: CustomDINOv2 = instantiate(cnos_cfg.model.descriptor_model).to(device)

all_images = []
all_segmentations = []
object_classes = []
dino_cls_descriptors = []

sequences = sorted(path_to_split.iterdir())
for sequence in tqdm(sequences, desc="Processing sequences", total=len(sequences)):

    if not sequence.is_dir():
        continue

    rgb_folder = sequence / 'rgb'
    segmentation_folder = sequence / 'mask_visib'
    scene_gt = sequence / 'scene_gt.json'

    object_id = extract_object_id(scene_gt, 0)[1]

    rgb_files = sorted(rgb_folder.iterdir())
    seg_files = sorted(segmentation_folder.iterdir())
    for image_path, seg_path in tqdm(list(zip(rgb_files, seg_files)), total=len(rgb_files), desc="Processing images"):

        image = Image.open(image_path).convert('RGB')
        image_tensor = torch.from_numpy(np.asarray(image)).to(device)

        # Load segmentation mask
        segmentation = Image.open(seg_path)
        segmentation_mask = torch.from_numpy(np.array(segmentation)).unsqueeze(0).to(device)

        segmentation_bbox = masks_to_boxes(segmentation_mask)
        image_np = image_tensor.to(torch.uint8).numpy(force=True)
        detections = Detections({'masks': segmentation_mask, 'boxes': segmentation_bbox})
        dino_cls_descriptor, dino_dense_descriptor = dino_descriptor(image_np, detections)

        all_images.append(image_path)
        all_segmentations.append(seg_path)
        object_classes.append(object_id)
        dino_cls_descriptors.append(dino_cls_descriptor.squeeze().numpy(force=True))

object_classes = np.array(object_classes)
dino_cls_descriptors = np.array(dino_cls_descriptors)
all_images = np.array(all_images)
all_segmentations = np.array(all_segmentations)

permutation = np.random.permutation(len(all_images))
all_images = all_images[permutation]
all_segmentations = all_segmentations[permutation]
object_classes = object_classes[permutation]
dino_cls_descriptors = dino_cls_descriptors[permutation]

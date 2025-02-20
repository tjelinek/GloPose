import csv
import math
from pathlib import Path
from typing import Iterable, Dict, Tuple, cast, List

import torch
import trimesh
import imageio
from PIL import Image
from torchvision import transforms

from tracker_config import TrackerConfig


def load_texture(texture_path: Path, texture_size: int) -> torch.Tensor:
    texture = torch.from_numpy(imageio.v2.imread(texture_path))
    texture = texture.permute(2, 0, 1)[None].cuda() / 255.0
    if max(texture.shape[-2:]) > texture_size:
        resize = transforms.Resize(size=texture_size)
        texture = resize(texture)

    return texture


def load_mesh(mesh_path: Path):
    import kaolin
    from kaolin.io.utils import mesh_handler_naive_triangulate

    gt_mesh_prototype: kaolin.rep.SurfaceMesh = kaolin.io.obj.import_mesh(str(mesh_path), with_materials=True,
                                                            heterogeneous_mesh_handler=mesh_handler_naive_triangulate)

    gt_mesh_prototype.uvs[:, 1] = 1.0 - gt_mesh_prototype.uvs[:, 1]
    # Fixing import that was changed in this commit
    # https://github.com/NVIDIAGameWorks/kaolin/commit/9cf895aa9af7769ae6ef23e654c1a42bcf094988

    return gt_mesh_prototype


def load_mesh_using_trimesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh")
    return cast(trimesh.Trimesh, mesh)


def load_gt_annotations_file(file_path) -> Tuple[torch.Tensor, torch.Tensor]:
    # Initialize empty lists to store the data
    frames = []
    rotations_degrees = []
    translations = []

    # Load the CSV file
    with open(file_path, 'r') as csvfile:
        reader: Iterable[Dict] = csv.DictReader(csvfile)
        for row in reader:
            # Append frame number
            frames.append(int(row['frame']))

            # Append rotations in degrees
            rotations_degrees.append([
                float(row['rot_x']),
                float(row['rot_y']),
                float(row['rot_z'])
            ])

            # Append translations
            translations.append([
                float(row['trans_x']),
                float(row['trans_y']),
                float(row['trans_z'])
            ])

    # Convert rotations from degrees to radians
    rotations_radians = [[math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2])] for rot in
                         rotations_degrees]

    # Create tensors
    rotations_tensor = torch.tensor(rotations_radians).cuda()
    translations_tensor = torch.tensor(translations).cuda()

    return rotations_tensor, translations_tensor


def load_gt_data(config: TrackerConfig):
    gt_texture = None
    gt_mesh = None
    gt_rotations = None
    gt_translations = None
    if config.gt_texture_path is not None:
        gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
    if config.gt_mesh_path is not None:
        gt_mesh = load_mesh(Path(config.gt_mesh_path))
    if config.gt_track_path is not None:
        gt_rotations, gt_translations = load_gt_annotations_file(config.gt_track_path)

    return gt_texture, gt_mesh, gt_rotations, gt_translations


def get_initial_image_and_segment(images_list, segmentations_list, dataset_delimeters=None, segmentation_channel=0) \
        -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if dataset_delimeters is None:
        dataset_delimeters = [0]

    images, segmentations = [], []
    for delimeter in dataset_delimeters:
        first_segment = Image.open(segmentations_list[delimeter])
        first_image = Image.open(images_list[delimeter])
        first_segment_resized = first_segment.resize(first_image.size, Image.NEAREST)
        transform = transforms.ToTensor()
        first_segment_tensor = transform(first_segment_resized)[segmentation_channel].squeeze()
        first_image_tensor = transform(first_image).squeeze()

        images.append(first_image_tensor), segmentations.append(first_segment_tensor)

    return images, segmentations

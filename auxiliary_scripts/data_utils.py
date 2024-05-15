import csv
import math
from pathlib import Path
from typing import Iterable, Dict, Tuple

import kaolin
import torch
from kaolin.io.utils import mesh_handler_naive_triangulate
from torchvision import transforms

import imageio


def load_texture(texture_path: Path, texture_size: int) -> torch.Tensor:
    texture = torch.from_numpy(imageio.v2.imread(texture_path))
    texture = texture.permute(2, 0, 1)[None].cuda() / 255.0
    if max(texture.shape[-2:]) > texture_size:
        resize = transforms.Resize(size=texture_size)
        texture = resize(texture)

    return texture


def load_mesh(mesh_path: Path) -> kaolin.rep.SurfaceMesh:
    gt_mesh_prototype = kaolin.io.obj.import_mesh(str(mesh_path), with_materials=True,
                                                  heterogeneous_mesh_handler=mesh_handler_naive_triangulate)

    return gt_mesh_prototype


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
    rotations_tensor = torch.tensor(rotations_radians).unsqueeze(0).cuda()
    translations_tensor = torch.tensor(translations).unsqueeze(0).unsqueeze(1).cuda()

    return rotations_tensor, translations_tensor

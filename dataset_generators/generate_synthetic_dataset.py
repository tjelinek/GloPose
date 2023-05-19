import types
import warnings
from pathlib import Path

import torch
import kaolin
from kornia.geometry import quaternion_to_rotation_matrix

from dataset_generators.generator_utils import setup_renderer, generate_and_save_images, generate_1_DoF_rotation, \
    generate_rotating_textured_object
from utils import quaternion_from_euler, deg_to_rad, load_config
from main_settings import dataset_folder

warnings.filterwarnings("ignore")


def generate_8_colored_sphere(config, rendering_destination, segmentation_destination):
    prototype_path = Path('./prototypes/sphere.obj')
    DEVICE = 'cuda'
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)

    colors = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [255, 255, 0],  # yellow
        [0, 255, 255],  # cyan
        [255, 0, 255],  # magenta
        [255, 165, 0],  # orange
        [255, 255, 255],  # white
    ]

    vertices_features = torch.zeros(mesh.vertices.shape[0], 3)

    for i, vertex in zip(range(len(mesh.vertices)), mesh.vertices):
        color_code = [vertex[0] > 0, vertex[1] > 0, vertex[2] > 0]
        vertices_features[i] = torch.tensor(colors[color_code[0] + 2 * color_code[1] + 4 * color_code[2]])

    width = 1000
    height = 1000
    faces = mesh.faces

    rendering = setup_renderer(config, faces, height, width, DEVICE)

    translation = torch.zeros((1, 3))[None]
    face_features = vertices_features[mesh.faces][None]

    rotations = generate_1_DoF_rotation()

    for i, (yaw, pitch, roll) in enumerate(rotations):
        rotation_quaternion = quaternion_from_euler(roll=torch.Tensor([deg_to_rad(roll)]),
                                                    pitch=torch.Tensor([deg_to_rad(pitch)]),
                                                    yaw=torch.Tensor([deg_to_rad(yaw)]))

        rotation_matrix = quaternion_to_rotation_matrix(torch.Tensor(rotation_quaternion))[None]

        with torch.no_grad():
            _, _, _, ren_mask, ren_features, _ \
                = rendering.render_mesh_with_dibr(face_features.to(DEVICE), rotation_matrix.to(DEVICE),
                                                  translation.to(DEVICE), mesh.vertices.to(DEVICE))

            generate_and_save_images(i, ren_features, ren_mask, rendering_destination,
                                     segmentation_destination)


def generate_textured_sphere(config, rendering_destination: Path, segmentation_destination: Path):
    prototype_path = Path('./prototypes/sphere.obj')
    tex_path = Path('./prototypes/tex3.png')

    width = 1000
    height = 1000

    generate_rotating_textured_object(config, prototype_path, rendering_destination, segmentation_destination, tex_path,
                                      width, height)


if __name__ == '__main__':
    _config = load_config('./configs/config_deep.yaml')
    _config = types.SimpleNamespace(**_config)

    synthetic_dataset_folder = dataset_folder / Path('SyntheticObjects')
    rendering_dir = Path('renderings')
    segmentation_dir = Path('segmentations')

    rendering_path = synthetic_dataset_folder / Path('Textured_Sphere') / rendering_dir
    segmentation_path = synthetic_dataset_folder / Path('Textured_Sphere') / segmentation_dir
    generate_textured_sphere(_config, rendering_path, segmentation_path)

    rendering_path = synthetic_dataset_folder / Path('8_Colored_Sphere') / rendering_dir
    segmentation_path = synthetic_dataset_folder / Path('8_Colored_Sphere') / segmentation_dir
    generate_8_colored_sphere(_config, rendering_path, segmentation_path)

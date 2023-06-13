import kaolin
import torch
import types
import warnings
from pathlib import Path

from dataset_generators.generator_utils import setup_renderer, generate_1_DoF_rotation, \
    generate_rotating_textured_object, render_object_poses
from main_settings import dataset_folder
from utils import load_config

warnings.filterwarnings("ignore")


def generate_8_colored_sphere(config, rendering_destination, segmentation_destination, optical_flow_destination):
    prototype_path = Path('./prototypes/sphere.obj')
    DEVICE = 'cuda'
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)

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

    face_features = vertices_features[mesh.faces][None]

    rotations = generate_1_DoF_rotation(step=2.0)

    # Render the object without using texture maps
    render_object_poses(rendering, mesh.vertices, face_features, None, rotations, optical_flow_destination,
                        rendering_destination, segmentation_destination, DEVICE)


def generate_6_colored_cube(config, rendering_destination, segmentation_destination, optical_flow_destination):
    DEVICE = 'cuda'
    width = 1000
    height = 1000

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)

    colors = [
        [255, 0, 0],  # red
        [0, 255, 0],  # green
        [0, 0, 255],  # blue
        [255, 255, 0],  # yellow
        [0, 255, 255],  # cyan
        [255, 0, 255],  # magenta
    ]

    vertices = torch.tensor([
        [1, 1, 1],  # Quadrant 1 (+x, +y, +z)
        [1, 1, -1],  # Quadrant 2 (+x, +y, -z)
        [1, -1, 1],  # Quadrant 3 (+x, -y, +z)
        [1, -1, -1],  # Quadrant 4 (+x, -y, -z)
        [-1, 1, 1],  # Quadrant 5 (-x, +y, +/-z)
        [-1, 1, -1],  # Quadrant 5 (-x, +y, +/-z)
        [-1, -1, 1],  # Quadrant 6 (-x, -y, +/-z)
        [-1, -1, -1],  # Quadrant 6 (-x, -y, +/-z)
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 2, 3], [3, 1, 0],
        [1, 3, 7], [7, 5, 1],
        [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0],
        [0, 1, 5], [5, 4, 0],
        [2, 6, 7], [7, 3, 2]
    ], dtype=torch.int64)

    rendering = setup_renderer(config, faces, height, width, DEVICE)

    translation = torch.zeros((1, 3))[None]
    face_features = torch.zeros((1, len(faces), 3, 3))

    for i, face in enumerate(faces):
        color = colors[i // 2]

        face_features[0, i, :, :] = torch.tensor(color)

    rotations = generate_1_DoF_rotation(step=2.0)

    # Render the object without using texture maps
    render_object_poses(rendering, vertices, face_features, None, rotations, optical_flow_destination,
                        rendering_destination, segmentation_destination, DEVICE)


def generate_textured_sphere(config, rendering_destination: Path, segmentation_destination: Path,
                             optical_flow_destination):
    prototype_path = Path('./prototypes/sphere.obj')
    tex_path = Path('./prototypes/tex.png')

    width = 1000
    height = 1000

    generate_rotating_textured_object(config, prototype_path, tex_path, rendering_destination, segmentation_destination,
                                      optical_flow_destination, width, height)


if __name__ == '__main__':
    _config = load_config('./configs/config_deep.yaml')
    _config = types.SimpleNamespace(**_config)

    synthetic_dataset_folder = dataset_folder / Path('SyntheticObjects')
    rendering_dir = Path('renderings')
    segmentation_dir = Path('segmentations')
    optical_flow_dir = Path('optical_flow')

    rendering_path = synthetic_dataset_folder / Path('Textured_Sphere') / rendering_dir
    segmentation_path = synthetic_dataset_folder / Path('Textured_Sphere') / segmentation_dir
    optical_flow_path = synthetic_dataset_folder / Path('Textured_Sphere') / optical_flow_dir
    generate_textured_sphere(_config, rendering_path, segmentation_path, optical_flow_path)

    rendering_path = synthetic_dataset_folder / Path('8_Colored_Sphere') / rendering_dir
    segmentation_path = synthetic_dataset_folder / Path('8_Colored_Sphere') / segmentation_dir
    optical_flow_path = synthetic_dataset_folder / Path('8_Colored_Sphere') / optical_flow_dir
    generate_8_colored_sphere(_config, rendering_path, segmentation_path, optical_flow_path)

    rendering_path = synthetic_dataset_folder / Path('6_Colored_Cube') / rendering_dir
    segmentation_path = synthetic_dataset_folder / Path('6_Colored_Cube') / segmentation_dir
    optical_flow_path = synthetic_dataset_folder / Path('6_Colored_Cube') / optical_flow_dir
    generate_6_colored_cube(_config, rendering_path, segmentation_path, optical_flow_path)

import kaolin
import torch
import types
import warnings
from pathlib import Path

from dataset_generators.generator_utils import setup_renderer, \
    generate_rotating_and_translating_textured_object, render_object_poses, generate_rotations_y, generate_rotations_xy, \
    generate_zero_rotations, generate_sinusoidal_translations, generate_circular_translation
from main_settings import dataset_folder
from utils import load_config

warnings.filterwarnings("ignore")


def generate_8_colored_sphere(config, rendering_destination, segmentation_destination, optical_flow_destination,
                              gt_tracking_log_file, rotations):
    prototype_path = Path('./prototypes/sphere.obj')
    DEVICE = 'cuda'
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)
    gt_tracking_log_file.parent.mkdir(parents=True, exist_ok=True)

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

    # Render the object without using texture maps
    render_object_poses(rendering, mesh.vertices, face_features, None, rotations, None,
                        optical_flow_destination, rendering_destination, segmentation_destination, gt_tracking_log_file,
                        DEVICE, None, None)


def generate_6_colored_cube(config, rendering_destination, segmentation_destination, optical_flow_destination,
                            gt_tracking_log_file, rotations):
    DEVICE = 'cuda'
    width = 1000
    height = 1000

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)
    gt_tracking_log_file.parent.mkdir(parents=True, exist_ok=True)

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

    rendering = setup_renderer(config, faces.numpy(), height, width, DEVICE)

    face_features = torch.zeros((1, len(faces), 3, 3))

    for i, face in enumerate(faces):
        color = colors[i // 2]

        face_features[0, i, :, :] = torch.tensor(color)

    # Render the object without using texture maps
    render_object_poses(rendering, vertices, face_features, None, rotations, None, optical_flow_destination,
                        rendering_destination, segmentation_destination, gt_tracking_log_file, DEVICE, None, None)


def generate_textured_sphere(config, rendering_destination: Path, segmentation_destination: Path,
                             optical_flow_destination, gt_tracking_log_file, rotations, translations=None):
    prototype_path = Path('./prototypes/sphere.obj')
    tex_path = Path('./prototypes/tex.png')

    width = 1000
    height = 1000

    generate_rotating_and_translating_textured_object(config, prototype_path, tex_path, rendering_destination,
                                                      segmentation_destination, optical_flow_destination,
                                                      gt_tracking_log_file, width, height, rotations=rotations,
                                                      translations=translations)


def generate_translations_for_obj(obj_name, synthetic_dataset_folder, rendering_dir, segmentation_dir,
                                  optical_flow_dir, gt_tracking_log_dir, steps, translations):
    rendering_path = synthetic_dataset_folder / obj_name / rendering_dir
    segmentation_path = synthetic_dataset_folder / obj_name / segmentation_dir
    optical_flow_path = synthetic_dataset_folder / obj_name / optical_flow_dir
    gt_tracking_log_file = synthetic_dataset_folder / obj_name / gt_tracking_log_dir / Path('gt_tracking_log.csv')

    rots = generate_zero_rotations(steps=steps)

    generate_textured_sphere(_config, rendering_path, segmentation_path, optical_flow_path, gt_tracking_log_file,
                             rots, translations=translations)


def generate_rotating_objects():
    rot_mags = {2, 5, 10}
    for rot_mag in rot_mags:

        rot_gens = [
            (generate_rotations_y, 'y'),
            # (generate_rotations_x, 'x'),
            (generate_rotations_xy, 'xy'),
            # (generate_rotations_xyz, 'xyz')
        ]

        for rots_gen, suffix in rot_gens:

            objects = [
                (f'Textured_Sphere_{rot_mag}_{suffix}', generate_textured_sphere),
                (f'8_Colored_Sphere_{rot_mag}_{suffix}', generate_8_colored_sphere),
                (f'6_Colored_Cube_{rot_mag}_{suffix}', generate_6_colored_cube)
            ]

            rots = rots_gen(step=float(rot_mag))

            for obj_name, generate_obj_func in objects:
                rendering_path = synthetic_dataset_folder / obj_name / rendering_dir
                segmentation_path = synthetic_dataset_folder / obj_name / segmentation_dir
                optical_flow_path = synthetic_dataset_folder / obj_name / optical_flow_dir
                gt_tracking_log_file = synthetic_dataset_folder / obj_name / gt_tracking_log_dir / Path(
                    'gt_tracking_log.csv')

                generate_obj_func(_config, rendering_path, segmentation_path, optical_flow_path, gt_tracking_log_file,
                                  rots)


if __name__ == '__main__':
    _config = load_config('./configs/config_generator.yaml')
    _config = types.SimpleNamespace(**_config)

    synthetic_dataset_folder = dataset_folder / Path('SyntheticObjects')
    rendering_dir = Path('renderings')
    segmentation_dir = Path('segmentations')
    optical_flow_dir = Path('optical_flow')
    gt_tracking_log_dir = Path('gt_tracking_log')

    generate_rotating_objects()

    # Generate translating spheres
    steps = 72
    translations = generate_sinusoidal_translations(steps=steps)
    generate_translations_for_obj('Translating_Textured_Sphere', synthetic_dataset_folder, rendering_dir,
                                  segmentation_dir, optical_flow_dir, gt_tracking_log_dir, steps, translations)

    translations = generate_circular_translation(steps=steps)
    generate_translations_for_obj('Circulating_Textured_Sphere', synthetic_dataset_folder, rendering_dir,
                                  segmentation_dir, optical_flow_dir, gt_tracking_log_dir, steps, translations)

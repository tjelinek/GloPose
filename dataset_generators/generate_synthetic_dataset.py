import kaolin
import torch
import warnings
from pathlib import Path

import dataset_generators.scenarios
from dataset_generators.scenarios import MovementScenario
from dataset_generators.generator_utils import setup_renderer, \
    generate_rotating_and_translating_textured_object, render_object_poses
from main_settings import dataset_folder
from utils import load_config

warnings.filterwarnings("ignore")

RESOLUTION = 800


def generate_8_colored_sphere(config, rendering_destination, segmentation_destination, optical_flow_destination,
                              gt_tracking_log_file, movement_scenario: MovementScenario):
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
        [255, 255, 0],  # orange
        [255, 255, 255],  # white
    ]

    vertices_features = torch.zeros(mesh.vertices.shape[0], 3)

    for i, vertex in zip(range(len(mesh.vertices)), mesh.vertices):
        color_code = [vertex[0] >= 0, vertex[1] >= 0, vertex[2] >= 0]
        vertices_features[i] = torch.tensor(colors[color_code[0] + 2 * color_code[1] + 4 * color_code[2]])

    width = RESOLUTION
    height = RESOLUTION
    faces = mesh.faces

    rendering = setup_renderer(config, faces, height, width, DEVICE)

    face_features = vertices_features[mesh.faces][None]

    # Render the object without using texture maps
    render_object_poses(rendering, mesh.vertices, face_features, None, movement_scenario, optical_flow_destination,
                        rendering_destination, segmentation_destination, gt_tracking_log_file)


def generate_6_colored_cube(config, rendering_destination, segmentation_destination, optical_flow_destination,
                            gt_tracking_log_file, movement_scenario: MovementScenario):
    DEVICE = 'cuda'
    width = RESOLUTION
    height = RESOLUTION

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
    render_object_poses(rendering, vertices, face_features, None, movement_scenario, optical_flow_destination,
                        rendering_destination, segmentation_destination, gt_tracking_log_file)


def generate_textured_sphere(config, rendering_destination: Path, segmentation_destination: Path,
                             optical_flow_destination, gt_tracking_log_file, movement_scenario: MovementScenario):
    prototype_path = Path('./prototypes/sphere.obj')
    tex_path = Path('./prototypes/tex.png')

    width = RESOLUTION
    height = RESOLUTION

    generate_rotating_and_translating_textured_object(config, movement_scenario, prototype_path, tex_path,
                                                      rendering_destination, segmentation_destination,
                                                      optical_flow_destination, gt_tracking_log_file, width, height)


def generate_translations_for_obj(obj_name, synthetic_dataset_folder_, rendering_dir_, segmentation_dir_,
                                  optical_flow_dir_, gt_tracking_log_dir_, movement_scenario):
    rendering_path = synthetic_dataset_folder_ / obj_name / rendering_dir_
    segmentation_path = synthetic_dataset_folder_ / obj_name / segmentation_dir_
    optical_flow_path = synthetic_dataset_folder_ / obj_name / optical_flow_dir_
    gt_tracking_log_file = synthetic_dataset_folder_ / obj_name / gt_tracking_log_dir_ / Path('gt_tracking_log.csv')

    generate_textured_sphere(_config, rendering_path, segmentation_path, optical_flow_path, gt_tracking_log_file,
                             movement_scenario)


def generate_rotating_objects():
    rot_mags = {2}
    for rot_mag in rot_mags:

        rot_gens = [
            (dataset_generators.scenarios.generate_rotations_y, 'y'),
            (dataset_generators.scenarios.generate_rotations_z, 'z'),
            (dataset_generators.scenarios.generate_rotations_xy, 'xy'),
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


def generate_translating_objects():
    nums_steps = {180}
    for num_steps in nums_steps:

        trans_gens = [
            (dataset_generators.scenarios.generate_in_depth_translations, 'in_depth'),
            (dataset_generators.scenarios.generate_circular_translation, 'in_plane'),
        ]

        for trans_gen, suffix in trans_gens:

            objects = [
                (f'Textured_Sphere_translations_{suffix}', generate_textured_sphere),
                (f'8_Colored_Sphere_translations_{suffix}', generate_8_colored_sphere),
                (f'6_Colored_Cube_translations_{suffix}', generate_6_colored_cube)
            ]

            movement_scenario = trans_gen(steps=num_steps)

            for obj_name, generate_obj_func in objects:
                rendering_path = synthetic_dataset_folder / obj_name / rendering_dir
                segmentation_path = synthetic_dataset_folder / obj_name / segmentation_dir
                optical_flow_path = synthetic_dataset_folder / obj_name / optical_flow_dir
                gt_tracking_log_file = synthetic_dataset_folder / obj_name / gt_tracking_log_dir / Path(
                    'gt_tracking_log.csv')

                generate_obj_func(_config, rendering_path, segmentation_path, optical_flow_path, gt_tracking_log_file,
                                  movement_scenario)


if __name__ == '__main__':
    _config = load_config('./configs/config_generator.yaml')

    synthetic_dataset_folder = dataset_folder / Path('SyntheticObjectsWorkshop')
    rendering_dir = Path('renderings')
    segmentation_dir = Path('segmentations')
    optical_flow_dir = Path('optical_flow')
    gt_tracking_log_dir = Path('gt_tracking_log')

    generate_rotating_objects()
    generate_translating_objects()

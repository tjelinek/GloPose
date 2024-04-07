import torch
import warnings
from pathlib import Path

import dataset_generators.scenarios
import dataset_generators.scenarios
from dataset_generators.generator_utils import prepare_scenarios_for_kubric, \
    generate_rotating_and_translating_textured_object
from main_settings import dataset_folder
from utils import load_config

warnings.filterwarnings("ignore")


def get_cube_vertices_and_faces():
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

    return vertices, faces


def generate_8_colored_sphere():
    prototype_path = Path('./prototypes/8-colored-sphere/8-colored-sphere.obj')
    tex_path = Path('./prototypes/8-colored-sphere/8-colored-sphere-tex.png')

    return prototype_path, tex_path


def generate_6_colored_cube():
    prototype_path = Path('./prototypes/6-colored-cube/6-colored-cube.obj')
    tex_path = Path('./prototypes/6-colored-cube/6-colored-cube-tex.png')

    return prototype_path, tex_path


def generate_textured_sphere():
    prototype_path = Path('./prototypes/textured-sphere/sphere.obj')
    tex_path = Path('./prototypes/textured-sphere/tex.png')

    return prototype_path, tex_path


def generate_textured_cube():
    prototype_path = Path('./prototypes/textured-cube/textured-cube.obj')
    tex_path = Path('./prototypes/textured-cube/tex.png')

    return prototype_path, tex_path


def generate_rotating_objects():
    rot_mags = {5, 10}

    for background_image_path in [None, 'dataset_generators/backgrounds/landscape_crop.jpg']:
        background_suffix = ''
        if background_image_path is not None:
            background_suffix = '_background'

        for rot_mag in rot_mags:

            rot_gens = [
                (dataset_generators.scenarios.generate_rotations_y, 'y'),
                (dataset_generators.scenarios.generate_rotations_x, 'x'),
                (dataset_generators.scenarios.generate_rotations_z, 'z'),
                (dataset_generators.scenarios.generate_rotations_xy, 'xy'),
            ]

            for rots_gen, suffix in rot_gens:

                objects = [
                    (f'Textured_Cube_{rot_mag}_{suffix}{background_suffix}', generate_textured_cube),
                    (f'Textured_Sphere_{rot_mag}_{suffix}{background_suffix}', generate_textured_sphere),
                    (f'8_Colored_Sphere_{rot_mag}_{suffix}{background_suffix}', generate_8_colored_sphere),
                    (f'6_Colored_Cube_{rot_mag}_{suffix}{background_suffix}', generate_6_colored_cube)
                ]

                movement_scenario = rots_gen(step=float(rot_mag))

                for obj_name, object_getter in objects:
                    prototype_path, texture_path = object_getter()
                    generate_object_using_function(movement_scenario, background_image_path,
                                                   obj_name, prototype_path, texture_path)


def generate_translating_objects():
    nums_steps = {180}
    for background_image_path in ['dataset_generators/backgrounds/landscape_crop.jpg', None]:
        background_suffix = ''
        if background_image_path is not None:
            background_suffix = '_background'

        for num_steps in nums_steps:

            trans_gens = [
                (dataset_generators.scenarios.generate_in_depth_translations, 'in_depth'),
                (dataset_generators.scenarios.generate_circular_translation, 'in_plane'),
            ]

            for trans_gen, suffix in trans_gens:

                objects = [
                    (f'Textured_Sphere_translations_{suffix}{background_suffix}', generate_textured_cube),
                    (f'Textured_Sphere_translations_{suffix}{background_suffix}', generate_textured_sphere),
                    (f'8_Colored_Sphere_translations_{suffix}{background_suffix}', generate_8_colored_sphere),
                    (f'6_Colored_Cube_translations_{suffix}{background_suffix}', generate_6_colored_cube)
                ]

                movement_scenario = trans_gen(steps=num_steps)

                for obj_name, object_getter in objects:
                    prototype_path, texture_path = object_getter()
                    generate_object_using_function(movement_scenario, background_image_path,
                                                   obj_name, prototype_path, texture_path)


def generate_object_using_function(movement_scenario, background_image_path, obj_name,
                                   prototype_path, texture_path):
    rendering_path = synthetic_dataset_folder / obj_name / rendering_dir
    segmentation_path = synthetic_dataset_folder / obj_name / segmentation_dir
    optical_flow_relative_path = synthetic_dataset_folder / obj_name / optical_flow_relative_dir
    optical_flow_absolute_path = synthetic_dataset_folder / obj_name / optical_flow_absolute_dir
    gt_tracking_log_file = synthetic_dataset_folder / obj_name / gt_tracking_log_dir / Path('gt_tracking_log.csv')

    width = 1000
    height = 1000

    if rendering_method == 'DIB-R':
        generate_rotating_and_translating_textured_object(config, movement_scenario, prototype_path, texture_path,
                                                          rendering_path, segmentation_path,
                                                          optical_flow_relative_path, optical_flow_absolute_path,
                                                          gt_tracking_log_file, width, height,
                                                          background_image_path=background_image_path)

    elif rendering_method == 'kubric':
        prepare_scenarios_for_kubric(config, movement_scenario, prototype_path, texture_path,
                                     rendering_path, segmentation_path,
                                     optical_flow_relative_path, optical_flow_absolute_path,
                                     gt_tracking_log_file, width, height,
                                     background_image_path=background_image_path)
    else:
        raise ValueError("\'rendering_method\' must be either \'kubric\' or \'DIB-R\'")


if __name__ == '__main__':
    config = load_config('./configs/config_deep.py')

    rendering_method = 'DIB-R'  # 'kubric' or 'DIB-R'
    synthetic_dataset_folder = dataset_folder / Path('SyntheticObjects')
    rendering_dir = Path('renderings')
    segmentation_dir = Path('segmentations')
    optical_flow_relative_dir = Path('optical_flow_relative')
    optical_flow_absolute_dir = Path('optical_flow_absolute')
    gt_tracking_log_dir = Path('gt_tracking_log')

    generate_rotating_objects()
    generate_translating_objects()

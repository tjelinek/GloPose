from pathlib import Path

import torch
from kornia.geometry import Se3, Quaternion

from dataset_generators import scenarios
from models.rendering import get_Se3_obj_to_cam_from_kaolin_params
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D
from utils.data_utils import load_texture, load_mesh
from utils.general import load_config


def main():
    dataset = 'SyntheticObjects'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'Textured_Sphere_5_y',
            'Textured_Cube_5_y',
            'Textured_Sphere_5_z',
            'Translating_Textured_Sphere',
            'Textured_Sphere_10_xy',
            'Rotating_Translating_Textured_Sphere_5_y',
            'Rotating_Translating_Textured_Sphere_5_xy',
            'Rotating_Contra_Translating_Textured_Sphere_5_y',
            'Rotating_Contra_Translating_Textured_Sphere_5_xy',
            '8_Colored_Sphere_5_x',
            '6_Colored_Cube_5_z']

    sequence = sequences[0]

    config = load_config(args.config)

    if '8_Colored_Sphere' in sequence:
        gt_mesh_path = Path('prototypes/8-colored-sphere/8-colored-sphere.obj')
        gt_texture_path = Path('prototypes/8-colored-sphere/8-colored-sphere-tex.png')
    elif 'Textured_Sphere' in sequence:
        gt_mesh_path = Path('prototypes/sphere.obj')
        gt_texture_path = Path('prototypes/tex.png')
    elif 'Textured_Cube' in sequence:
        gt_mesh_path = Path('prototypes/textured-cube/textured-cube.obj')
        gt_texture_path = Path('prototypes/textured-cube/tex.png')
    else:
        gt_texture_path = None
        gt_mesh_path = None

    experiment_name = args.experiment

    config.gt_texture_path = gt_texture_path
    config.gt_mesh_path = gt_mesh_path

    config.experiment_name = experiment_name
    config.sequence = sequence
    config.dataset = dataset
    config.purge_cache = True

    config.large_images_results_write_frequency = 1

    skip_frames = 1
    gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
    gt_mesh = load_mesh(Path(config.gt_mesh_path))
    gt_rotations = torch.deg2rad(scenarios.generate_rotations_y(step=5).rotations).to(config.device)
    # gt_rotations = torch.cat([gt_rotations, gt_rotations], dim=0)
    # gt_rotations = torch.deg2rad(scenarios.random_walk_on_a_sphere().rotations).to(torch.float32).to(config.device)
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]

    images_paths = images_paths[::skip_frames]
    gt_rotations = gt_rotations[::skip_frames]
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations * 0
    gt_translations = gt_translations.to(config.device)

    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    camera_trans = torch.FloatTensor(config.camera_position)[None].to(config.device)
    up = torch.FloatTensor(config.camera_up)[None].to(config.device)
    obj_center = torch.FloatTensor(config.obj_center)[None].to(config.device)

    gt_Se3_obj2cam = get_Se3_obj_to_cam_from_kaolin_params(camera_trans, up, obj_center)

    gt_Se3_cam2obj = Se3_obj_relative_to_Se3_cam2obj(gt_obj_1_to_obj_i_Se3, gt_Se3_obj2cam)

    if args.output_folder is not None:
        write_folder = Path(args.output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / sequence

    config.input_frames = gt_rotations.shape[0]

    tracker = Tracker6D(config, write_folder, gt_texture=gt_texture, gt_mesh=gt_mesh,
                        gt_Se3_cam2obj=gt_Se3_cam2obj, images_paths=images_paths)

    tracker.run_filtering_with_reconstruction()




if __name__ == "__main__":
    main()

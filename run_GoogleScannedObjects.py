import torch
from kornia.geometry import Se3, Quaternion

from dataset_generators import scenarios
from models.rendering import get_Se3_obj_to_cam_from_config
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj
from utils.runtime_utils import parse_args
from tracker6d import run_tracking_on_sequence
from utils.data_utils import load_mesh, load_texture
from utils.general import load_config
from pathlib import Path


def main():
    dataset = 'GoogleScannedObjects'
    args = parse_args()

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            # 'INTERNATIONAL_PAPER_Willamette_4_Brown_Bag_500Count',
            # 'Twinlab_Nitric_Fuel',
            'Squirrel',
            # 'STACKING_BEAR',
            # 'Schleich_Allosaurus',
            # 'Nestl_Skinny_Cow_Heavenly_Crisp_Candy_Bar_Chocolate_Raspberry_6_pack_462_oz_total',
            # 'SCHOOL_BUS',
            # 'Sootheze_Cold_Therapy_Elephant',
            # 'TOP_TEN_HI',
            'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
        ]

    sequence = sequences[0]

    config = load_config(args.config)

    config.camera_position = (0, -5.0, 0)
    config.camera_up = (0, 0, 1)

    gt_model_path = config.default_data_folder / Path(dataset) / Path('models') / Path(sequence)
    gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
    gt_mesh_path = gt_model_path / Path('meshes/model.obj')

    experiment_name = args.experiment

    config.gt_texture_path = gt_texture_path
    config.gt_mesh_path = gt_mesh_path

    config.experiment_name = experiment_name
    config.sequence = sequence
    config.dataset = dataset

    skip_frames = 1
    gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
    gt_mesh = load_mesh(Path(config.gt_mesh_path))
    gt_rotations = torch.deg2rad(scenarios.random_walk_on_a_sphere().rotations).to(torch.float32).to(config.device)
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]

    images_paths = images_paths[::skip_frames]
    gt_rotations = gt_rotations[::skip_frames]
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations
    gt_translations = gt_translations.to(config.device)

    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    gt_Se3_obj2cam = get_Se3_obj_to_cam_from_config(config)
    gt_Se3_cam2obj = Se3_obj_relative_to_Se3_cam2obj(gt_obj_1_to_obj_i_Se3, gt_Se3_obj2cam)

    if args.output_folder is not None:
        write_folder = Path(args.output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / sequence

    config.input_frames = gt_rotations.shape[0]

    run_tracking_on_sequence(config, write_folder, gt_texture=gt_texture, gt_mesh=gt_mesh,
                             gt_Se3_cam2obj=gt_Se3_cam2obj, images_paths=images_paths)


if __name__ == "__main__":
    main()

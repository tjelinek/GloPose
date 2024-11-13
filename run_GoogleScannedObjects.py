import torch

from dataset_generators import scenarios
from dataset_generators.track_augmentation import modify_rotations
from main_settings import tmp_folder, dataset_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from auxiliary_scripts.data_utils import load_mesh, load_texture
from utils import load_config
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
            'Sootheze_Cold_Therapy_Elephant',
            # 'TOP_TEN_HI',
            'Transformers_Age_of_Extinction_Mega_1Step_Bumblebee_Figure',
        ]

    for sequence in sequences:
        config = load_config(args.config)

        # config.camera_position = (3.14, -5.0, -2.81)
        config.camera_position = (0, -5.0, 0)
        config.camera_up = (0, 0, 1)

        gt_model_path = Path(dataset_folder) / Path(dataset) / Path('models') / Path(sequence)
        gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
        gt_mesh_path = gt_model_path / Path('meshes/model.obj')
        # gt_tracking_path = Path(dataset_folder) / Path(dataset) / Path('gt_tracking_log') / Path(sequence) / \
        #                    Path('gt_tracking_log.csv')

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = gt_mesh_path
        # config.gt_track_path = gt_tracking_path
        config.sequence = sequence

        gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
        gt_mesh = load_mesh(Path(config.gt_mesh_path))
        # gt_rotations = torch.deg2rad(scenarios.generate_rotations_z(5).rotations).cuda().to(torch.float32)
        gt_rotations = torch.deg2rad(scenarios.random_walk_on_a_sphere().rotations).cuda().to(torch.float32)[::4]
        gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations.cuda()

        if config.augment_gt_track:
            gt_rotations, gt_translations = modify_rotations(gt_rotations, gt_translations)

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence

        config.input_frames = gt_rotations.shape[0]

        run_tracking_on_sequence(config, write_folder, gt_texture, gt_mesh, gt_rotations, gt_translations)


if __name__ == "__main__":
    main()

from pathlib import Path
from dataset_generators import scenarios
from utils.experiment_runners import run_tracking
from utils.runtime_utils import parse_args
from utils.general import load_config


def main():
    dataset = 'GoogleScannedObjects'
    args = parse_args()

    # Load config first so we can modify dataset-specific parameters
    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
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

    # Set camera parameters specific to GoogleScannedObjects
    config.camera_position = (0, -5.0, 0)
    config.camera_up = (0, 0, 1)

    # Construct paths specific to GoogleScannedObjects
    gt_model_path = config.default_data_folder / Path(dataset) / Path('models') / Path(sequence)
    gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
    gt_mesh_path = gt_model_path / Path('meshes/model.obj')

    # Run tracking with z-axis rotations for GoogleScannedObjects
    run_tracking(
        config=config,
        dataset=dataset,
        sequence=sequence,
        experiment=args.experiment,
        output_folder=args.output_folder,
        gt_mesh_path=gt_mesh_path,
        gt_texture_path=gt_texture_path,
        rotation_generator=scenarios.generate_rotations_z
    )


if __name__ == "__main__":
    main()
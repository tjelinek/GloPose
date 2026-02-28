from pathlib import Path

from dataset_generators import scenarios
from utils.dataset_sequences import get_google_scanned_objects_sequences
from utils.experiment_runners import run_on_synthetic_data
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'GoogleScannedObjects'
    args = parse_args()

    # Load config first so we can modify dataset-specific parameters
    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = get_google_scanned_objects_sequences(
            config.paths.google_scanned_objects_data_folder / 'models')[:1]

    for sequence in sequences:
        with exception_logger(sequence):
            # Set camera parameters specific to GoogleScannedObjects
            config.renderer.camera_position = (0, -5.0, 0)
            config.renderer.camera_up = (0, 0, 1)

            # Construct paths specific to GoogleScannedObjects
            gt_model_path = config.paths.google_scanned_objects_data_folder / Path('models') / Path(sequence)
            gt_texture_path = gt_model_path / Path('materials/textures/texture.png')
            gt_mesh_path = gt_model_path / Path('meshes/model.obj')

            # Run tracking with z-axis rotations for GoogleScannedObjects
            run_on_synthetic_data(
                config=config,
                dataset=dataset,
                sequence=sequence,
                experiment=args.experiment,
                output_folder=args.output_folder,
                gt_mesh_path=gt_mesh_path,
                gt_texture_path=gt_texture_path,
                rotation_generator=scenarios.random_walk_on_a_sphere
            )


if __name__ == "__main__":
    main()

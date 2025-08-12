from pathlib import Path
from typing import Tuple, List

from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def get_handal_sequences() -> Tuple[List[Path], List[Path]]:
    dataset_path = Path("/mnt/personal/jelint19/data/HANDAL")

    train_sequences = []
    test_sequences = []

    for category_dir in dataset_path.iterdir():
        if not category_dir.is_dir():
            continue

        category_name = category_dir.name

        train_dir = category_dir / "train"
        if train_dir.exists():
            for sequence_dir in train_dir.iterdir():
                if sequence_dir.is_dir():
                    train_sequences.append(f"{category_name}/{sequence_dir.name}")

        test_dir = category_dir / "test"
        if test_dir.exists():
            for sequence_dir in test_dir.iterdir():
                if sequence_dir.is_dir():
                    test_sequences.append(f"{category_name}/{sequence_dir.name}")

    return train_sequences, test_sequences


HANDAL_TRAIN_SEQUENCES, HANDAL_TEST_SEQUENCES = get_handal_sequences()


def main():
    dataset = 'handal'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = HANDAL_TEST_SEQUENCES[4:5]

    for obj_type_sequence in sequences:
        with exception_logger():

            if obj_type_sequence in HANDAL_TRAIN_SEQUENCES:
                sequence_type = 'train'
            elif obj_type_sequence in HANDAL_TEST_SEQUENCES:
                sequence_type = 'test'
            else:
                raise ValueError(f"Unknown sequence {obj_type_sequence}")

            obj, sequence = obj_type_sequence.split('/')
            config = load_config(args.config)

            experiment_name = args.experiment
            output_folder = args.output_folder

            config.experiment_name = experiment_name
            config.sequence = sequence
            config.dataset = dataset
            config.image_downsample = 0.25
            config.large_images_results_write_frequency = 5

            config.skip_indices *= 1

            sequence_type = 'val'

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, output_folder, True)


if __name__ == "__main__":
    main()

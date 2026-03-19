import argparse

from utils.bop_challenge import set_config_for_bop_onboarding
from utils.dataset_sequences import get_hot3d_onboarding_sequences
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import exception_logger


def parse_hot3d_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, default="configs/base_config.py")
    parser.add_argument("--sequences", required=False, nargs='*', default=None)
    parser.add_argument("--output_folder", required=False)
    parser.add_argument("--experiment", required=False, default='default')
    parser.add_argument("--device", required=False, default='aria', choices=['aria', 'quest3'],
                        help="HOT3D camera device: 'aria' or 'quest3'")
    return parser.parse_args()


def main():
    dataset = 'hot3d'
    args = parse_hot3d_args()

    config = load_config(args.config)
    hot3d_dynamic, hot3d_static = get_hot3d_onboarding_sequences(
        config.paths.bop_data_folder, device=args.device)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (hot3d_static + hot3d_dynamic)[0:1]

    if not sequences:
        print(f"No HOT3D sequences found for device='{args.device}'. "
              f"Check that scenewise directories exist under {config.paths.bop_data_folder / 'hot3d'}:")
        print(f"  object_ref_{args.device}_static_scenewise/")
        print(f"  object_ref_{args.device}_dynamic_scenewise/")
        print("You may need to extract the tarballs first.")
        return

    print(f"Running {len(sequences)} HOT3D sequences (device={args.device}): {sequences[:5]}{'...' if len(sequences) > 5 else ''}")

    for sequence in sequences:
        with exception_logger(sequence):
            config = load_config(args.config)

            experiment_name = args.experiment

            config.run.experiment_name = experiment_name
            config.run.dataset = dataset
            config.input.image_downsample = .5
            config.input.hot3d_device = args.device

            config.input.depth_scale_to_meter = 0.001

            config.input.skip_indices *= 4

            output_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            sequence_type = 'onboarding'

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, output_folder)


if __name__ == "__main__":
    main()

from utils.bop_challenge import set_config_for_bop_onboarding
from utils.dataset_sequences import get_hot3d_onboarding_sequences
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'hot3d'
    args = parse_args()

    config = load_config(args.config)
    hot3d_dynamic, hot3d_static = get_hot3d_onboarding_sequences(
        config.paths.bop_data_folder)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (hot3d_static + hot3d_dynamic)[0:1]

    for sequence in sequences:
        with exception_logger(sequence):
            config = load_config(args.config)

            experiment_name = args.experiment

            config.run.experiment_name = experiment_name
            config.run.dataset = dataset
            config.input.image_downsample = .5

            config.input.depth_scale_to_meter = 0.001

            config.input.skip_indices *= 1

            output_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            sequence_type = 'onboarding'

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, output_folder)


if __name__ == "__main__":
    main()

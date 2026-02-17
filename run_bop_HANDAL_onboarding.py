from utils.bop_challenge import set_config_for_bop_onboarding
from utils.dataset_sequences import get_bop_onboarding_sequences
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'handal'
    args = parse_args()

    config = load_config(args.config)
    handal_dynamic, handal_up, handal_down, handal_both = get_bop_onboarding_sequences(
        config.default_data_folder / 'bop', dataset)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (handal_both + handal_up + handal_down + handal_dynamic)[31:32]

    for sequence in sequences:

        with exception_logger():
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.sequence = sequence
            config.dataset = dataset
            config.image_downsample = .5
            config.large_images_results_write_frequency = 4
            config.bop_config.onboarding_type = 'static'
            config.bop_config.static_onboarding_sequence = 'down'
            config.depth_scale_to_meter = 0.001

            config.skip_indices *= 1

            output_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            sequence_type = 'onboarding'

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, output_folder)


if __name__ == "__main__":
    main()

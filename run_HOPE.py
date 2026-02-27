from utils.bop_challenge import set_config_for_bop_onboarding
from utils.dataset_sequences import get_bop_onboarding_sequences
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'hope'
    args = parse_args()

    config = load_config(args.config)
    hope_dynamic, hope_up, hope_down, hope_both = get_bop_onboarding_sequences(
        config.bop_data_folder, dataset)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (hope_both + hope_up + hope_down + hope_dynamic)[4:5]

    for sequence in sequences:

        with exception_logger():
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.dataset = dataset
            config.image_downsample = .5

            config.depth_scale_to_meter = 0.001

            config.skip_indices *= 1

            sequence_type = 'onboarding'

            write_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, write_folder)


if __name__ == "__main__":
    main()

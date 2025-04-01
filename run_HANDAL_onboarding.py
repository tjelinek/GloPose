from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args


def main():
    dataset = 'handal'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'obj_000001',
            'obj_000010',
            'obj_000020',
            'obj_000030',
            'obj_000040',
        ][:1]

    for sequence in sequences:
        config = load_config(args.config)

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.25
        config.large_images_results_write_frequency = 2
        config.bop_config.static_onboarding_sequence = 'down'
        skip_indices = 4

        sequence_type = 'onboarding'
        onboarding_type = 'static'

        run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, skip_indices,
                             onboarding_type, True)


if __name__ == "__main__":
    main()

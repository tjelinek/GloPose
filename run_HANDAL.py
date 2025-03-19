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
            '000001',
            '000002',
            '000003',
            '000004',
            '000005',
        ]

    for sequence in sequences:
        config = load_config(args.config)

        if config.gt_flow_source == 'GenerateSynthetic':
            exit()

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5
        config.large_images_results_write_frequency = 4

        skip_indices = 1

        sequence_type = 'val'
        onboarding_type = None

        run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, skip_indices,
                             onboarding_type)


if __name__ == "__main__":
    main()

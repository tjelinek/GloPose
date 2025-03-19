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
            'obj_000001_down',
            'obj_000010_up',
            'obj_000020_down',
            'obj_000030_up',
            'obj_000040_down',
        ]

    for sequence in sequences:
        config = load_config(args.config)


        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5

        skip_indices = 4

        sequence_type = 'onboarding'
        onboarding_type = 'dynamic'

        run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, skip_indices,
                             onboarding_type)


if __name__ == "__main__":
    main()

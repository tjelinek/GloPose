from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args


def main():
    dataset = 'hope'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'obj_000001', 'obj_000006', 'obj_000011', 'obj_000016', 'obj_000021', 'obj_000026',
            'obj_000002', 'obj_000007', 'obj_000012', 'obj_000017', 'obj_000022', 'obj_000027',
            'obj_000003', 'obj_000008', 'obj_000013', 'obj_000018', 'obj_000023', 'obj_000028',
            'obj_000004', 'obj_000009', 'obj_000014', 'obj_000019', 'obj_000024',
            'obj_000005', 'obj_000010', 'obj_000015', 'obj_000020', 'obj_000025',
        ][:1]

    for sequence in sequences:
        config = load_config(args.config)

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5
        config.large_images_results_write_frequency = 5
        config.similarity_transformation = 'depths'

        if config.per_dataset_skip_indices:
            config.skip_indices = 1

        sequence_type = 'onboarding'
        onboarding_type = config.bop_config.onboarding_type

        run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, onboarding_type)


if __name__ == "__main__":
    main()

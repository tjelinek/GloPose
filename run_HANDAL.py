from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'handal'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            '000001_000000', '000001_000001', '000001_000002', '000001_000003', '000001_000004',
            '000002_000000', '000002_000001', '000002_000002', '000002_000003', '000002_000004',
            '000002_000005',
            '000003_000000', '000003_000001', '000003_000002', '000003_000003', '000003_000004',
            '000004_000000', '000004_000001', '000004_000002', '000004_000003', '000004_000004',
            '000004_000005',
            '000005_000000', '000005_000001', '000005_000002', '000005_000003', '000005_000004',
            '000006_000000', '000006_000001', '000006_000002', '000006_000003', '000006_000004',
            '000006_000005',
            '000007_000000', '000007_000001', '000007_000002', '000007_000003', '000007_000004',
            '000007_000005',
            '000008_000000', '000008_000001', '000008_000002', '000008_000003', '000008_000004',
            '000009_000000', '000009_000001', '000009_000002', '000009_000003', '000009_000004',
            '000010_000000', '000010_000001', '000010_000002', '000010_000003', '000010_000004',
            '000010_000005',
                    ][4:5]

    for sequence_obj in sequences:

        with exception_logger():

            sequence, obj = sequence_obj.split('_')
            obj_id = int(obj)
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.sequence = sequence_obj
            config.dataset = dataset
            config.image_downsample = 0.25
            config.large_images_results_write_frequency = 5

            if config.per_dataset_skip_indices:
                config.skip_indices = 1

            sequence_type = 'val'

            run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, 1.0, True,
                                 scene_obj_id=obj_id)


if __name__ == "__main__":
    main()

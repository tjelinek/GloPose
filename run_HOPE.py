from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'hope'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = [
            'obj_000001_dynamic', 'obj_000006_dynamic', 'obj_000011_dynamic', 'obj_000016_dynamic',
            'obj_000021_dynamic', 'obj_000026_dynamic', 'obj_000002_dynamic', 'obj_000007_dynamic',
            'obj_000012_dynamic', 'obj_000017_dynamic', 'obj_000022_dynamic', 'obj_000027_dynamic',
            'obj_000003_dynamic', 'obj_000008_dynamic', 'obj_000013_dynamic', 'obj_000018_dynamic',
            'obj_000023_dynamic', 'obj_000028_dynamic', 'obj_000004_dynamic', 'obj_000009_dynamic',
            'obj_000014_dynamic', 'obj_000019_dynamic', 'obj_000024_dynamic', 'obj_000005_dynamic',
            'obj_000010_dynamic', 'obj_000015_dynamic', 'obj_000020_dynamic', 'obj_000025_dynamic',
            'obj_000001_up', 'obj_000006_up', 'obj_000011_up', 'obj_000016_up',
            'obj_000021_up', 'obj_000026_up', 'obj_000002_up', 'obj_000007_up',
            'obj_000012_up', 'obj_000017_up', 'obj_000022_up', 'obj_000027_up',
            'obj_000003_up', 'obj_000008_up', 'obj_000013_up', 'obj_000018_up',
            'obj_000023_up', 'obj_000028_up', 'obj_000004_up', 'obj_000009_up',
            'obj_000014_up', 'obj_000019_up', 'obj_000024_up', 'obj_000005_up',
            'obj_000010_up', 'obj_000015_up', 'obj_000020_up', 'obj_000025_up',
            'obj_000001_down', 'obj_000006_down', 'obj_000011_down', 'obj_000016_down',
            'obj_000021_down', 'obj_000026_down', 'obj_000002_down', 'obj_000007_down',
            'obj_000012_down', 'obj_000017_down', 'obj_000022_down', 'obj_000027_down',
            'obj_000003_down', 'obj_000008_down', 'obj_000013_down', 'obj_000018_down',
            'obj_000023_down', 'obj_000028_down', 'obj_000004_down', 'obj_000009_down',
            'obj_000014_down', 'obj_000019_down', 'obj_000024_down', 'obj_000005_down',
            'obj_000010_down', 'obj_000015_down', 'obj_000020_down', 'obj_000025_down',
        ][:1]

    for sequence in sequences:

        with exception_logger():
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
            run_only_on_frames_with_known_pose = True

            sequence_name_split = sequence.split('_')
            if len(sequence_name_split) == 3:
                if sequence_name_split[2] == 'down':
                    config.bop_config.onboarding_type = 'static'
                    config.bop_config.static_onboarding_sequence = 'down'
                elif sequence_name_split[2] == 'up':
                    config.bop_config.onboarding_type = 'static'
                    config.bop_config.static_onboarding_sequence = 'up'
                elif sequence_name_split[2] == 'dynamic':
                    config.bop_config.onboarding_type = 'dynamic'
                    run_only_on_frames_with_known_pose = False
                    sequence = '_'.join(sequence_name_split[:2])
                    config.skip_indices = 4

            run_on_bop_sequences(dataset, experiment_name, sequence, sequence_type, args, config, 1.0,
                                 only_frames_with_known_poses=run_only_on_frames_with_known_pose)


if __name__ == "__main__":
    main()

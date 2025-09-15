from utils.bop_challenge import set_config_for_bop_onboarding
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


HOPE_ONBOARDING_DYNAMIC_SEQUENCES = [
    'obj_000001_dynamic', 'obj_000002_dynamic', 'obj_000003_dynamic', 'obj_000004_dynamic',
    'obj_000005_dynamic', 'obj_000006_dynamic', 'obj_000007_dynamic', 'obj_000008_dynamic',
    'obj_000009_dynamic', 'obj_000010_dynamic', 'obj_000011_dynamic', 'obj_000012_dynamic',
    'obj_000013_dynamic', 'obj_000014_dynamic', 'obj_000015_dynamic', 'obj_000016_dynamic',
    'obj_000017_dynamic', 'obj_000018_dynamic', 'obj_000019_dynamic', 'obj_000020_dynamic',
    'obj_000021_dynamic', 'obj_000022_dynamic', 'obj_000023_dynamic', 'obj_000024_dynamic',
    'obj_000025_dynamic', 'obj_000026_dynamic', 'obj_000027_dynamic', 'obj_000028_dynamic',
]

HOPE_ONBOARDING_STATIC_UP_SEQUENCES = [
    'obj_000001_up', 'obj_000002_up', 'obj_000003_up', 'obj_000004_up',
    'obj_000005_up', 'obj_000006_up', 'obj_000007_up', 'obj_000008_up',
    'obj_000009_up', 'obj_000010_up', 'obj_000011_up', 'obj_000012_up',
    'obj_000013_up', 'obj_000014_up', 'obj_000015_up', 'obj_000016_up',
    'obj_000017_up', 'obj_000018_up', 'obj_000019_up', 'obj_000020_up',
    'obj_000021_up', 'obj_000022_up', 'obj_000023_up', 'obj_000024_up',
    'obj_000025_up', 'obj_000026_up', 'obj_000027_up', 'obj_000028_up',
]

HOPE_ONBOARDING_STATIC_DOWN_SEQUENCES = [
    'obj_000001_down', 'obj_000002_down', 'obj_000003_down', 'obj_000004_down',
    'obj_000005_down', 'obj_000006_down', 'obj_000007_down', 'obj_000008_down',
    'obj_000009_down', 'obj_000010_down', 'obj_000011_down', 'obj_000012_down',
    'obj_000013_down', 'obj_000014_down', 'obj_000015_down', 'obj_000016_down',
    'obj_000017_down', 'obj_000018_down', 'obj_000019_down', 'obj_000020_down',
    'obj_000021_down', 'obj_000022_down', 'obj_000023_down', 'obj_000024_down',
    'obj_000025_down', 'obj_000026_down', 'obj_000027_down', 'obj_000028_down',
]

HOPE_ONBOARDING_STATIC_BOTH_SIDES_SEQUENCES = [
    'obj_000001_both', 'obj_000002_both', 'obj_000003_both', 'obj_000004_both',
    'obj_000005_both', 'obj_000006_both', 'obj_000007_both', 'obj_000008_both',
    'obj_000009_both', 'obj_000010_both', 'obj_000011_both', 'obj_000012_both',
    'obj_000013_both', 'obj_000014_both', 'obj_000015_both', 'obj_000016_both',
    'obj_000017_both', 'obj_000018_both', 'obj_000019_both', 'obj_000020_both',
    'obj_000021_both', 'obj_000022_both', 'obj_000023_both', 'obj_000024_both',
    'obj_000025_both', 'obj_000026_both', 'obj_000027_both', 'obj_000028_both',
]


def main():
    dataset = 'hope'
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (HOPE_ONBOARDING_STATIC_BOTH_SIDES_SEQUENCES + HOPE_ONBOARDING_STATIC_UP_SEQUENCES +
                     HOPE_ONBOARDING_STATIC_DOWN_SEQUENCES + HOPE_ONBOARDING_DYNAMIC_SEQUENCES)[4:5]

    for sequence in sequences:

        with exception_logger():
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.dataset = dataset
            config.image_downsample = .5
            config.large_images_results_write_frequency = 5
            config.depth_scale_to_meter = 0.001

            config.skip_indices *= 1

            sequence_type = 'onboarding'

            write_folder = args.output_folder

            set_config_for_bop_onboarding(config, sequence)

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, write_folder)


if __name__ == "__main__":
    main()

from pathlib import Path

from data_providers.frame_provider import PrecomputedSegmentationProvider
from tracker6d import Tracker6D
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
                                read_object_id, read_static_onboarding_world2cam, add_extrinsics_to_pinhole_params, \
                                read_pinhole_params
from utils.experiment_runners import reindex_frame_dict
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


BOP_TLESS_ONBOARDING_SEQUENCES = [
    "tless@train_primesense@000001", "tless@train_primesense@000002", "tless@train_primesense@000003",
    "tless@train_primesense@000004", "tless@train_primesense@000005", "tless@train_primesense@000006",
    "tless@train_primesense@000007", "tless@train_primesense@000008", "tless@train_primesense@000009",
    "tless@train_primesense@000010", "tless@train_primesense@000011", "tless@train_primesense@000012",
    "tless@train_primesense@000013", "tless@train_primesense@000014", "tless@train_primesense@000015",
    "tless@train_primesense@000016", "tless@train_primesense@000017", "tless@train_primesense@000018",
    "tless@train_primesense@000019", "tless@train_primesense@000020", "tless@train_primesense@000021",
    "tless@train_primesense@000022", "tless@train_primesense@000023", "tless@train_primesense@000024",
    "tless@train_primesense@000025", "tless@train_primesense@000026", "tless@train_primesense@000027",
    "tless@train_primesense@000028", "tless@train_primesense@000029", "tless@train_primesense@000030",
]

BOP_LMO_ONBOARDING_SEQUENCES = [
    "lmo@train@000001", "lmo@train@000005", "lmo@train@000006",
    "lmo@train@000008", "lmo@train@000009", "lmo@train@000010",
    "lmo@train@000011", "lmo@train@000012",
]

BOP_ICBIN_ONBOARDING_SEQUENCES = [
    "icbin@train@000001", "icbin@train@000002"
]


def main():
    args = parse_args()
    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = (BOP_TLESS_ONBOARDING_SEQUENCES + BOP_LMO_ONBOARDING_SEQUENCES +
                     BOP_ICBIN_ONBOARDING_SEQUENCES)[0:1]

    for sequence_code in sequences:
        sequence_code_split = sequence_code.split('@')
        dataset, onboarding_folder, sequence_name = sequence_code_split[0]

        with exception_logger():
            config = load_config(args.config)

            experiment_name = args.experiment

            config.experiment_name = experiment_name
            config.dataset = dataset
            config.sequence = sequence_name
            config.image_downsample = .5
            config.large_images_results_write_frequency = 5
            config.depth_scale_to_meter = 0.001
            config.skip_indices *= 1

            sequence_type = 'onboarding'

            # Path to BOP dataset
            bop_folder = config.default_data_folder / 'bop'
            # Determine output folder
            if args.output_folder is not None:
                folder = Path(args.output_folder) / dataset / f'{sequence_name}'
            else:
                folder = config.default_results_folder / experiment_name / dataset / f'{sequence_name}'

            set_config_for_bop_onboarding(config, sequence)

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, write_folder)



if __name__ == "__main__":
    main()

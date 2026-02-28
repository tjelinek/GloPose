from utils.dataset_sequences import get_bop_val_sequences
from utils.experiment_runners import run_on_bop_sequences
from utils.general import load_config
from utils.runtime_utils import parse_args, exception_logger


def main():
    dataset = 'handal'
    args = parse_args()

    config = load_config(args.config)

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = get_bop_val_sequences(config.paths.bop_data_folder / dataset / 'val')[4:5]

    for sequence_obj in sequences:
        with exception_logger():
            sequence, obj = sequence_obj.split('_')
            obj_id = int(obj)
            config = load_config(args.config)

            experiment_name = args.experiment
            output_folder = args.output_folder

            config.run.experiment_name = experiment_name
            config.run.sequence = sequence
            config.run.dataset = dataset
            config.input.image_downsample = .5

            config.input.run_only_on_frames_with_known_pose = True
            config.input.skip_indices *= 1

            sequence_type = 'val'

            run_on_bop_sequences(dataset, experiment_name, sequence_type, config, 1.0, output_folder,
                                 scene_obj_id=obj_id)


if __name__ == "__main__":
    main()

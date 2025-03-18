import time
from pathlib import Path

from utils.data_utils import get_initial_image_and_segment
from utils.bop_challenge import get_pinhole_params, load_gt_images_and_segmentations, extract_gt_Se3_cam2obj
from utils.general import load_config
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D


def main():
    dataset = 'HANDAL'
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

        if config.gt_flow_source == 'GenerateSynthetic':
            exit()

        experiment_name = args.experiment

        config.experiment_name = experiment_name
        config.sequence = sequence
        config.dataset = dataset
        config.image_downsample = 0.5

        skip_indices = 4

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence
        else:
            write_folder = config.default_results_folder / experiment_name / dataset / sequence

        t0 = time.time()

        sequence_folder = config.default_data_folder / 'bop' / 'handal' / 'static' / sequence
        image_folder = sequence_folder / 'rgb'
        segmentation_folder = sequence_folder / 'mask_visib'

        gt_images, gt_segs = load_gt_images_and_segmentations(image_folder, segmentation_folder)

        pose_json_path = sequence_folder / 'scene_gt.json'

        dict_gt_Se3_cam2obj = extract_gt_Se3_cam2obj(pose_json_path, None, config.device)
        gt_Se3_obj2cam_frame0 = dict_gt_Se3_cam2obj[min(dict_gt_Se3_cam2obj.keys())]

        valid_indices = sorted(list(dict_gt_Se3_cam2obj.keys()))[::skip_indices]
        gt_images = [gt_images[i] for i in valid_indices]
        gt_segs = [gt_segs[i] for i in valid_indices]
        dict_gt_Se3_cam2obj = {i: dict_gt_Se3_cam2obj[frame] for i, frame in enumerate(valid_indices)}

        first_image, first_segmentation = get_initial_image_and_segment(gt_images, gt_segs, segmentation_channel=0)

        pinhole_params = get_pinhole_params(sequence_folder / 'scene_camera.json')

        config.camera_intrinsics = pinhole_params[min(valid_indices)].intrinsics.squeeze().numpy(force=True)
        config.camera_extrinsics = pinhole_params[min(valid_indices)].extrinsics.squeeze().numpy(force=True)
        config.input_frames = len(gt_images)
        config.frame_provider = 'precomputed'
        config.segmentation_provider = 'SAM2'

        print('Data loading took {:.2f} seconds'.format((time.time() - t0) / 1))

        sfb = Tracker6D(config, write_folder, initial_gt_Se3_cam2obj=gt_Se3_obj2cam_frame0,
                        gt_Se3_cam2obj=dict_gt_Se3_cam2obj, images_paths=gt_images, segmentation_paths=gt_segs,
                        initial_segmentation=first_segmentation,
                        initial_image=first_image)
        sfb.run_filtering_with_reconstruction()

        exit()


if __name__ == "__main__":
    main()

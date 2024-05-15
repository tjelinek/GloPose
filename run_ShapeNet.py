import sys

import numpy as np
import torch
from kaolin.io.shapenet import ShapeNetV2

from auxiliary_scripts.data_utils import load_texture
from dataset_generators import scenarios
from main_settings import tmp_folder
from runtime_utils import run_tracking_on_sequence, parse_args
from utils import load_config
from pathlib import Path

sys.path.append('repositories/OSTrack/S2DNet')


def main():
    dataset = 'ShapeNet'
    args = parse_args()

    if args.sequences is not None and len(args.sequences) > 0:
        sequences = args.sequences
    else:
        sequences = range(10)

    # shapenet_categories = ['02773838', '02828884', '02843684', '02871439', '02876657', '02924116', '03742115',
    #                        '02958343', '02738535', '03691459', '04099429', '04530566', '04591713']

    shapenet_path = Path('/mnt/personal/jelint19/data/ShapeNetCore.v2/')
    dataset_shapenet = ShapeNetV2(str(shapenet_path), split=1.)

    config = load_config(args.config)
    config.camera_position = (0, 0, 5.0)

    for sequence in sequences:

        dataset_item = dataset_shapenet[sequence]
        annotation = dataset_shapenet.get_attributes(sequence)

        gt_mesh_path = Path(annotation['path'])
        gt_texture_path = gt_mesh_path.parent.parent / 'images/texture0.jpg'

        gt_rotations_np = np.stack(scenarios.generate_rotations_yz(5).rotations, axis=0)
        gt_rotations = torch.from_numpy(gt_rotations_np).unsqueeze(0).cuda()
        gt_translations = torch.zeros_like(gt_rotations).unsqueeze(0)

        gt_mesh = dataset_item.data
        try:  # TODO more complex objects have more complex texture
            gt_texture = load_texture(gt_texture_path, config.texture_size).to(torch.float32)
        except:
            continue

        experiment_name = args.experiment

        config.input_frames = gt_rotations.shape[1]
        config.gt_texture_path = gt_texture_path
        config.gt_mesh_path = annotation['path']

        sequence_name = annotation['labels'][0].replace(' ', '_') + '_' + annotation['name'].split('/')[1]
        config.sequence = sequence_name

        if args.output_folder is not None:
            write_folder = Path(args.output_folder) / dataset / sequence_name
        else:
            write_folder = Path(tmp_folder) / experiment_name / dataset / sequence_name

        run_tracking_on_sequence(config, write_folder, gt_texture, gt_mesh, gt_rotations, gt_translations)


if __name__ == "__main__":
    main()

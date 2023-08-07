from collections import namedtuple

import torch
import warnings
import types
import os
import shutil
from pathlib import Path

from dataset_generators.generator_utils import generate_rotating_textured_object, generate_rotations_x, \
    generate_rotations_z
from utils import load_config, deg_to_rad
from main_settings import dataset_folder

warnings.filterwarnings("ignore")


def dataset_from_google_research(config, dataset_path: Path, rotations, initial_rotation_axis_angle,
                                 initial_translation):
    models_path = dataset_path / 'models'
    for file in os.listdir(models_path):
        d = models_path / file
        if os.path.isdir(d):
            if file == 'original' or file == 'masks_U2Net':
                continue
            texture_path = d / Path('materials') / Path('textures') / Path('texture.png')
            model_path = d / Path('meshes') / Path('model.mtl')
            mesh_path = d / Path('meshes') / Path('model.obj')

            shutil.copyfile(texture_path, model_path.parent / texture_path.name)

            segmentation_destination = dataset_path / Path('segmentations') / file
            rendering_destination = dataset_path / Path('renderings') / file
            optical_flow_destination = dataset_path / Path('optical_flow') / file
            gt_tracking_log_file = dataset_path / Path('gt_tracking_log') / file / Path('gt_tracking_log.csv')

            width = 1000
            height = 1000

            generate_rotating_textured_object(config, mesh_path, texture_path, rendering_destination,
                                              segmentation_destination, optical_flow_destination, gt_tracking_log_file,
                                              width, height, initial_rotation_axis_angle=initial_rotation_axis_angle,
                                              rotations=rotations,
                                              initial_translation=initial_translation)


if __name__ == '__main__':
    config = load_config('configs/config_deep.yaml')
    config = types.SimpleNamespace(**config)

    GeneratorConfig = namedtuple('GeneratorConfig',
                                 ['initial_rotation_axis_angle', 'initial_translation', 'rotations',
                                  'name'])

    configurations = {
        'GoogleScannedObjects':
            GeneratorConfig(initial_rotation_axis_angle=deg_to_rad(torch.Tensor([-90, 0, 0]).to('cuda')),
                            initial_translation=torch.Tensor([0, 0, 0]).to('cuda'),
                            rotations=generate_rotations_z(5.0), name='GoogleScannedObjects'),
        'GoogleScannedObjects_default_pose':
            GeneratorConfig(initial_rotation_axis_angle=deg_to_rad(torch.Tensor([0, 0, 0]).to('cuda')),
                            initial_translation=torch.Tensor([0, 0, 0]).to('cuda'),
                            rotations=generate_rotations_x(5.0), name='GoogleScannedObjects_default_pose'),
    }

    for gen_cfg_name, gen_cfg in configurations.items():
        dataset_folder = dataset_folder / Path(gen_cfg_name)

        dataset_from_google_research(config, dataset_folder, gen_cfg.rotations, gen_cfg.initial_rotation_axis_angle,
                                     gen_cfg.initial_translation)

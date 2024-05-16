import numpy as np
import warnings
import os
import shutil
from pathlib import Path

from dataclasses import replace
from dataset_generators.generator_utils import (generate_rotating_and_translating_textured_object,
                                                prepare_scenarios_for_kubric)
from dataset_generators.scenarios import generate_rotations_z
from utils import load_config
from main_settings import dataset_folder

warnings.filterwarnings("ignore")


def dataset_from_google_research(config, dataset_path: Path, movement_scenario, rendering_method='DIB-R'):
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
            optical_flow_relative_destination = dataset_path / Path('optical_flow_relative') / file
            optical_flow_absolute_destination = dataset_path / Path('optical_flow_absolute') / file
            gt_tracking_log_file = dataset_path / Path('gt_tracking_log') / file / Path('gt_tracking_log.csv')

            width = 1000
            height = 1000

            if rendering_method == 'DIB-R':
                generate_rotating_and_translating_textured_object(config, movement_scenario, mesh_path, texture_path,
                                                                  rendering_destination, segmentation_destination,
                                                                  optical_flow_relative_destination,
                                                                  optical_flow_absolute_destination,
                                                                  gt_tracking_log_file, width, height)

            elif rendering_method == 'kubric':
                prepare_scenarios_for_kubric(config, movement_scenario, mesh_path, texture_path,
                                             rendering_destination, segmentation_destination,
                                             optical_flow_relative_destination, optical_flow_absolute_destination,
                                             gt_tracking_log_file, width, height)


if __name__ == '__main__':
    config = load_config('configs/config_deep.py')

    configurations = {
        'GoogleScannedObjects': replace(generate_rotations_z(5.0), initial_rotation=np.asarray([0.0, 0.0, 0.0])),
    }

    for gen_cfg_name, movement_scenario in configurations.items():
        generated_data_folder = dataset_folder / Path(gen_cfg_name)

        dataset_from_google_research(config, generated_data_folder, movement_scenario, rendering_method='DIB-R')

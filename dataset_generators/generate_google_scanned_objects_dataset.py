import warnings
import types
import os
import shutil
from pathlib import Path

from dataset_generators.generator_utils import generate_rotating_textured_object
from utils import load_config
from main_settings import dataset_folder


warnings.filterwarnings("ignore")


def dataset_from_google_research(config, dataset_path: Path):
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

            width = 1506
            height = 2000

            generate_rotating_textured_object(config, mesh_path, texture_path, rendering_destination,
                                              segmentation_destination, optical_flow_destination, gt_tracking_log_file,
                                              width, height)


if __name__ == '__main__':
    config = load_config('configs/config_deep.yaml')
    config = types.SimpleNamespace(**config)

    dataset_folder = dataset_folder / Path('GoogleScannedObjects')

    dataset_from_google_research(config, dataset_folder)


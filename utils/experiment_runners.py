from pathlib import Path
import torch
from kornia.geometry import Se3, Quaternion

from dataset_generators import scenarios
from models.rendering import get_Se3_obj_to_cam_from_kaolin_params
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj
from utils.runtime_utils import parse_args
from tracker6d import Tracker6D
from utils.data_utils import load_texture, load_mesh
from utils.general import load_config


def run_tracking(config, dataset, sequence, experiment=None, output_folder=None,
                 gt_mesh_path=None, gt_texture_path=None, rotation_generator=None, skip_frames=1):
    """
    Common function to run 6D tracking across different datasets

    Parameters:
    -----------
    config : object
        Configuration object that has already been loaded and customized
    dataset : str
        Name of the dataset (e.g., 'SyntheticObjects', 'GoogleScannedObjects')
    sequence : str
        Specific sequence to run tracking on
    experiment : str, optional
        Experiment name
    output_folder : str, optional
        Output folder to save results
    gt_mesh_path : Path, optional
        Path to the ground truth mesh
    gt_texture_path : Path, optional
        Path to the ground truth texture
    rotation_generator : function, optional
        Function to generate rotations (defaults to y-axis rotations)
    skip_frames : int, optional
        Number of frames to skip in the sequence (defaults to 1 - no skipping)
    """
    # Use provided experiment name or get from config if available
    if experiment is None and hasattr(config, 'experiment_name'):
        experiment = config.experiment_name

    # Update configuration
    config.experiment_name = experiment
    config.sequence = sequence
    config.dataset = dataset
    config.purge_cache = True
    config.large_images_results_write_frequency = 1

    # Set mesh and texture paths
    config.gt_texture_path = gt_texture_path
    config.gt_mesh_path = gt_mesh_path

    # Load mesh and texture
    gt_texture = load_texture(Path(config.gt_texture_path), config.texture_size)
    gt_mesh = load_mesh(Path(config.gt_mesh_path))

    # Generate rotations
    if rotation_generator is None:
        rotation_generator = scenarios.generate_rotations_y

    gt_rotations = torch.deg2rad(rotation_generator(step=5).rotations).to(config.device)
    gt_rotations = gt_rotations[::skip_frames]

    # Create image paths
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]
    images_paths = images_paths[::skip_frames]

    # Generate translations (zero by default)
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations * 0
    gt_translations = gt_translations.to(config.device)

    # Create Se3 transformations
    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    # Set up camera parameters
    camera_trans = torch.FloatTensor(config.camera_position)[None].to(config.device)
    up = torch.FloatTensor(config.camera_up)[None].to(config.device)
    obj_center = torch.FloatTensor(config.obj_center)[None].to(config.device)

    gt_Se3_obj2cam = get_Se3_obj_to_cam_from_kaolin_params(camera_trans, up, obj_center)
    gt_Se3_cam2obj = Se3_obj_relative_to_Se3_cam2obj(gt_obj_1_to_obj_i_Se3, gt_Se3_obj2cam)

    # Set up output folder
    if output_folder is not None:
        write_folder = Path(output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment / dataset / sequence

    # Set number of input frames
    config.input_frames = gt_rotations.shape[0]

    # Create and run tracker
    tracker = Tracker6D(config, write_folder, gt_texture=gt_texture, gt_mesh=gt_mesh,
                        gt_Se3_cam2obj=gt_Se3_cam2obj, images_paths=images_paths)

    tracker.run_filtering_with_reconstruction()

    return tracker
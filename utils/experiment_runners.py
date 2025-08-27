import hashlib
from pathlib import Path
from typing import Dict, List, Any

import torch
from kornia.geometry import Se3, Quaternion

from dataset_generators import scenarios
from tracker_config import TrackerConfig
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
    read_pinhole_params, read_static_onboarding_world2cam, add_extrinsics_to_pinhole_params, read_object_id
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj
from tracker6d import Tracker6D
from utils.data_utils import load_texture, load_mesh
from data_providers.frame_provider import PrecomputedSegmentationProvider


def run_on_synthetic_data(config: TrackerConfig, dataset: str, sequence: str, experiment=None, output_folder=None,
                          gt_mesh_path: Path = None, gt_texture_path: Path = None, rotation_generator=None):
    """
    Common function to run 6D tracking across different datasets

    Parameters:
    -----------
    config : TrackerConfig
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
    from models.rendering import get_Se3_obj2cam_from_kaolin_params

    # Use provided experiment name or get from config if available
    if experiment is None and hasattr(config, 'experiment_name'):
        experiment = config.experiment_name

    # Update configuration
    config.experiment_name = experiment
    config.sequence = sequence
    config.dataset = dataset
    config.purge_cache = False
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
    gt_rotations = torch.cat([gt_rotations, gt_rotations], dim=0)

    # Create image paths
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]

    # Generate translations (zero by default)
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations * 0
    gt_translations = gt_translations.to(config.device)

    trace_hash = hashlib.md5(gt_rotations.numpy(force=True).tobytes() + gt_translations.numpy(force=True).tobytes())
    config.special_hash = trace_hash.hexdigest()

    # Set number of input frames
    config.input_frames = gt_rotations.shape[0]

    # Create Se3 transformations
    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    # Set up camera parameters
    camera_trans = torch.FloatTensor(config.camera_position)[None].to(config.device)
    up = torch.FloatTensor(config.camera_up)[None].to(config.device)
    obj_center = torch.FloatTensor(config.obj_center)[None].to(config.device)

    gt_Se3_obj2cam = get_Se3_obj2cam_from_kaolin_params(camera_trans, up, obj_center)
    gt_Se3_cam2obj = Se3_obj_relative_to_Se3_cam2obj(gt_obj_1_to_obj_i_Se3, gt_Se3_obj2cam)
    gt_Se3_cam2obj_dict = {i: gt_Se3_cam2obj[i] for i in range(config.input_frames)}

    gt_Se3_world2cam = Se3.identity(config.input_frames, config.device)
    gt_Se3_world2cam_dict = {i: gt_Se3_world2cam[i] for i in range(config.input_frames)}

    # Set up output folder
    if output_folder is not None:
        write_folder = Path(output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment / dataset / sequence

    # Create and run tracker
    tracker = Tracker6D(config, write_folder, input_images=images_paths, gt_texture=gt_texture, gt_mesh=gt_mesh,
                        gt_Se3_cam2obj=gt_Se3_cam2obj_dict, gt_Se3_world2cam=gt_Se3_world2cam_dict)

    tracker.run_pipeline()

    return tracker


def reindex_frame_dict(frame_dict: Dict[int, Any], valid_frames: List[int]):
    frame_dict = {
        i: frame_dict[frame]
        for i, frame in enumerate(valid_frames) if frame in frame_dict
    }
    return frame_dict


def run_on_bop_sequences(dataset: str, experiment_name: str, sequence_type: str, config: TrackerConfig, gt_cam_scale,
                         output_folder: Path = None, scene_obj_id: int = None):

    onboarding_type = config.bop_config.onboarding_type
    sequence = config.sequence

    # Path to BOP dataset
    bop_folder = config.default_data_folder / 'bop'

    if onboarding_type == 'static':
        static_onboarding_sequence = config.bop_config.static_onboarding_sequence
        config.special_hash = static_onboarding_sequence
    elif onboarding_type == 'dynamic':
        config.special_hash = 'dynamic'
        static_onboarding_sequence = None
    elif sequence_type in ['test', 'train', 'val']:
        config.special_hash = f'{scene_obj_id:06d}'
        static_onboarding_sequence = None
    else:
        raise ValueError("This should not happen")

    # Determine output folder
    if output_folder is not None:
        write_folder = Path(output_folder) / dataset / f'{sequence}_{config.special_hash}'
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / f'{sequence}_{config.special_hash}'

    # Load images and segmentations
    gt_images, gt_segs, gt_depths, sequence_starts =\
        get_bop_images_and_segmentations(bop_folder, dataset, sequence, sequence_type,
                                         onboarding_type, static_onboarding_sequence, scene_obj_id=scene_obj_id)

    # Get camera-to-object transformations
    dict_gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence, sequence_type,
                                                              gt_cam_scale, onboarding_type,
                                                              sequence_starts, static_onboarding_sequence, scene_obj_id,
                                                              device=config.device)

    object_id = read_object_id(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                               static_onboarding_sequence, scene_obj_id, sequence_starts)
    config.object_id = object_id

    # Apply frame skipping
    if config.run_only_on_frames_with_known_pose:
        valid_frames = sorted(dict_gt_Se3_cam2obj.keys())
    else:
        valid_frames = list(range(min(gt_images.keys()), max(gt_images.keys()) + 1))

    gt_images = [gt_images[i] for i in valid_frames]
    gt_segs = [gt_segs[i] for i in valid_frames]
    if gt_depths is not None:
        gt_depths = [gt_depths[i] for i in valid_frames]
    dict_gt_Se3_cam2obj = reindex_frame_dict(dict_gt_Se3_cam2obj, valid_frames)

    # Get initial image and segmentation
    first_segmentation = PrecomputedSegmentationProvider.get_initial_segmentation(gt_images, gt_segs,
                                                                                  segmentation_channel=0)

    # Get camera parameters
    pinhole_params = read_pinhole_params(bop_folder, dataset, sequence, sequence_type, config.image_downsample,
                                         onboarding_type, static_onboarding_sequence, sequence_starts, config.device)

    gt_Se3_world2cam = None
    if onboarding_type == 'static' or sequence_type in ['val', 'train']:
        gt_Se3_world2cam = read_static_onboarding_world2cam(bop_folder, dataset, sequence, sequence_type,
                                                            onboarding_type, static_onboarding_sequence,
                                                            sequence_starts, config.device)
    if gt_Se3_world2cam is not None:
        pinhole_params = add_extrinsics_to_pinhole_params(pinhole_params, gt_Se3_world2cam)

    pinhole_params = reindex_frame_dict(pinhole_params, valid_frames)
    if gt_Se3_world2cam is not None:
        gt_Se3_world2cam = reindex_frame_dict(gt_Se3_world2cam, valid_frames)

    if dict_gt_Se3_cam2obj is not None:
        gt_Se3_world2cam = {i: cam2obj.inverse() for i, cam2obj in dict_gt_Se3_cam2obj.items()}
        dict_gt_Se3_cam2obj = None

    # Update config with frame information
    config.input_frames = len(gt_images)
    config.frame_provider = 'precomputed'
    config.segmentation_provider = 'SAM2'

    # Initialize and run the tracker
    tracker = Tracker6D(config, write_folder, input_images=gt_images, gt_Se3_world2cam=gt_Se3_world2cam,
                        gt_pinhole_params=pinhole_params, input_segmentations=gt_segs, depth_paths=gt_depths,
                        initial_segmentation=first_segmentation)

    tracker.run_pipeline()

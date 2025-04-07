import hashlib
from pathlib import Path
import torch
from kornia.geometry import Se3, Quaternion

from dataset_generators import scenarios
from models.rendering import get_Se3_obj2cam_from_kaolin_params
from tracker_config import TrackerConfig
from utils.bop_challenge import get_bop_images_and_segmentations, read_gt_Se3_cam2obj_transformations, \
    read_pinhole_params
from utils.math_utils import Se3_obj_relative_to_Se3_cam2obj
from tracker6d import Tracker6D
from utils.data_utils import load_texture, load_mesh, get_initial_image_and_segment


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
    gt_rotations = gt_rotations[::skip_frames]

    # Create image paths
    images_paths = [Path(f'{i}.png') for i in range(gt_rotations.shape[0])]
    images_paths = images_paths[::skip_frames]

    # Generate translations (zero by default)
    gt_translations = scenarios.generate_sinusoidal_translations(steps=gt_rotations.shape[0]).translations * 0
    gt_translations = gt_translations.to(config.device)

    trace_hash = hashlib.md5(gt_rotations.numpy(force=True).tobytes() + gt_translations.numpy(force=True).tobytes())
    config.special_hash = trace_hash.hexdigest()

    # Create Se3 transformations
    gt_obj_1_to_obj_i_Se3 = Se3(Quaternion.from_axis_angle(gt_rotations), gt_translations)

    # Set up camera parameters
    camera_trans = torch.FloatTensor(config.camera_position)[None].to(config.device)
    up = torch.FloatTensor(config.camera_up)[None].to(config.device)
    obj_center = torch.FloatTensor(config.obj_center)[None].to(config.device)

    gt_Se3_obj2cam = get_Se3_obj2cam_from_kaolin_params(camera_trans, up, obj_center)
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


def run_on_bop_sequences(dataset: str, experiment_name: str, sequence: str, sequence_type: str, args,
                         config: TrackerConfig, skip_indices: int, onboarding_type: str = None,
                         only_frames_with_known_poses: bool = False):
    """
    Run the 6D tracker on BOP dataset sequences.

    Args:
        dataset: The dataset name (e.g., 'hope', 'handal')
        experiment_name: Name of the experiment
        sequence: Sequence identifier
        sequence_type: Type of sequence (e.g., 'val', 'onboarding')
        args: Command line arguments
        config: Tracker configuration
        skip_indices: Number of frames to skip when processing
        onboarding_type: Type of onboarding data, if applicable
        only_frames_with_known_poses: Skip frames without known poses

    Returns:
        None
    """
    # Determine output folder
    if args.output_folder is not None:
        write_folder = Path(args.output_folder) / dataset / sequence
    else:
        write_folder = config.default_results_folder / experiment_name / dataset / sequence

    # Path to BOP dataset
    bop_folder = config.default_data_folder / 'bop'

    if onboarding_type == 'static':
        static_onboarding_sequence = config.bop_config.static_onboarding_sequence
        config.special_hash = static_onboarding_sequence
    else:
        static_onboarding_sequence = None

    # Load images and segmentations
    gt_images, gt_segs, sequence_starts = get_bop_images_and_segmentations(bop_folder, dataset, sequence, sequence_type,
                                                                           onboarding_type, static_onboarding_sequence)

    # Get camera-to-object transformations
    dict_gt_Se3_cam2obj = read_gt_Se3_cam2obj_transformations(bop_folder, dataset, sequence, sequence_type,
                                                              onboarding_type, sequence_starts,
                                                              static_onboarding_sequence, device=config.device)

    # Get first frame camera pose
    gt_Se3_cam2obj_frame0 = dict_gt_Se3_cam2obj[min(dict_gt_Se3_cam2obj.keys())]

    # Apply frame skipping
    if only_frames_with_known_poses:
        valid_frames = list(dict_gt_Se3_cam2obj.keys())
    else:
        valid_frames = list(range(min(gt_images.keys()), max(gt_images.keys()) + 1))

    valid_frames = valid_frames[::skip_indices]
    gt_images = [gt_images[i] for i in valid_frames]
    gt_segs = [gt_segs[i] for i in valid_frames]
    dict_gt_Se3_cam2obj = {
        i: dict_gt_Se3_cam2obj[frame]
        for i, frame in enumerate(valid_frames)
    }

    # Get initial image and segmentation
    first_image, first_segmentation = get_initial_image_and_segment(
        gt_images,
        gt_segs,
        segmentation_channel=0
    )

    # Get camera parameters
    pinhole_params = read_pinhole_params(bop_folder, dataset, sequence, sequence_type, onboarding_type,
                                         static_onboarding_sequence, sequence_starts)

    # Set camera parameters in config
    min_index = min(valid_frames)
    config.camera_intrinsics = pinhole_params[min_index].intrinsics.squeeze().numpy(force=True)
    config.camera_extrinsics = pinhole_params[min_index].extrinsics.squeeze().numpy(force=True)

    # Update config with frame information
    config.input_frames = len(gt_images)
    config.frame_provider = 'precomputed'
    config.segmentation_provider = 'SAM2'

    # Initialize and run the tracker
    tracker = Tracker6D(
        config,
        write_folder,
        initial_gt_Se3_cam2obj=gt_Se3_cam2obj_frame0,
        gt_Se3_cam2obj=dict_gt_Se3_cam2obj,
        images_paths=gt_images,
        segmentation_paths=gt_segs,
        initial_segmentation=first_segmentation,
        initial_image=first_image
    )

    tracker.run_filtering_with_reconstruction()

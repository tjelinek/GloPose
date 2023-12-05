import argparse
import numpy

import pickle
import logging

from pathlib import Path

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np
import shutil


def triangle_cycle_indexing(x, N):
    """
    x: current frame
    N: length of the pattern
    return idx of the pattern frame to simulate back and forth motion
    0 to N ... no changes (increasing indexes)
    N to 2N ... decreasing indexes
    2N to 3N ... same as 0 to N
    3N to 4N ... same as N to 2N
    ....
    """
    k = N - 1
    if (x % k == 0) or (x % k == N - 1):
        sign = 0
    elif x % (2 * k) > k:
        sign = -1
    else:
        sign = 1
    return k - np.abs((x % (2 * k)) - k), sign


def render_objects_using_kubric(FLAGS, scenario_cfg):
    # --- Common setups & resources
    BACKGROUND_OFFSET = 0
    FLAGS.frame_end = scenario_cfg['movement_scenario']['steps']

    scene, rng_main, output_dir, scratch_dir = kb.setup(FLAGS)
    output_dir = Path('/datagrid') / scenario_cfg['scenario_name']
    output_dir.mkdir(exist_ok=True, parents=True)

    simulator = PyBullet(scene, scratch_dir)

    motion_blur = rng_main.uniform(FLAGS.min_motion_blur, FLAGS.max_motion_blur)
    if motion_blur > 0.0:
        logging.info(f"Using motion blur strength {motion_blur}")

    renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=FLAGS.samples_per_pixel,
                       motion_blur=motion_blur)
    kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
    hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)

    # --- Populate the scene
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    seed_background = FLAGS.seed if FLAGS.seed else rng_main.randint(0, 2147483647)
    rng_background = np.random.RandomState(seed=seed_background)

    hdri_id = rng_background.choice(train_backgrounds)
    hdri_id = 'stuttgart_suburbs'

    background_hdri = hdri_source.create(asset_id=hdri_id)
    # assert isinstance(background_hdri, kb.Texture)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    if scenario_cfg['background_image_path'] is not None:
        renderer.bg_hdri_node = renderer._set_background_hdri(background_hdri.filename)

    # Camera
    logging.info("Setting up the Camera...")

    scene.camera = kb.PerspectiveCamera(focal_length=FLAGS.focal_length, sensor_width=FLAGS.sensor_width)
    scene.camera.position = (0., float(scenario_cfg['config']['camera_distance'] + BACKGROUND_OFFSET), 0.)
    scene.camera.look_at((0, 0, 0))

    obj = kb.FileBasedObject(
        asset_id=scenario_cfg['scenario_name'],
        render_filename=str(scenario_cfg['prototype_path']),
        bounds=((-1, -1, -1), (1, 1, 1)),
        simulation_filename=None)
    obj.metadata["is_dynamic"] = False
    scene += obj

    # repeater
    for frame_id in range(scenario_cfg['movement_scenario']['steps']):
        object_pos = tuple(scenario_cfg['movement_scenario']['translations'][frame_id] +
                           scenario_cfg['movement_scenario']['initial_translation'] +
                           numpy.asarray([0., 0., BACKGROUND_OFFSET]))
        object_pos = (object_pos[0], object_pos[2], object_pos[1])
        obj.position = object_pos

        # TODO handle initial quaternion composition
        quaternion = tuple(scenario_cfg['movement_scenario']['rotation_quaternions'][frame_id])
        quaternion = (quaternion[0], quaternion[1], quaternion[3], quaternion[2])
        obj.quaternion = quaternion
        obj.keyframe_insert("position", frame_id)
        obj.keyframe_insert("quaternion", frame_id)

    # --- Rendering
    if FLAGS.save_state:
        logging.info("Saving the renderer state to '%s' ",
                     output_dir / "scene.blend")
        renderer.save_state(output_dir / "scene.blend")

    # Run dynamic objects simulation
    logging.info("Running the simulation ...")
    animation, collisions = simulator.run(frame_start=0,
                                          frame_end=scene.frame_end + 1)

    logging.info("Rendering the scene ...")
    data_stack = renderer.render()

    # --- Postprocessing
    kb.compute_visibility(data_stack["segmentation"], scene.assets)
    visible_foreground_assets = [asset for asset in scene.foreground_assets
                                 if np.max(asset.metadata["visibility"]) > 0]
    visible_foreground_assets = sorted(  # sort assets by their visibility
        visible_foreground_assets,
        key=lambda asset: np.sum(asset.metadata["visibility"]),
        reverse=True)

    data_stack["segmentation"] = kb.adjust_segmentation_idxs(
        data_stack["segmentation"],
        scene.assets,
        visible_foreground_assets)
    scene.metadata["num_instances"] = len(visible_foreground_assets)

    # Save to image files
    kb.write_image_dict(data_stack, output_dir)
    kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                      visible_foreground_assets)

    # --- Metadata
    logging.info("Collecting and storing metadata for each object.")
    kb.write_json(filename=output_dir / "metadata.json", data={
        "flags": vars(FLAGS),
        "metadata": kb.get_scene_metadata(scene),
        "camera": kb.get_camera_info(scene.camera),
        "instances": kb.get_instance_info(scene, visible_foreground_assets),
    })
    kb.write_json(filename=output_dir / "events.json", data={
        "collisions": kb.process_collisions(
            collisions, scene, assets_subset=visible_foreground_assets),
    })

    kb.done()
    shutil.rmtree(scratch_dir)


def get_generator_flags():
    parser = kb.ArgumentParser()
    parser.add_argument("--objects_split", choices=["train", "test"],
                        default="train")
    # Configuration for the objects of the scene
    parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
                        help="minimum number of dynamic (tossed) objects")
    parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
                        help="maximum number of dynamic (tossed) objects")
    # Configuration for the floor and background
    parser.add_argument("--floor_friction", type=float, default=0.3)
    parser.add_argument("--floor_restitution", type=float, default=0.5)
    parser.add_argument("--backgrounds_split", choices=["train", "test"],
                        default="train")
    parser.add_argument("--camera", choices=["fixed_random"],
                        default="fixed_random")
    parser.add_argument("--general_scenario", choices=["forward"],
                        default='forward')
    parser.add_argument("--cycle_frames_for_objects", type=int, default=-1,
                        help="half-life of the cycle")
    parser.add_argument("--camera_steps", type=int, default=2,
                        help="number of steps for camera linear motion (default: 2, only start and end point)")
    parser.add_argument("--max_motion_blur", type=float, default=0.0)
    parser.add_argument("--min_motion_blur", type=float, default=0.0)
    # Configuration for the source of the assets
    parser.add_argument("--kubasic_assets", type=str,
                        default="gs://kubric-public/assets/KuBasic/KuBasic.json")
    parser.add_argument("--hdri_assets", type=str,
                        default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    parser.add_argument("--gso_assets", type=str,
                        default="gs://kubric-public/assets/GSO/GSO.json")
    parser.add_argument("--shapenet_assets", type=str,
                        default="gs://kubric-unlisted/assets/ShapeNetCore.v2.json")
    parser.add_argument("--save_state", dest="save_state", action="store_true")
    parser.add_argument("--gravity_z", type=float, default=-9.81,
                        help="Gravity setting for z-axis.")
    parser.add_argument("--focal_length", type=float, default=35., help="focal length of the camera (mm)")
    parser.add_argument("--sensor_width", type=float, default=32., help="width of the camera sensor (mm)")
    parser.add_argument("--object_list_name", choices=["gso", "shapenet"],
                        default="gso")
    parser.add_argument("--samples_per_pixel", type=int, default=64,
                        help="renderer setting - samples per pixel")
    parser.set_defaults(save_state=False, frame_end=3, frame_rate=12,
                        resolution=800)
    parser.add_argument('--pkl_file', nargs='?', type=str, help='Path to the .pkl file (optional)')
    FLAGS = parser.parse_args()

    FLAGS.frame_end = 180

    return FLAGS


if __name__ == '__main__':
    flags = get_generator_flags()
    pkl_file = flags.pkl_file

    datagrid_path = Path('/datagrid')

    if pkl_file:
        print("Opening .pkl file")
        # Process the specified .pkl file
        with open(datagrid_path / pkl_file, 'rb') as file:
            loaded_config = pickle.load(file)
            render_objects_using_kubric(flags, loaded_config)
    else:
        # Loop over all .pkl files in the directory
        pkl_files = datagrid_path.glob('*.pkl')
        for scenario_config_path in pkl_files:
            with open(scenario_config_path, 'rb') as file:
                loaded_config = pickle.load(file)
                if 'translation' not in loaded_config['scenario_name'] \
                        or 'background' not in loaded_config['scenario_name']:
                    continue
                render_objects_using_kubric(flags, loaded_config)

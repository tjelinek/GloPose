from dataclasses import asdict

import imageio.v3 as iio
import kaolin
import torchvision
import numpy as np
import torch
import csv
import pickle
from kornia.geometry.conversions import axis_angle_to_quaternion, quaternion_to_rotation_matrix
from pathlib import Path

from dataset_generators.scenarios import MovementScenario
from models.encoder import EncoderResult
from models.rendering import RenderingKaolin
from utils import qnorm, qmult, normalize_vertices
from flow import visualize_flow_with_images


def save_renderings(image_idx, ren_features, ren_mask, rendering_destination, segmentation_destination,
                    background_image):
    """
    Generate and save rendering and segmentation images.

    This function takes the features of a rendered object, and a mask, both of which are 2D images,
    converts them to numpy arrays, and then saves them as png images in the specified directories.

    Args:
        image_idx (int): The index of the image, used to generate the file name.
        ren_features (torch.Tensor): A tensor representing the features of the rendered object.
        ren_mask (torch.Tensor): A tensor representing the mask of the rendered object.
        rendering_destination (Path): The destination directory where the rendered image will be saved.
        segmentation_destination (Path): The destination directory where the mask image will be saved.
        background_image (torch.Tensor): Background image.

    Returns:
        None
    """
    ren_mask_repeated = ren_mask.repeat(1, 1, 3, 1, 1)
    ren_features = ren_features * ren_mask_repeated + background_image * (1 - ren_mask_repeated)

    ren_features_np = ren_features.numpy(force=True)[0, 0].astype('uint8')
    ren_features_np = ren_features_np.transpose(1, 2, 0)
    ren_mask_np = ren_mask_repeated.numpy(force=True).astype('uint8')[0, 0] * 255
    ren_mask_np = ren_mask_np.transpose((1, 2, 0))

    i_str = format(image_idx, '03d')
    rendering_file_name = rendering_destination / (i_str + '.png')
    segmentation_file_name = segmentation_destination / (i_str + '.png')

    iio.imwrite(segmentation_file_name, ren_mask_np)
    iio.imwrite(rendering_file_name, ren_features_np)


def save_optical_flow(image_idx, optical_flow, optical_flow_destination):
    i_str = format(image_idx, '03d')
    optical_flow_file_name = optical_flow_destination / (i_str + '.pt')
    torch.save(optical_flow, optical_flow_file_name)


def setup_renderer(config, faces, height, width, device):
    """
    Setup a Kaolin renderer based on given configuration and parameters.

    This function initiates a RenderingKaolin object with provided configuration and parameters.
    The renderer is used to render 3D objects, and this function also moves several properties
    of the renderer to the specified device (defined by global constant DEVICE, set to 'cuda').

    Args:
        config (dict): The configuration dictionary which includes settings for rendering.
        faces (np.ndarray): A numpy array containing the faces of the 3D object that is to be rendered.
        height (int): The height of the output image from the renderer.
        width (int): The width of the output image from the renderer.
        device (str): PyTorch device that is used for rendering

    Returns:
        RenderingKaolin: An instance of the RenderingKaolin class with the specified configuration and parameters,
                         and its properties moved to the DEVICE
    """
    rendering = RenderingKaolin(config, faces, width, height)
    rendering.obj_center = rendering.obj_center.to(device)
    rendering.faces = rendering.faces.to(device)
    rendering.camera_rot = rendering.camera_rot.to(device)
    rendering.camera_trans = rendering.camera_trans.to(device)
    rendering.camera_proj = rendering.camera_proj.to(device)
    return rendering


def prepare(gt_tracking_log_file, optical_flow_absolute_destination, optical_flow_relative_destination,
            rendering_destination, segmentation_destination, movement_scenario):
    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_relative_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_absolute_destination.mkdir(parents=True, exist_ok=True)
    gt_tracking_log_file.parent.mkdir(parents=True, exist_ok=True)

    create_tracking_log(movement_scenario, gt_tracking_log_file)


def generate_rotating_and_translating_textured_object(config, movement_scenario, prototype_path, texture_path: Path,
                                                      rendering_destination: Path, segmentation_destination: Path,
                                                      optical_flow_relative_destination,
                                                      optical_flow_absolute_destination, gt_tracking_log_file, width,
                                                      height, background_image_path: str = None):
    prepare(gt_tracking_log_file, optical_flow_absolute_destination, optical_flow_relative_destination,
            rendering_destination, segmentation_destination, movement_scenario)

    tex = iio.imread(str(texture_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].cuda()
    heterogeneous_mesh_handler = kaolin.io.utils.heterogeneous_mesh_handler_naive_homogenize
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True,
                                     heterogeneous_mesh_handler=heterogeneous_mesh_handler)
    vertices = mesh.vertices[None]

    vertices = normalize_vertices(vertices)

    faces = mesh.faces
    face_features = mesh.uvs[mesh.face_uvs_idx][None]

    rendering = setup_renderer(config, faces, width, height, 'cuda')

    if background_image_path is not None:
        image = iio.imread(background_image_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((height, width)),
            torchvision.transforms.ToTensor()
        ])
        background_image = transform(image)[None, None].cuda() * 255.0
    else:
        background_image = torch.zeros(1, 1, 3, height, width).cuda()

    render_object_poses(rendering, vertices, face_features, texture_maps, movement_scenario,
                        optical_flow_relative_destination, optical_flow_absolute_destination, rendering_destination,
                        segmentation_destination, background_image)


def prepare_scenarios_for_kubric(config, movement_scenario: MovementScenario, prototype_path, texture_path: Path,
                                 rendering_destination: Path, segmentation_destination: Path,
                                 optical_flow_relative_destination: Path,
                                 optical_flow_absolute_destination: Path, gt_tracking_log_file: Path, width: int,
                                 height, background_image_path: str = None):
    prepare(gt_tracking_log_file, optical_flow_absolute_destination, optical_flow_relative_destination,
            rendering_destination, segmentation_destination, movement_scenario)

    scenario_name = segmentation_destination.parent.stem
    scenario_path = segmentation_destination.parent.parent / (scenario_name + '.pkl')

    datagrid_path = Path('/mnt/personal/jelint19/data')
    optical_flow_absolute_destination = Path('/datagrid') / optical_flow_absolute_destination.relative_to(datagrid_path)
    optical_flow_relative_destination = Path('/datagrid') / optical_flow_relative_destination.relative_to(datagrid_path)
    rendering_destination = Path('/datagrid') / rendering_destination.relative_to(datagrid_path)
    segmentation_destination = Path('/datagrid') / segmentation_destination.relative_to(datagrid_path)

    texture_path = Path('/') / texture_path
    prototype_path = Path('/') / prototype_path

    scenario = {
        'gt_tracking_log_file': gt_tracking_log_file,
        'optical_flow_absolute_destination': optical_flow_absolute_destination,
        'optical_flow_relative_destination': optical_flow_relative_destination,
        'rendering_destination': rendering_destination,
        'segmentation_destination': segmentation_destination,
        'rendering_width': width,
        'rendering_height': height,
        'background_image_path': background_image_path,
        'texture_path': texture_path,
        'prototype_path': prototype_path,
        'movement_scenario': movement_scenario.get_dict(),
        'config': asdict(config),
        'scenario_name': scenario_name
    }

    with open(scenario_path, 'wb') as file:
        pickle.dump(scenario, file)


def create_tracking_log(movement_scenario: MovementScenario, gt_tracking_log_file: Path):
    initial_translation = torch.from_numpy(movement_scenario.initial_translation).cuda().to(torch.float32)
    translations_tensors = [torch.from_numpy(arr).to(torch.float32) for arr in movement_scenario.translations]
    translations = torch.stack(translations_tensors)[None][None].cuda()
    translations[..., :] += initial_translation

    initial_rotation_x, initial_rotation_y, initial_rotation_z = movement_scenario.initial_rotation

    log_rows = []

    for frame_i, (rotation_x, rotation_y, rotation_z) in enumerate(movement_scenario.rotations):
        rotation_x += initial_rotation_x
        rotation_y += initial_rotation_y
        rotation_z += initial_rotation_z

        current_translation = translations[:, :, frame_i, :][None]

        log_rows.append([frame_i, rotation_x, rotation_y, rotation_z] + current_translation[0, 0, 0].tolist())

    with open(gt_tracking_log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"])
        for row in log_rows:
            writer.writerow(row)


def render_object_poses(rendering: RenderingKaolin, vertices, face_features, texture_maps,
                        movement_scenario: MovementScenario,
                        optical_flow_relative_destination, optical_flow_absolute_destination, rendering_destination,
                        segmentation_destination, background_image):
    # The initial rotations are expected to be in radians

    initial_translation = torch.from_numpy(movement_scenario.initial_translation).cuda().to(torch.float32)

    initial_rotation_axis_angle = torch.from_numpy(np.deg2rad(movement_scenario.initial_rotation))
    initial_rotation_axis_angle = initial_rotation_axis_angle.cuda().to(torch.float32)
    initial_rotation_quaternion = axis_angle_to_quaternion(initial_rotation_axis_angle)

    prev_encoder_result = None
    prev_rendering_rgb = None
    first_rendering_rgb = None
    first_encoder_result = None

    vertices = vertices.cuda().to(torch.float32)
    face_features = face_features.cuda().to(torch.float32)

    quaternions = torch.zeros(1, len(movement_scenario.rotations), 4).cuda()
    quaternions[:, :, 0] = 1.0

    translations_tensors = [torch.from_numpy(arr).to(torch.float32) for arr in movement_scenario.translations]
    translations = torch.stack(translations_tensors)[None][None].cuda()
    translations[..., :] += initial_translation

    for frame_i, rotation_quaternion in enumerate(movement_scenario.rotation_quaternions):
        rotation_quaternion_tensor = torch.from_numpy(rotation_quaternion).to(torch.float32)

        composed_rotation_quaternion_tensor = qmult(qnorm(initial_rotation_quaternion[None]),
                                                    qnorm(rotation_quaternion_tensor[None]))[0].cuda()
        quaternions[0, frame_i] = composed_rotation_quaternion_tensor

        rotation_matrix = quaternion_to_rotation_matrix(composed_rotation_quaternion_tensor)[None]
        current_translation = translations[:, :, frame_i, :][None]
        current_encoder_result = EncoderResult(translations=current_translation,
                                               quaternions=composed_rotation_quaternion_tensor[None, None],
                                               vertices=vertices,
                                               texture_maps=None,
                                               lights=None,
                                               translation_difference=None,
                                               quaternion_difference=None)

        if prev_encoder_result is None:
            prev_encoder_result = current_encoder_result
        if first_encoder_result is None:
            first_encoder_result = current_encoder_result

        flow_arcs_indices = [(0, 0)]

        with torch.no_grad():
            # breakpoint()
            rendering_result = rendering.render_mesh_with_dibr(face_features, rotation_matrix,
                                                               current_translation[0, 0], vertices)

            optical_flow_relative, _, _ = rendering.compute_theoretical_flow(current_encoder_result,
                                                                             prev_encoder_result,
                                                                             flow_arcs_indices)

            optical_flow_absolute, _, _ = rendering.compute_theoretical_flow(current_encoder_result,
                                                                             first_encoder_result, flow_arcs_indices)

            if texture_maps is not None:
                rendering_result = rendering.forward(translation=translations[:, :, frame_i:frame_i + 1, :],
                                                     quaternion=quaternions[:, frame_i:frame_i + 1, :],
                                                     unit_vertices=vertices, face_features=face_features,
                                                     texture_maps=texture_maps)

                rendering_rgb = rendering_result.rendered_image
                ren_silhouette = rendering_result.rendered_image_segmentation
            else:
                rendering_rgb = rendering_result.ren_mesh_vertices_features.permute(0, -1, 1, 2)[None]
                ren_silhouette = rendering_result.ren_mask[None, None]

            if first_rendering_rgb is None:
                first_rendering_rgb = rendering_rgb.clone()

            if prev_rendering_rgb is not None:
                generate_optical_flow_illustration(rendering_rgb, prev_rendering_rgb, optical_flow_relative.clone(),
                                                   frame_i, optical_flow_relative_destination)
                generate_optical_flow_illustration(rendering_rgb, first_rendering_rgb, optical_flow_absolute.clone(),
                                                   frame_i, optical_flow_absolute_destination)

            save_renderings(frame_i, rendering_rgb, ren_silhouette, rendering_destination, segmentation_destination,
                            background_image)

            save_optical_flow(frame_i, optical_flow_relative, optical_flow_relative_destination)
            save_optical_flow(frame_i, optical_flow_absolute, optical_flow_absolute_destination)

            del prev_rendering_rgb
            prev_rendering_rgb = rendering_rgb

        prev_encoder_result = current_encoder_result


def generate_optical_flow_illustration(ren_features, prev_ren_features, optical_flow, frame_i, save_destination):
    prev_ren_features_ = prev_ren_features.squeeze().to(torch.uint8).detach()  # Shape [C, H, W]
    ren_features_ = ren_features.squeeze().to(torch.uint8).detach()  # Shape [C, H, W]

    optical_flow_ = optical_flow.squeeze().permute(1, 2, 0)
    optical_flow_[..., 0] = optical_flow_[..., 0] * optical_flow_.shape[-2]
    optical_flow_[..., 1] = optical_flow_[..., 1] * optical_flow_.shape[-3]
    optical_flow_ = optical_flow_.numpy(force=True)

    flow_illustration = visualize_flow_with_images([prev_ren_features_], ren_features_, [optical_flow_])
    del optical_flow_
    optical_flow_illustration_destination = \
        save_destination.with_name(save_destination.stem +
                                   "_illustration" + save_destination.suffix)
    optical_flow_illustration_destination.mkdir(exist_ok=True, parents=True)
    file_name = optical_flow_illustration_destination / f"flow_{frame_i}.png"
    iio.imwrite(file_name, flow_illustration)

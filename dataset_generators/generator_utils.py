import imageio
import kaolin
import numpy as np
import torch
import csv
from kornia.geometry import quaternion_to_rotation_matrix
from kornia.geometry.conversions import QuaternionCoeffOrder, angle_axis_to_quaternion
from pathlib import Path

from dataset_generators.scenarios import generate_rotations_y
from models.encoder import EncoderResult
from models.rendering import RenderingKaolin
from utils import deg_to_rad, qnorm, qmult, normalize_vertices
from flow import visualize_flow_with_images


def save_images_and_flow(image_idx, ren_features, ren_mask, optical_flow, rendering_destination,
                         segmentation_destination, optical_flow_destination):
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
        optical_flow (Tensor): Saved theoretical optical flow Tensor
        optical_flow_destination (Path): The destination directory where the optical flow Tensor will be saved.

    Returns:
        None
    """
    ren_features_np = ren_features.numpy(force=True)[0, 0].astype('uint8')
    ren_features_np = ren_features_np.transpose(1, 2, 0)
    ren_mask_np = ren_mask.numpy(force=True).astype('uint8')[0, 0] * 255
    ren_mask_np = np.tile(ren_mask_np, (3, 1, 1)).transpose((1, 2, 0))

    i_str = format(image_idx, '03d')
    rendering_file_name = rendering_destination / (i_str + '.png')
    segmentation_file_name = segmentation_destination / (i_str + '.png')
    optical_flow_file_name = optical_flow_destination / (i_str + '.pt')

    torch.save(optical_flow, optical_flow_file_name)
    imageio.imwrite(segmentation_file_name, ren_mask_np)
    imageio.imwrite(rendering_file_name, ren_features_np)


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


def generate_rotating_and_translating_textured_object(config, movement_scenario, prototype_path, texture_path: Path,
                                                      rendering_destination: Path, segmentation_destination: Path,
                                                      optical_flow_destination, gt_tracking_log_file, width, height):
    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)
    gt_tracking_log_file.parent.mkdir(parents=True, exist_ok=True)

    tex = imageio.imread(str(texture_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].cuda()
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)
    vertices = mesh.vertices[None]

    vertices = normalize_vertices(vertices)

    faces = mesh.faces
    face_features = mesh.uvs[mesh.face_uvs_idx][None]

    rendering = setup_renderer(config, faces, width, height, 'cuda')

    render_object_poses(rendering, vertices, face_features, texture_maps, movement_scenario, optical_flow_destination,
                        rendering_destination, segmentation_destination, gt_tracking_log_file)


def render_object_poses(rendering, vertices, face_features, texture_maps, movement_scenario, optical_flow_destination,
                        rendering_destination, segmentation_destination, gt_tracking_log_file):
    # The initial rotations are expected to be in radians

    initial_translation = torch.from_numpy(movement_scenario.initial_translation).cuda().to(torch.float32)

    initial_rotation_axis_angle = torch.from_numpy(movement_scenario.initial_rotation).cuda().to(torch.float32)
    initial_rotation_quaternion = angle_axis_to_quaternion(initial_rotation_axis_angle, order=QuaternionCoeffOrder.WXYZ)

    prev_encoder_result = None
    prev_rendering_rgb = None

    vertices = vertices.cuda().to(torch.float32)
    face_features = face_features.cuda().to(torch.float32)

    quaternions = torch.zeros(1, len(movement_scenario.rotations), 4).cuda()
    quaternions[:, :, 0] = 1.0

    translations_tensors = [torch.from_numpy(arr).to(torch.float32) for arr in movement_scenario.translations]
    translations = torch.stack(translations_tensors)[None][None].cuda()
    translations[..., :] += initial_translation

    log_rows = []

    for frame_i, (rotation_x, rotation_y, rotation_z) in enumerate(movement_scenario.rotations):
        axis_angle_tensor = torch.Tensor([deg_to_rad(rotation_x),
                                          deg_to_rad(rotation_y),
                                          deg_to_rad(rotation_z)]).to(torch.float32)
        rotation_quaternion_tensor = angle_axis_to_quaternion(axis_angle_tensor,
                                                              order=QuaternionCoeffOrder.WXYZ)  # Shape (4)
        composed_rotation_quaternion_tensor = qmult(qnorm(initial_rotation_quaternion[None]),
                                                    qnorm(rotation_quaternion_tensor[None]))[0].cuda()
        quaternions[0, frame_i] = composed_rotation_quaternion_tensor

        rotation_matrix = quaternion_to_rotation_matrix(composed_rotation_quaternion_tensor,
                                                        order=QuaternionCoeffOrder.WXYZ)[None]
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
        flow_arcs_indices = [(0, 0)]

        with torch.no_grad():
            log_rows.append([frame_i, rotation_x, rotation_y, rotation_z] + current_translation[0, 0, 0].tolist())
            rendering_result = rendering.render_mesh_with_dibr(face_features, rotation_matrix, current_translation[0],
                                                               vertices)

            optical_flow, _, _ = rendering.compute_theoretical_flow(current_encoder_result, prev_encoder_result,
                                                                    flow_arcs_indices)

            if texture_maps is not None:
                ren_features_and_mask = rendering.forward(translation=translations[:, :, frame_i:frame_i + 1, :],
                                                          quaternion=quaternions[:, frame_i:frame_i + 1, :],
                                                          unit_vertices=vertices, face_features=face_features,
                                                          texture_maps=texture_maps)

                rendering_rgb, ren_silhouette = ren_features_and_mask
            else:
                rendering_rgb = rendering_result.ren_mesh_vertices_features.permute(0, -1, 1, 2)[None]
                ren_silhouette = rendering_result.ren_mask[None, None]

            if prev_rendering_rgb is not None:
                generate_optical_flow_illustration(rendering_rgb, prev_rendering_rgb, optical_flow.clone(), frame_i,
                                                   optical_flow_destination)

            save_images_and_flow(frame_i, rendering_rgb, ren_silhouette,
                                 optical_flow.detach().clone().cpu(), rendering_destination,
                                 segmentation_destination, optical_flow_destination)

            del prev_rendering_rgb
            prev_rendering_rgb = rendering_rgb

        prev_encoder_result = current_encoder_result

    with open(gt_tracking_log_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"])
        for row in log_rows:
            writer.writerow(row)


def generate_optical_flow_illustration(ren_features, prev_ren_features, optical_flow, frame_i, save_destination):
    prev_ren_features_ = prev_ren_features.squeeze().to(torch.uint8).detach()  # Shape [C, H, W]
    ren_features_ = ren_features.squeeze().to(torch.uint8).detach()  # Shape [C, H, W]

    optical_flow_ = optical_flow.squeeze().permute(1, 2, 0)
    optical_flow_[..., 0] = optical_flow_[..., 0] * optical_flow_.shape[-2]
    optical_flow_[..., 1] = optical_flow_[..., 1] * optical_flow_.shape[-3]
    optical_flow_ = optical_flow_.numpy(force=True)

    flow_illustration = visualize_flow_with_images(prev_ren_features_, ren_features_, optical_flow_)
    del optical_flow_
    optical_flow_illustration_destination = \
        save_destination.with_name(save_destination.stem +
                                   "_illustration" + save_destination.suffix)
    optical_flow_illustration_destination.mkdir(exist_ok=True, parents=True)
    file_name = optical_flow_illustration_destination / f"flow_{frame_i}.png"
    imageio.imwrite(file_name, flow_illustration)

import imageio
import kaolin
import numpy as np
import torch
from kornia.geometry import quaternion_to_rotation_matrix
from kornia.geometry.conversions import QuaternionCoeffOrder, angle_axis_to_quaternion
from pathlib import Path

from models.encoder import EncoderResult
from models.rendering import RenderingKaolin
from utils import deg_to_rad
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
    ren_features_np = ren_features.cpu().numpy()[0].astype('uint8')
    i_str = format(image_idx, '03d')
    rendering_file_name = rendering_destination / (i_str + '.png')
    segmentation_file_name = segmentation_destination / (i_str + '.png')
    optical_flow_file_name = optical_flow_destination / (i_str + '.pt')
    ren_mask_np = ren_mask.cpu().numpy().astype('uint8')[0] * 255
    ren_mask_np_rep = np.tile(ren_mask_np, (3, 1, 1)).transpose((1, 2, 0))
    # flow_image = flow_viz.flow_to_image(optical_flow)
    torch.save(optical_flow, optical_flow_file_name)
    imageio.imwrite(segmentation_file_name, ren_mask_np_rep)
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


def generate_2_DoF_rotations(step=10.0):
    rotations_x = np.arange(0.0, 1 * 360.0 + 0.001, step)
    rotations_y = np.concatenate([np.zeros(rotations_x.shape[0] // 2), np.arange(0.0, 0.5 * 360.0 + 0.001, 10.0)])
    rotations_z = np.zeros(rotations_y.shape)
    return list(zip(rotations_x, rotations_y, rotations_z))


def generate_1_DoF_rotation(step=10.0):
    rotations_x = np.arange(0.0, 1 * 360.0 + 0.001, step)
    rotations_y = np.zeros(rotations_x.shape)
    rotations_z = np.zeros(rotations_x.shape)

    return list(zip(rotations_x, rotations_y, rotations_z))


def generate_rotating_textured_object(config, prototype_path, texture_path: Path, rendering_destination: Path,
                                      segmentation_destination: Path, optical_flow_destination, width, height,
                                      DEVICE='cuda', rotations=None):
    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)
    optical_flow_destination.mkdir(parents=True, exist_ok=True)

    tex = imageio.imread(str(texture_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].to(DEVICE)
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)
    vertices = mesh.vertices[None]

    magnification = 1 / (vertices.max() - vertices.mean()) * 1.0

    vertices *= magnification
    faces = mesh.faces
    face_features = mesh.uvs[mesh.face_uvs_idx][None]

    rendering = setup_renderer(config, faces, width, height, DEVICE)

    if rotations is None:
        rotations = generate_1_DoF_rotation(2.0)

    render_object_poses(rendering, vertices, face_features, texture_maps, rotations, optical_flow_destination,
                        rendering_destination, segmentation_destination, DEVICE)


def render_object_poses(rendering, vertices, face_features, texture_maps, rotations, optical_flow_destination,
                        rendering_destination, segmentation_destination, DEVICE):
    prev_encoder_result = None
    prev_ren_features = None

    for frame_i, (rotation_x, rotation_y, rotation_z) in enumerate(rotations):
        axis_angle_tensor = torch.Tensor([deg_to_rad(rotation_x), deg_to_rad(rotation_y), deg_to_rad(rotation_z)])
        rotation_quaternion_tensor = angle_axis_to_quaternion(axis_angle_tensor,
                                                              order=QuaternionCoeffOrder.WXYZ)  # Shape (4)
        rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion_tensor,
                                                        order=QuaternionCoeffOrder.WXYZ)[None]

        current_encoder_result = EncoderResult(translations=rendering.obj_center[None, None].to(DEVICE),
                                               quaternions=rotation_quaternion_tensor[None, None].to(DEVICE),
                                               vertices=vertices.to(DEVICE),
                                               texture_maps=None,
                                               lights=None,
                                               translation_difference=None,
                                               quaternion_difference=None)

        if prev_encoder_result is None:
            prev_encoder_result = current_encoder_result

        with torch.no_grad():
            rendering_result = rendering.render_mesh_with_dibr(face_features.to(DEVICE), rotation_matrix.to(DEVICE),
                                                               rendering.obj_center, vertices.to(DEVICE))

            optical_flow = rendering.compute_theoretical_flow(current_encoder_result, prev_encoder_result)

            if texture_maps is not None:
                ren_features = kaolin.render.mesh.texture_mapping(rendering_result.ren_mesh_vertices_features,
                                                                  texture_maps, mode='bilinear')
            else:
                ren_features = rendering_result.ren_mesh_vertices_features

            if prev_ren_features is not None:
                prev_ren_features_ = prev_ren_features[0].to(torch.uint8).permute(2, 0, 1)  # Shape [C, H, W]
                # generate_optical_flow_illustration(ren_features, prev_ren_features_, optical_flow, frame_i,
                #                                    optical_flow_destination)

            save_images_and_flow(frame_i, ren_features, rendering_result.ren_mask,
                                 optical_flow.detach().cpu(), rendering_destination,
                                 segmentation_destination, optical_flow_destination)

            prev_ren_features = ren_features

        prev_encoder_result = current_encoder_result


def generate_optical_flow_illustration(ren_features, prev_ren_features_, optical_flow, frame_i, save_destination):
    ren_features_ = ren_features[0].to(torch.uint8).permute(2, 0, 1)  # Shape [C, H, W]
    optical_flow_ = optical_flow[0, 0].clone()
    optical_flow_[..., 0] = optical_flow_[..., 0] * optical_flow_.shape[-2]
    optical_flow_[..., 1] = optical_flow_[..., 1] * optical_flow_.shape[-3]
    optical_flow_ = optical_flow_.detach().cpu().numpy()
    flow_illustration = visualize_flow_with_images(prev_ren_features_, ren_features_, optical_flow_)
    optical_flow_illustration_destination = \
        save_destination.with_name(save_destination.stem +
                                   "_illustration" + save_destination.suffix)
    optical_flow_illustration_destination.mkdir(exist_ok=True, parents=True)
    file_name = optical_flow_illustration_destination / f"flow_{frame_i}.png"
    imageio.imwrite(file_name, flow_illustration)

import math
import warnings
import os
import shutil
import imageio
import kaolin
import numpy as np
import torch
import types
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from kornia.geometry import quaternion_to_rotation_matrix

from models.rendering import RenderingKaolin
from main_settings import dataset_folder
from utils import quaternion_from_euler, load_config, deg_to_rad


def generate_and_save_images(image_idx, ren_features, ren_mask, rendering_destination, segmentation_destination):
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

    Returns:
        None
    """
    ren_features_np = ren_features.cpu().numpy()[0].astype('uint8')
    i_str = format(image_idx, '03d')
    rendering_file_name = rendering_destination / (i_str + '.png')
    segmentation_file_name = segmentation_destination / (i_str + '.png')
    ren_mask_np = ren_mask.cpu().numpy().astype('uint8')[0] * 255
    ren_mask_np_rep = np.tile(ren_mask_np, (3, 1, 1)).transpose((1, 2, 0))
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


def generate_2_DoF_rotations():
    rotations_pitch = np.arange(0.0, 1 * 360.0 + 0.001, 10.0)
    rotations_yaw = np.concatenate([np.zeros(rotations_pitch.shape[0] // 2), np.arange(0.0, 0.5 * 360.0 + 0.001, 10.0)])
    rotations_roll = np.zeros(rotations_yaw.shape)
    return list(zip(rotations_pitch, rotations_roll, rotations_yaw))


def generate_1_DoF_rotation():
    rotations_pitch = np.arange(0.0, 1 * 360.0 + 0.001, 10.0)
    rotations_yaw = np.zeros(rotations_pitch.shape)
    rotations_roll = np.zeros(rotations_yaw.shape)
    return list(zip(rotations_pitch, rotations_roll, rotations_yaw))


def generate_rotating_textured_object(config, prototype_path, rendering_destination: Path,
                                      segmentation_destination: Path, texture_path: Path, width, height, DEVICE='cuda'):
    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)

    tex = imageio.imread(str(texture_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].to(DEVICE)
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)
    vertices = mesh.vertices[None]

    magnification = 1 / (vertices.max() - vertices.mean()) * 0.8

    vertices *= magnification
    faces = mesh.faces
    face_features = mesh.uvs[mesh.face_uvs_idx][None]
    translation = torch.zeros((1, 3))[None]

    rendering = setup_renderer(config, faces, width, height, DEVICE)

    rotations = generate_2_DoF_rotations()

    for i, (yaw, pitch, roll) in enumerate(rotations):
        rotation_quaternion = quaternion_from_euler(roll=torch.Tensor([deg_to_rad(roll)]),
                                                    pitch=torch.Tensor([deg_to_rad(pitch)]),
                                                    yaw=torch.Tensor([deg_to_rad(yaw)]))

        rotation_matrix = quaternion_to_rotation_matrix(torch.Tensor(rotation_quaternion))[None]

        with torch.no_grad():
            face_normals, face_vertices_cam, red_index, ren_mask, \
                ren_mesh_vertices_features, ren_mesh_vertices_coords \
                = rendering.render_mesh_with_dibr(face_features.to(DEVICE), rotation_matrix.to(DEVICE),
                                                  translation.to(DEVICE), vertices.to(DEVICE))

            ren_features = kaolin.render.mesh.texture_mapping(ren_mesh_vertices_features, texture_maps, mode='bilinear')
            generate_and_save_images(i, ren_features, ren_mask, rendering_destination, segmentation_destination)

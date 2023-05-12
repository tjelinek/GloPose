import math
import warnings
import os
import shutil
import imageio
import kaolin
import numpy as np
import torch
import types

from pathlib import Path
from kornia.geometry import quaternion_to_rotation_matrix
from models.rendering import RenderingKaolin
from utils import quaternion_from_euler, load_config

warnings.filterwarnings("ignore")

DEVICE = 'cuda'


def generate_textured_sphere(rendering_destination: Path, segmentation_destination: Path):
    prototype_path = Path('./prototypes/sphere_voronoi.obj')
    tex_path = Path('./prototypes/tex3.png')
    config = load_config('configs/config_deep.yaml')

    generate_rotating_textured_object(config, prototype_path, rendering_destination, segmentation_destination, tex_path)


def generate_rotating_textured_object(config, prototype_path, rendering_destination: Path,
                                      segmentation_destination: Path, texture_path: Path):

    rendering_destination.mkdir(parents=True, exist_ok=True)
    segmentation_destination.mkdir(parents=True, exist_ok=True)

    tex = imageio.imread(str(texture_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].to(DEVICE)
    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)
    vertices = mesh.vertices[None]
    vertices *= 6.0
    faces = mesh.faces
    face_features = mesh.uvs[mesh.face_uvs_idx][None]
    translation = torch.zeros((1, 3))[None]
    width = 1506
    height = 2000
    rendering = RenderingKaolin(config, faces, width, height)
    rendering.obj_center = rendering.obj_center.to(DEVICE)
    rendering.faces = rendering.faces.to(DEVICE)
    rendering.camera_rot = rendering.camera_rot.to(DEVICE)
    rendering.camera_trans = rendering.camera_trans.to(DEVICE)
    rendering.camera_proj = rendering.camera_proj.to(DEVICE)
    i = 0
    for rotation_yaw in np.arange(0.0, 1 * 360.0 + 0.001, 10.0):
        rotation_quaternion = quaternion_from_euler(roll=torch.Tensor([0.0]),
                                                    pitch=torch.Tensor([math.pi * rotation_yaw / 180.0]),
                                                    yaw=torch.Tensor([-math.pi / 2.0]))

        rotation_matrix = quaternion_to_rotation_matrix(torch.Tensor(rotation_quaternion))[None]

        with torch.no_grad():
            face_normals, face_vertices_cam, red_index, ren_mask, \
                ren_mesh_vertices_features, ren_mesh_vertices_coords \
                = rendering.render_mesh_with_dibr(face_features.to(DEVICE), rotation_matrix.to(DEVICE),
                                                  translation.to(DEVICE), vertices.to(DEVICE))

            ren_features = kaolin.render.mesh.texture_mapping(ren_mesh_vertices_features, texture_maps, mode='bilinear')
            ren_features_np = ren_features.cpu().numpy()[0].astype('uint8')

            i_str = format(i, '03d')

            rendering_file_name = rendering_destination / (i_str + '.png')
            segmentation_file_name = segmentation_destination / (i_str + '.png')

            ren_mask_np = ren_mask.cpu().numpy().astype('uint8')[0] * 255
            ren_mask_np_rep = np.tile(ren_mask_np, (3, 1, 1)).transpose((1, 2, 0))

            imageio.imwrite(segmentation_file_name, ren_mask_np_rep)
            imageio.imwrite(rendering_file_name, ren_features_np)

        i += 1


def dataset_from_google_research(config, dataset_path: Path):
    for file in os.listdir(dataset_path):
        d = dataset_path / file
        if os.path.isdir(d):
            if file == 'original' or file == 'masks_U2Net':
                continue
            texture_path = d / Path('materials') / Path('textures') / Path('texture.png')
            model_path = d / Path('meshes') / Path('model.mtl')
            mesh_path = d / Path('meshes') / Path('model.obj')

            shutil.copyfile(texture_path, model_path.parent / texture_path.name)

            segmentation_destination = dataset_path / Path('masks_U2Net') / file
            rendering_destination = dataset_path / Path('original') / file

            generate_rotating_textured_object(config, mesh_path,
                                              rendering_destination,
                                              segmentation_destination,
                                              texture_path)



if __name__ == '__main__':
    config = load_config('configs/config_deep.yaml')
    config = types.SimpleNamespace(**config)

    # output_path = Path('data/360photo/original/concept/100')
    # segmentation_path = Path('data/360photo/masks_U2Net/concept/100')
    # generate_textured_sphere(output_path, segmentation_path)

    dataset_from_google_research(config, Path('data/GoogleScannedObjects'))

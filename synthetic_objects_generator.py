import math
import warnings
from pathlib import Path

import imageio
import kaolin
import numpy as np
import torch
from kornia.geometry import quaternion_to_rotation_matrix

from models.rendering import RenderingKaolin
from utils import quaternion_from_euler, load_config

warnings.filterwarnings("ignore")


def generate_textured_sphere(rendering_destination: Path, segmentation_destination: Path):
    DEVICE = 'cuda'

    prototype_path = Path('./prototypes/sphere_voronoi.obj')
    tex_path = Path('./prototypes/tex3.png')
    config = load_config('configs/config_deep.yaml')

    tex = imageio.imread(str(tex_path))
    texture_maps = torch.Tensor(tex).permute(2, 0, 1)[None].to(DEVICE)

    mesh = kaolin.io.obj.import_mesh(str(prototype_path), with_materials=True)
    vertices = mesh.vertices[None]
    vertices *= 2.0

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
    for rotation_yaw in np.arange(0.0, 1*360.0 + 0.001, 10.0):
        rotation_quaternion = quaternion_from_euler(roll=torch.Tensor([0.0]),
                                                    pitch=torch.Tensor([math.pi * rotation_yaw / 180.0]),
                                                    yaw=torch.Tensor([0.0]))

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


if __name__ == '__main__':
    output_path = Path('data/360photo/original/concept/100')
    segmentation_path = Path('data/360photo/masks_U2Net/concept/100')
    generate_textured_sphere(output_path, segmentation_path)

from pathlib import Path

import kaolin
import meshio
import meshzoo
import plyfile
import numpy as np
import torch
import re

import pyvista as pv

from models.initial_mesh import generate_face_features


def load_obj(path):
    """

    :param path:
    :return:
        vertices (torch.Tensor): of shape (num_vertices, 3)

        faces (torch.LongTensor): of shape (num_faces, face_size)

        uvs (torch.Tensor): of shape (num_uvs, 2): An unwrapped texture map

        face_uvs_idx (torch.LongTensor): of shape (num_faces, face_size). This is a mapping from num faces to the 2D
                                                                           texture coordinate.


        materials (list of dict): a list of materials (see return values of load_mtl())

        materials_orders (torch.LongTensor): of shape (num_same_material_groups, 2) showing the order in which materials
         are used over face_uvs_idx and the first indices in which they start to be used. A material can be used
         multiple times.

        vertex_normals (torch.Tensor): of shape (num_vertices, 3)

        face_normals (torch.LongTensor): of shape (num_faces, face_size)


    """
    print("Loading mesh located at", path)
    return kaolin.io.obj.import_mesh(path, with_materials=True)


def write_obj_mesh(vertices, faces, face_features, name, materials_model_name=None):
    if materials_model_name is None:
        materials_model_name = "model.mtl"

    file = open(name, "w")
    file.write("mtllib " + materials_model_name + "\n")
    file.write("o FMO\n")
    for ver in vertices:
        file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0], ver[1], ver[2]))
    for ffeat in face_features:
        for feat in ffeat:
            if len(feat) == 3:
                file.write("vt {:.6f} {:.6f} {:.6f} \n".format(feat[0], feat[1], feat[2]))
            else:
                file.write("vt {:.6f} {:.6f} \n".format(feat[0], feat[1]))
    file.write("usemtl Material.002\n")
    file.write("s 1\n")
    for fi in range(faces.shape[0]):
        fc = faces[fi] + 1
        ti = 3 * fi + 1
        file.write("f {}/{} {}/{} {}/{}\n".format(fc[0], ti, fc[1], ti + 1, fc[2], ti + 2))
    file.close()

import numpy as np
import torch

def generate_initial_mesh(meshsize):
    vertices, faces = meshzoo.icosa_sphere(meshsize)
    face_features = generate_face_features(vertices, faces)
    return vertices, faces, face_features


def generate_face_features(vertices, faces):
    face_features = np.zeros([faces.shape[0], 3, 2])
    for ki in range(faces.shape[0]):
        for pi in range(3):
            ind = faces[ki, pi]
            face_features[ki, pi] = [np.arctan2(-vertices[ind, 0], vertices[ind, 1]) / (2 * np.pi),
                                     np.arcsin(vertices[ind, 2]) / np.pi]
        if face_features[ki, :, 0].min() < -0.25 and face_features[ki, :, 0].max() > 0.25:
            face_features[ki, :, 0][face_features[ki, :, 0] < 0] = face_features[ki, :, 0][
                                                                       face_features[ki, :, 0] < 0] + 1
    face_features = (0.5 + face_features)  # *0.98 + 0.01
    return face_features


def sphere_to_cube(vertices):
    # ensure vertices are a unit sphere
    vertices = vertices / torch.sqrt(torch.sum(vertices**2, dim=-1, keepdim=True))

    # apply spheric coordinates
    radius = 1.0
    theta = torch.acos(vertices[..., 2] / radius)
    phi = torch.atan2(vertices[..., 1], vertices[..., 0])

    # map to cube
    xi = torch.sin(theta) * torch.cos(phi)
    yi = torch.sin(theta) * torch.sin(phi)
    zi = torch.cos(theta)

    # scale factor for each dimension
    scale = torch.pow(1. / (xi.abs() + yi.abs() + zi.abs()), 1 / 3)

    # cube vertices
    cube_vertices = torch.stack([scale * xi, scale * yi, scale * zi], dim=-1)

    return cube_vertices

import numpy as np
import torch


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


def average_face_vertex_features(faces, face_features, num_vertices=None):
    r"""Given features assigned for every vertex of every face, computes per-vertex features by
    averaging values across all faces incident each vertex.

    Args:
       faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
            shape :math:`(\text{num_faces}, \text{face_size})`.
       face_features (torch.FloatTensor): any features assigned for every vertex of every face, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, N)`.
       num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

    Return:
        (torch.FloatTensor): of shape (B, V, 3)
    """
    if num_vertices is None:
        num_vertices = int(faces.max()) + 1

    B = face_features.shape[0]
    V = num_vertices
    F = faces.shape[0]
    FSz = faces.shape[1]
    Nfeat = face_features.shape[-1]
    vertex_features = torch.zeros((B, V, Nfeat), dtype=face_features.dtype, device=face_features.device)
    counts = torch.zeros((B, V), dtype=face_features.dtype, device=face_features.device)

    faces = faces.unsqueeze(0).repeat(B, 1, 1)
    fake_counts = torch.ones((B, F), dtype=face_features.dtype, device=face_features.device)
    #              B x F          B x F x 3
    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    for i in range(FSz):
        vertex_features.scatter_add_(1, faces[..., i:i + 1].repeat(1, 1, Nfeat), face_features[..., i, :])
        counts.scatter_add_(1, faces[..., i], fake_counts)

    counts = counts.clip(min=1).unsqueeze(-1)
    vertex_normals = vertex_features / counts
    return vertex_normals

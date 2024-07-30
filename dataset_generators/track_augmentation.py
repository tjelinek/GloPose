import numpy as np
import torch


def modify_rotations(gt_rotations, seed=42):
    """
    Modify the given rotations in axis-angle representation.

    :param gt_rotations: Tensor of shape [1, N, 3] representing axis-angle rotations
    :param seed: Random seed for reproducibility
    :return: Modified rotations in axis-angle representation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Extract the number of rotations
    _, N, _ = gt_rotations.shape

    # Extract y-dimension rotations
    y_rotations = gt_rotations[0, :, 1]

    # Generate x-dimension rotations (twice as fast)
    x_rotations = 2. * y_rotations

    # Generate z-dimension rotations with random walk
    z_rotations = np.cumsum(np.deg2rad(np.random.uniform(-2, 8, N))).astype(np.float32)
    z_rotations = torch.tensor(z_rotations).cuda()

    # Combine the rotations into a new axis-angle tensor
    modified_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    modified_rotations = modified_rotations.unsqueeze(0).cuda()  # Shape [1, N, 3]

    return modified_rotations

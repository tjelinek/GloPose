import numpy as np
import torch


def modify_rotations(gt_rotations, gt_translations, seed=42):
    """
    Modify the given rotations in axis-angle representation.

    :param gt_rotations: Tensor of shape [1, N, 3] representing axis-angle rotations
    :param gt_translations: Tensor of shape [1, 1, N, 3] representing axis-angle rotations
    :param seed: Random seed for reproducibility
    :return: Modified rotations in axis-angle representation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    gt_rotations = gt_rotations.repeat(2, 1)
    gt_translations = gt_translations.repeat(2, 1)

    # Extract the number of rotations
    N, _ = gt_rotations.shape

    # Extract y-dimension rotations
    y_rotations = gt_rotations[:, 1] * 1.8

    # Generate x-dimension rotations (twice as fast)
    x_rotations = y_rotations * 0.05 * torch.sin(y_rotations * 5)

    # Generate z-dimension rotations with random walk
    z_rotations = np.cumsum(np.deg2rad(np.random.uniform(-2, 8, N))).astype(np.float32)
    z_rotations = torch.tensor(z_rotations).cuda()

    # Combine the rotations into a new axis-angle tensor
    modified_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    modified_rotations = modified_rotations.cuda()  # Shape [N, 3]

    return modified_rotations, gt_translations

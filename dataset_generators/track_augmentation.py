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
    gt_rotations = gt_rotations[:72]
    gt_translations = gt_translations[:72]

    gt_rotations = gt_rotations.repeat(2, 1)
    gt_translations = gt_translations.repeat(2, 1)

    # Extract the number of rotations
    N, _ = gt_rotations.shape

    # Extract y-dimension rotations
    y_rotations = gt_rotations[:, 1]

    # Generate x-dimension rotations (twice as fast)
    teeth = torch.deg2rad(torch.Tensor([5.])).repeat(N).cuda() * torch.Tensor([1., -1.]).repeat(N // 2).cuda()
    teeth[0] = 0
    x_rotations = y_rotations + teeth

    # Generate z-dimension rotations with random walk
    z_rotations = np.cumsum(np.deg2rad(np.random.uniform(-2, 8, N))).astype(np.float32)
    z_rotations = torch.tensor(z_rotations).cuda()

    # Combine the rotations into a new axis-angle tensor
    modified_rotations = torch.stack([x_rotations, y_rotations, z_rotations], dim=-1)
    modified_rotations = modified_rotations.cuda()  # Shape [N, 3]

    return modified_rotations, gt_translations


def modify_rotations_advanced(gt_rotations, gt_translations, seed=42):
    """
    Modify the given rotations in axis-angle representation.

    :param gt_rotations: Tensor of shape [1, N, 3] representing axis-angle rotations
    :param gt_translations: Tensor of shape [1, 1, N, 3] representing axis-angle rotations
    :param seed: Random seed for reproducibility
    :return: Modified rotations in axis-angle representation
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    gt_rotations = gt_rotations[:72]
    gt_translations = gt_translations[:72]

    gt_rotations = gt_rotations.repeat(2, 1)
    gt_translations = gt_translations.repeat(2, 1)

    # Extract the number of rotations
    N, _ = gt_rotations.shape

    # Extract y-dimension rotations
    y_rotations = gt_rotations[:, 1]
    z_rotations = gt_rotations[:, 2]

    y_rotations[N // 2: 3 * N // 4] = y_rotations[N // 2 - 1] + torch.cumsum(torch.deg2rad(torch.empty(N // 4).uniform_(-8, 2)), dim=0).cuda()
    y_rotations[3 * N // 4:] = y_rotations[3 * N // 4 - 1] + torch.cumsum(torch.deg2rad(torch.ones(N // 4) * 7),
                                                                           dim=0).cuda()

    # Generate x-dimension rotations (twice as fast)
    teeth = torch.deg2rad(torch.Tensor([5.])).repeat(N).cuda() * torch.Tensor([1., -1.]).repeat(N // 2).cuda()
    teeth2 = torch.deg2rad(torch.Tensor([5.])).repeat(N).cuda() * torch.Tensor([1., -2.]).repeat(N // 2).cuda()
    teeth[0] = 0
    x_rotations = -gt_rotations[:, 1] * 1.2 * -2. + teeth
    x_rotations[N // 2:] = x_rotations[N // 2 - 1] + torch.cumsum(torch.deg2rad(torch.empty(N // 2).uniform_(-2, 8)),
                                                                          dim=0).cuda()
    x_rotations[N // 4:] += teeth2[N // 4:]

    # Generate z-dimension rotations with random walk
    z_rotations[N // 4: N // 2] = z_rotations[N // 4 - 1] + torch.cumsum(torch.deg2rad(torch.empty(N // 4).uniform_(-2, 8)), dim=0).cuda()
    z_rotations[N // 2: 5 * N // 8] = z_rotations[N // 2 - 1] + torch.cumsum(torch.deg2rad(torch.empty(N // 8).uniform_(-6, 0)), dim=0).cuda()
    z_rotations[5 * N // 8:] = z_rotations[5 * N // 8 - 1]

    # Combine the rotations into a new axis-angle tensor
    modified_rotations = torch.stack([y_rotations, z_rotations, x_rotations], dim=-1)
    modified_rotations = modified_rotations.cuda()  # Shape [N, 3]

    return modified_rotations, gt_translations
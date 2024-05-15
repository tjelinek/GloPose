import os

import torch
from torchvision.utils import save_image
import numpy as np


def sync_directions_rgba(est_hs):
    for frmi in range(1, est_hs.shape[0]):
        tsr0 = est_hs[frmi - 1]
        tsr = est_hs[frmi]
        if frmi == 1:
            forward = np.min([torch.mean((tsr0[-1] - tsr[-1]) ** 2), torch.mean((tsr0[-1] - tsr[0]) ** 2)])
            backward = np.min([torch.mean((tsr0[0] - tsr[-1]) ** 2), torch.mean((tsr0[0] - tsr[0]) ** 2)])
            if backward < forward:
                est_hs[frmi - 1] = torch.flip(est_hs[frmi - 1], [0])
                tsr0 = est_hs[frmi - 1]

        if torch.mean((tsr0[-1] - tsr[-1]) ** 2) < torch.mean((tsr0[-1] - tsr[0]) ** 2):
            # reverse time direction for better alignment
            est_hs[frmi] = torch.flip(est_hs[frmi], [0])
    return est_hs


def write_renders(renders, tmp_folder, nrow=8, ids=None, im_name_base='im_recon'):
    name = im_name_base + '.png'
    if ids is not None:
        name = im_name_base + '{}.png'.format(ids)
    save_image(renders[0], os.path.join(tmp_folder, name), nrow=nrow)

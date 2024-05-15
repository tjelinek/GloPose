import os

import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from skimage.measure import label, regionprops
import cv2
import numpy as np
from PIL import Image


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


class SRWriter:
    def __init__(self, imtemp, path, available_gt=True):
        self.available_gt = available_gt
        if self.available_gt:
            fctr = 3
        else:
            fctr = 2
        if imtemp.shape[0] > imtemp.shape[1]:
            self.width = True
            shp = (imtemp.shape[0], imtemp.shape[1] * fctr, 3)
            self.value = imtemp.shape[1]
        else:
            self.width = False
            shp = (imtemp.shape[0] * fctr, imtemp.shape[1], 3)
            self.value = imtemp.shape[0]
        self.video = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 12, (shp[1], shp[0]), True)
        self.img = np.zeros(shp)

    def update_ls(self, lsf):
        if self.width:
            self.img[:, :self.value] = lsf
        else:
            self.img[:self.value, :] = lsf

    def write_next(self, hs, est):
        if hs is not None:
            if self.width:
                self.img[:, 2 * self.value:] = hs
            else:
                self.img[2 * self.value:, :] = hs
        if est is not None:
            if self.width:
                self.img[:, self.value:2 * self.value] = est
            else:
                self.img[self.value:2 * self.value, :] = est
        self.img[self.img > 1] = 1
        self.img[self.img < 0] = 0
        self.video.write((self.img.copy() * 255)[:, :, [2, 1, 0]].astype(np.uint8))

    def close(self):
        self.video.release()
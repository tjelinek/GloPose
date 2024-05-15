from pathlib import Path

from torchvision.utils import save_image


def write_renders(renders, tmp_folder, nrow=8, ids=None, im_name_base='im_recon'):
    name = im_name_base + '.png'
    if ids is not None:
        name = im_name_base + '{}.png'.format(ids)
    save_image(renders[0], Path(tmp_folder) / name, nrow=nrow)

import glob
import os
from argparse import Namespace
from typing import Iterable

import imageio
import torch

from GMA.core.network import RAFTGMA
from GMA.core.utils import flow_viz
from GMA.core.utils.utils import InputPadder
from GMA.evaluate_single import load_image
from cfg import *


args = Namespace(model='checkpoints/gma-sintel.pth', model_name='GMA', path=None, num_heads=1, position_only=False,
                 position_and_content=False)

model = torch.nn.DataParallel(RAFTGMA(args=args))
checkpoint_path = Path('../GMA/checkpoints/gma-sintel.pth')
model.load_state_dict(torch.load(checkpoint_path))
print(f"Loaded checkpoint at {checkpoint_path}")

model = model.module
model.to(DEVICE)
model.eval()


def get_flow_from_images(image1, image2):
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)

    return flow_low, flow_up


def get_flow_from_sequence(images_pairs: Iterable):
    for image1, image2 in images_pairs:
        yield get_flow_from_images(image1, image2)


def get_flow_from_files(files_source_dir: Path):

    images = glob.glob(os.path.join(files_source_dir, '*.png')) + \
             glob.glob(os.path.join(files_source_dir, '*.jpg'))

    images = sorted(images)

    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        flow_low, flow_up = get_flow_from_images(image1, image2)

        yield (imfile1, imfile2), (flow_low, flow_up)


def export_flow_from_files(files_source_dir: Path, flows_target_dir: Path = FLOW_OUT_DEFAULT_DIR):
    flows_target_dir.mkdir(exist_ok=True, parents=True)

    for (filename1, filename2), (flow_low, flow_up) in get_flow_from_files(files_source_dir):
        # flow_low = flow_low[0].permute(1, 2, 0).cpu().numpy()
        flow_up = flow_up[0].permute(1, 2, 0).cpu().numpy()

        # map flow to rgb image
        flow_up = flow_viz.flow_to_image(flow_up)

        flow_image_path = flows_target_dir / Path('flow_' + filename1 + '_' + filename2 + '.png')
        imageio.imwrite(flow_image_path, flow_up)


if __name__ == "__main__":

    path_to_dataset = Path("data/coin_tracking/images/pingpong1")
    export_flow_from_files(path_to_dataset)


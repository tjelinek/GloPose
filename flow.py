import glob
import os

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import imageio
import torchvision.transforms as T

from pathlib import Path
from typing import Iterable

from GMA.core.utils import flow_viz
from GMA.core.utils.utils import InputPadder
from cfg import FLOW_OUT_DEFAULT_DIR, DEVICE


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def get_flow_from_files(files_source_dir: Path, model):
    images = glob.glob(os.path.join(files_source_dir, '*.png')) + \
             glob.glob(os.path.join(files_source_dir, '*.jpg'))

    images = sorted(images)

    for imfile1, imfile2 in zip(images[:-1], images[1:]):
        image1 = load_image(imfile1)
        image2 = load_image(imfile2)

        flow_low, flow_up = get_flow_from_images(image1, image2, model)

        # print(flow_up.shape, flow_low.shape)

        yield (imfile1, imfile2), (flow_low, flow_up)


def visualize_flow_with_images(image1, image2, flow):
    width, height = image1.shape[-1], image1.shape[-2]

    # Tensors to PIL Images
    transform = T.ToPILImage()
    image1 = transform(image1[0] * 255.0)
    image2 = transform(image2[0] * 255.0)
    flow = transform(flow)

    draw1 = ImageDraw.Draw(image1)
    draw2 = ImageDraw.Draw(image2)
    step = 100

    r = 5 # radius of the drawn point

    for x in range(step, width, step):
        for y in range(step, height, step):
            draw1.ellipse((x - r, y - r, x + r, y + r), fill='red')
            draw2.ellipse((x - r, y - r, x + r, y + r), fill='red')

    canvas = Image.new('RGBA', (width * 3, height), (255, 255, 255, 255))
    canvas.paste(image1, (0, 0))
    canvas.paste(flow, (width, 0))
    canvas.paste(image2, (2 * width, 0))

    return canvas


def export_flow_from_files(files_source_dir: Path, model, flows_target_dir: Path = FLOW_OUT_DEFAULT_DIR):
    flows_target_dir.mkdir(exist_ok=True, parents=True)

    for (filename1, filename2), (flow_low, flow_up) in get_flow_from_files(files_source_dir, model):
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
        flow_low = flow_low[0].permute(1, 2, 0).cpu().detach().numpy()

        # map flow to rgb image
        flow_up = flow_viz.flow_to_image(flow_up)
        flow_low = flow_viz.flow_to_image(flow_low)

        file1_stem = Path(filename1).stem
        file2_stem = Path(filename2).stem

        image1 = load_image(filename1)
        image2 = load_image(filename2)

        flow = visualize_flow_with_images(image1, image2, flow_up)

        flow_image_path = flows_target_dir / Path('flow_' + file1_stem + '_' + file2_stem + '.png')
        print("Writing flow to ", flow_image_path)
        imageio.imwrite(flow_image_path, flow)


def get_flow_from_sequence(images_pairs: Iterable, model):
    for image1, image2 in images_pairs:
        yield get_flow_from_images(image1, image2, model)


def get_flow_from_images(image1, image2, model):
    padder = InputPadder(image1.shape)

    image1, image2 = padder.pad(image1, image2)
    height = image1.size()[-2]
    width = image2.size()[-1]
    transposed = False

    # print(image1.shape)

    if height > width:
        transposed = True
        image1 = image1.transpose(-1, -2)
        image2 = image2.transpose(-1, -2)
        width, height = height, width

    resizer = torchvision.transforms.Resize((1024, 440))

    image1 = resizer(image1)
    image2 = resizer(image2)

    flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)

    flow_low = torchvision.transforms.Resize((height, width))(flow_low)
    flow_up = torchvision.transforms.Resize((height, width))(flow_up)

    if transposed:
        width, height = height, width
        flow_low = flow_low.transpose(-1, -2)
        flow_up = flow_up.transpose(-1, -2)

    flow_low = padder.unpad(flow_low)
    flow_up = padder.unpad(flow_up)

    return flow_low, flow_up

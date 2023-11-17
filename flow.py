import glob
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import imageio
import torchvision.transforms as T

from pathlib import Path
from typing import Iterable
from argparse import Namespace
from abc import ABC, abstractmethod

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

        yield (imfile1, imfile2), (flow_low, flow_up)


def compare_flows_with_images(image1, image2, flow_up, flow_up_prime,
                              gt_silhouette_current=None, gt_silhouette_prev=None):
    """
    Visualizes flow_up and flow_up_prime along with image1 and image2.

    :param image1: uint8 image with 0-255 as color in [C, H, W] format
    :param image2: uint8 image with 0-255 as color in [C, H, W] format
    :param flow_up: np.ndarray [H, W, 2] flow
    :param flow_up_prime: np.ndarray [H, W, 2] flow
    :param gt_silhouette_current: Silhouette of the ground truth in the current frame
    :param gt_silhouette_prev: Silhouette of the ground truth in the previous frame
    :return: PIL Image
    """
    return visualize_flow_with_images(image1, image2, flow_up, flow_up_prime, gt_silhouette_current, gt_silhouette_prev)


def tensor_to_pil_with_alpha(tensor, alpha=0.2):
    # Convert tensor to numpy array and scale to 255 if necessary
    array = (tensor.numpy() * 255).astype(np.uint8)

    # Create an RGBA image with the given alpha value for the segmentation
    rgba_array = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    rgba_array[..., :1] = array[:, :, None]  # Assign the tensor values to red and green channels
    rgba_array[..., 3] = (array > 0) * alpha * 256.0  # Create alpha channel based on tensor values

    # Convert to PIL image
    img = Image.fromarray(rgba_array, 'RGBA')

    return img


def visualize_flow_with_images(image1, image2, flow_up, flow_up_prime,
                               gt_silhouette_current=None, gt_silhouette_prev=None):
    """

    :param image1: uint8 image with 0-255 as color in [C, H, W] format
    :param image2: uint8 image with 0-255 as color in [C, H, W] format
    :param flow_up: np.ndarray [H, W, 2] flow
    :param flow_up_prime: np.ndarray [H, W, 2] flow
    :param gt_silhouette_current: Tensor - silhouette of the ground truth in the current frame of shape [H, W]
    :param gt_silhouette_prev: Tensor - silhouette of the ground truth in the previous frame of shape [H, W]
    :return:
    """
    width, height = image1.shape[-1], image1.shape[-2]

    # Tensors to PIL Images
    transform = T.ToPILImage()
    image1_pil = transform(image1)
    image2_pil = transform(image2)

    if gt_silhouette_prev is not None:
        silh1_PIL = tensor_to_pil_with_alpha(gt_silhouette_prev, alpha=0.25)
        background1_PIL = Image.new('RGBA', image1_pil.size, (0, 0, 0, 0))
        background1_PIL.paste(silh1_PIL, (0, 0))
        background1_PIL.paste(image1_pil, (0, 0))
        image1_pil = background1_PIL
    if gt_silhouette_current is not None:
        silh2_PIL = tensor_to_pil_with_alpha(gt_silhouette_current, alpha=0.25)
        background2_PIL = Image.new('RGBA', image2_pil.size, (0, 0, 0, 0))
        background2_PIL.paste(silh2_PIL, (0, 0))
        background2_PIL.paste(image2_pil, (0, 0))
        image2_pil = background2_PIL

    flow_pil = None
    if flow_up is not None:
        if flow_up_prime is not None:
            flow_up_image = flow_viz.flow_to_image(np.concatenate([flow_up, flow_up_prime], axis=1))
        else:
            flow_up_image = flow_viz.flow_to_image(flow_up)
        flow_pil = transform(flow_up_image)

    flow_prime_pil = None
    if flow_up_prime is not None:
        flow_up_prime_image = flow_viz.flow_to_image(flow_up_prime)
        flow_prime_pil = transform(flow_up_prime_image)

    draw1 = ImageDraw.Draw(image1_pil)
    draw2 = ImageDraw.Draw(image2_pil)

    step = max(width, height) // 20

    r = max(height // 400, 1)  # radius of the drawn point

    for y in range(step, height, step):
        for x in range(step, width, step):
            draw1.ellipse((x - r, y - r, x + r, y + r), fill='black')
            draw2.ellipse((x - r, y - r, x + r, y + r), fill='black')

            if flow_up is not None:
                shift_up_x = flow_up[y, x, 0]
                shift_up_y = flow_up[y, x, 1]

                draw2.line((x, y, x + shift_up_x, y + shift_up_y), fill='red')

                draw2.line((x + shift_up_x, y + shift_up_y - r, x + shift_up_x, y + shift_up_y + r),
                           fill='red')
                draw2.line((x + shift_up_x - r, y + shift_up_y, x + shift_up_x + r, y + shift_up_y),
                           fill='red')
            if flow_up_prime is not None:
                shift_up_prime_x = flow_up_prime[y, x, 0]
                shift_up_prime_y = flow_up_prime[y, x, 1]

                draw2.line((x, y, x + shift_up_prime_x, y + shift_up_prime_y), fill='green')

                draw2.line((x + shift_up_prime_x - r, y + shift_up_prime_y - r, x + shift_up_prime_x + r,
                            y + shift_up_prime_y + r), fill='green')
                draw2.line((x + shift_up_prime_x - r, y + shift_up_prime_y + r, x + shift_up_prime_x + r,
                            y + shift_up_prime_y - r), fill='green')

    canvas_width_coef = int(flow_pil is not None) + int(flow_prime_pil is not None) + 2
    canvas = Image.new('RGBA', (width * canvas_width_coef, height), (255, 255, 255, 255))
    canvas.paste(image1_pil, (0, 0))
    if flow_pil is not None:
        canvas.paste(flow_pil, (width, 0))
    elif flow_prime_pil is not None and flow_pil is None:
        canvas.paste(flow_prime_pil, (width, 0))
    canvas.paste(image2_pil, ((canvas_width_coef - 1) * width, 0))

    return canvas


def export_flow_from_files(files_source_dir: Path, model, flows_target_dir: Path = FLOW_OUT_DEFAULT_DIR):
    flows_target_dir.mkdir(exist_ok=True, parents=True)

    for (filename1, filename2), (flow_low, flow_up) in get_flow_from_files(files_source_dir, model):
        flow_up = flow_up[0].permute(1, 2, 0).cpu().detach().numpy()
        # flow_low = flow_low[0].permute(1, 2, 0).cpu().detach().numpy()

        # map flow to rgb image
        flow_up_img = flow_viz.flow_to_image(flow_up)
        # flow_low_img = flow_viz.flow_to_image(flow_low)

        file1_stem = Path(filename1).stem
        file2_stem = Path(filename2).stem

        image1 = load_image(filename1)
        image2 = load_image(filename2)

        flow = visualize_flow_with_images(image1[0] * 255, image2[0] * 255, flow_up)

        flow_image_path = flows_target_dir / Path('flow_' + file1_stem + '_' + file2_stem + '.png')
        print("Writing flow to ", flow_image_path)
        imageio.imwrite(flow_image_path, flow)
        pure_flow_image_path = flows_target_dir / Path('flow_pure_' + file1_stem + '_' + file2_stem + '.png')
        imageio.imwrite(pure_flow_image_path, flow_up_img)


def get_flow_from_sequence(images_pairs: Iterable, model):
    for image1, image2 in images_pairs:
        yield get_flow_from_images(image1, image2, model)


def get_flow_from_images(image1, image2, model):
    padder = InputPadder(image1.shape)

    image1, image2 = padder.pad(image1, image2)

    flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)

    flow_low = padder.unpad(flow_low)
    flow_up = padder.unpad(flow_up)

    return flow_low, flow_up


def get_flow_from_images_mft(image1, image2, model):
    padder = InputPadder(image1.shape)

    image1, image2 = padder.pad(image1, image2)

    all_predictions = model(image1, image2, iters=12, test_mode=True)

    flow = padder.unpad(all_predictions['flow'])
    occlusion = padder.unpad(all_predictions['occlusion'].softmax(dim=1)[:, 1:2, :, :])
    uncertainty = padder.unpad(all_predictions['uncertainty'])

    return flow, occlusion, uncertainty


def get_flow_from_images_raft(image1, image2, model):
    with torch.no_grad():
        padder = InputPadder(image1.shape)

        image1, image2 = padder.pad(image1, image2)
        height = image1.size()[-1]
        width = image2.size()[-2]
        transposed = False

        if height > width:
            transposed = True
            image1 = image1.transpose(-1, -2)
            image2 = image2.transpose(-1, -2)
            width, height = height, width

        resizer = torchvision.transforms.Resize((1024, 440))

        height_scale = height / 440
        width_scale = width / 1024

        image1 = resizer(image1)
        image2 = resizer(image2)

        flow_low, flow_up = model(image1, image2, iters=12, test_mode=True)

        flow_low = torchvision.transforms.Resize((width, height))(flow_low)
        flow_up = torchvision.transforms.Resize((width, height))(flow_up)

        if transposed:
            flow_low = flow_low.transpose(-1, -2)
            flow_up = flow_up.transpose(-1, -2)

            flow_low[:, :, 0], flow_low[:, :, 1] = flow_low[:, :, 1], flow_low[:, :, 0]
            flow_up[:, :, 0], flow_up[:, :, 1] = flow_up[:, :, 1], flow_up[:, :, 0]

        flow_low = padder.unpad(flow_low)

        flow_up[:, :, 0] *= width_scale
        flow_up[:, :, 1] *= height_scale

        flow_up = padder.unpad(flow_up)

    return flow_low, flow_up


class FlowModelGetter(ABC):

    @staticmethod
    @abstractmethod
    def get_flow_model():
        raise NotImplementedError

    @staticmethod
    def prepare_model(args, model):
        model.load_state_dict(torch.load(args.model))
        print(f"Loaded checkpoint at {args.model}")
        model = model.module
        model.to(DEVICE)
        model.eval()
        return model


class FlowModelGetterRAFT(FlowModelGetter):

    @staticmethod
    def get_flow_model():
        sys.path.append('RAFTPrinceton')
        from RAFTPrinceton.core.raft import RAFT

        args = Namespace(model='tmp/raft_models/models/raft-things.pth', model_name='RAFTPrinceton', path=None,
                         mixed_precision=True,
                         alternate_corr=False, small=False)

        model = torch.nn.DataParallel(RAFT(args))
        model = FlowModelGetter.prepare_model(args, model)

        return model


class FlowModelGetterGMA(FlowModelGetter):

    @staticmethod
    def get_flow_model():
        sys.path.append('GMA')
        from GMA.core.network import RAFTGMA

        args = Namespace(model='GMA/checkpoints/gma-sintel.pth', model_name='GMA', path=None, num_heads=1,
                         position_only=False,
                         position_and_content=False, mixed_precision=True)

        model = torch.nn.DataParallel(RAFTGMA(args=args))
        model = FlowModelGetter.prepare_model(args, model)

        return model


class FlowModelGetterMFT(FlowModelGetter):
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(FlowModelGetterMFT.AttrDict, self).__init__(*args, **kwargs)
            self.__dict__.update(kwargs)

    @staticmethod
    def get_flow_model():
        # sys.path.append('MFTmaster.MFT')
        sys.path.append('MFTmaster')
        import MFTmaster

        default_config = MFTmaster.configs.MFT_cfg.get_config()

        model = MFTmaster.MFT.MFT.MFT(default_config)

        return model

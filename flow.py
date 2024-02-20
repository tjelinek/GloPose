import contextlib
import importlib
import os
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch
import torchvision
import torchvision.transforms as T

from argparse import Namespace
from abc import ABC, abstractmethod

from GMA.core.utils import flow_viz
from GMA.core.utils.utils import InputPadder
from cfg import DEVICE


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
    rgba_array[..., :3] = array[:, :, None]  # Assign the tensor values to red and green channels
    rgba_array[..., 3] = (array > 0) * alpha * 256.0  # Create alpha channel based on tensor values

    # Convert to PIL image
    img = Image.fromarray(rgba_array, 'RGBA')

    return img


def visualize_flow_with_images(image1, image2, flow_up, flow_up_prime=None, gt_silhouette_current=None,
                               gt_silhouette_prev=None, flow_occlusion_mask=None):
    """

    :param image1: uint8 image with [0-1] color range in [C, H, W] format
    :param image2: uint8 image with [0-1] color range in [C, H, W] format
    :param flow_up: np.ndarray [H, W, 2] flow
    :param flow_up_prime: np.ndarray [H, W, 2] flow
    :param gt_silhouette_current: Tensor - silhouette of the ground truth in the current frame of shape [H, W]
    :param gt_silhouette_prev: Tensor - silhouette of the ground truth in the previous frame of shape [H, W]
    :param flow_occlusion_mask: Tensor - indicating flow occlusion in format [H, W]
    :return:
    """
    width, height = image1.shape[-1], image1.shape[-2]

    # Tensors to PIL Images
    transform = T.ToPILImage()
    image1_pil = transform(image1)
    image2_pil = transform(image2)

    if flow_occlusion_mask is not None:
        occlusion_blend = int(255 * 0.25)
        occlusion_mask_pil = tensor_to_pil_with_alpha(flow_occlusion_mask, alpha=1.).convert("L")
        silver_image = Image.new('RGBA', image1_pil.size, (255, 255, 255, occlusion_blend))
        image1_pil.paste(silver_image, (0, 0), mask=occlusion_mask_pil)

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


def normalize_flow_to_unit_range(observed_flow):
    observed_flow[:, :, 0, ...] = observed_flow[:, :, 0, ...] / observed_flow.shape[-2]
    observed_flow[:, :, 1, ...] = observed_flow[:, :, 1, ...] / observed_flow.shape[-1]

    return observed_flow


def flow_unit_coords_to_image_coords(observed_flow: torch.Tensor):
    observed_flow[:, :, 0, ...] *= observed_flow.shape[-1]
    observed_flow[:, :, 1, ...] *= observed_flow.shape[-2]

    return observed_flow


def optical_flow_to_matched_coords(flow: torch.Tensor, step=1):
    height, width = flow.shape[-2], flow.shape[-1]
    x, y = torch.meshgrid(torch.arange(0, width, step), torch.arange(0, height, step))
    coords = torch.stack((x, y), dim=0).float().to(flow.device)  # Shape: [2, height, width]

    coords = coords.unsqueeze(0).unsqueeze(0)

    matched_coords = coords + flow[:, :, :, ::step, ::step]

    return matched_coords


def tensor_image_to_mft_format(image_tensor):
    return image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()


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


class FlowProvider(ABC):

    def __init__(self):
        super().__init__()
        self.flow_model = self.get_flow_model()

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

    def next_flow(self, source_image, target_image):
        padder = InputPadder(source_image.shape)

        image1, image2 = padder.pad(source_image, target_image)

        flow_low, flow_up = self.model(image1, image2, iters=12, test_mode=True)

        # flow_low = padder.unpad(flow_low)[None]
        flow_up = padder.unpad(flow_up)[None]

        return flow_up


class RAFTFlowProvider(FlowProvider):

    @staticmethod
    def get_flow_model():
        sys.path.append('RAFTPrinceton')
        from RAFTPrinceton.core.raft import RAFT

        args = Namespace(model='tmp/raft_models/models/raft-things.pth', model_name='RAFTPrinceton', path=None,
                         mixed_precision=True,
                         alternate_corr=False, small=False)

        model = torch.nn.DataParallel(RAFT(args))
        model = FlowProvider.prepare_model(args, model)

        return model


class GMAFlowProvider(FlowProvider):

    @staticmethod
    def get_flow_model():
        sys.path.append('GMA')
        from GMA.core.network import RAFTGMA

        args = Namespace(model='GMA/checkpoints/gma-sintel.pth', model_name='GMA', path=None, num_heads=1,
                         position_only=False,
                         position_and_content=False, mixed_precision=True)

        model = torch.nn.DataParallel(RAFTGMA(args=args))
        model = FlowProvider.prepare_model(args, model)

        return model


@contextlib.contextmanager
def temporary_change_directory(new_directory):
    original_directory = os.getcwd()
    try:
        os.chdir(new_directory)
        yield
    finally:
        os.chdir(original_directory)


class MFTFlowProvider(FlowProvider):

    def __init__(self):
        super().__init__()
        self.add_to_path()
        from MFT_tracker.MFT.MFT import MFT as MFTTracker
        self.need_to_init = True
        self.flow_model: MFTTracker = self.get_flow_model()

    @staticmethod
    def add_to_path():
        if 'MFT_tracker' not in sys.path:
            sys.path.append('MFT_tracker')

    def init(self, template):
        template_mft = tensor_image_to_mft_format(template)
        self.flow_model.init(template_mft)

    def next_flow(self, source_image, target_image):
        # source_image_mft = tensor_image_to_mft_format(source_image)
        target_image_mft = tensor_image_to_mft_format(target_image)

        all_predictions = self.flow_model.track(target_image_mft)

        flow = all_predictions.result.flow.cuda()[None, None]
        occlusion = all_predictions.result.occlusion.cuda()[None, None]
        sigma = all_predictions.result.sigma.cuda()[None, None]

        return flow, occlusion, sigma

    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(MFTFlowProvider.AttrDict, self).__init__(*args, **kwargs)
            self.__dict__.update(kwargs)

    @staticmethod
    def get_flow_model(config_name='MFT_cfg'):
        MFTFlowProvider.add_to_path()

        from MFT_tracker.MFT.MFT import MFT as MFTTracker
        config_module = importlib.import_module(f'MFT_tracker.configs.{config_name}')

        with temporary_change_directory("MFT_tracker"):
            default_config = config_module.get_config()
            model = MFTTracker(default_config)

        return model


class MFTEnsembleFlowProvider(MFTFlowProvider):
    def __init__(self):
        super().__init__()
        MFTFlowProvider.add_to_path()
        from MFT_tracker.MFT.MFT_ensemble import MFTEnsemble as MFTTracker
        self.need_to_init = True
        self.flow_model: MFTTracker = self.get_flow_model()

    @staticmethod
    def get_flow_model(config_name='MFT_RoMa_cfg'):
        MFTFlowProvider.add_to_path()

        from MFT_tracker.MFT.MFT_ensemble import MFTEnsemble as MFTTracker
        config_module = importlib.import_module(f'MFT_tracker.configs.{config_name}')

        with temporary_change_directory("MFT_tracker"):
            default_config = config_module.get_config()
            model = MFTTracker(default_config)

        return model

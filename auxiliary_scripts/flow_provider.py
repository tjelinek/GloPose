import contextlib
import importlib
import os
import sys

import torch

from argparse import Namespace
from abc import ABC, abstractmethod

import torchvision
import torch.nn.functional as F

from data_structures.keyframe_buffer import FrameObservation, FlowObservation
from flow import tensor_image_to_mft_format
from tracker_config import TrackerConfig


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class FlowProvider(ABC):

    def __init__(self, get_model=True, **kwargs):
        super().__init__()
        self.tracker_config: TrackerConfig = kwargs['config']
        if get_model:
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
        model.to('cuda')
        model.eval()
        return model

    @abstractmethod
    def next_flow(self, source_image, target_image):
        raise NotImplementedError

    def next_flow_observation(self, source_image: FrameObservation, target_image: FrameObservation) -> FlowObservation:
        flow, occl, sigma = self.next_flow(source_image.observed_image.float() * 255,
                                           target_image.observed_image.float() * 255)

        flow_observation = FlowObservation(observed_flow=flow,
                                           observed_flow_segmentation=source_image.observed_segmentation,
                                           observed_flow_uncertainty=sigma, observed_flow_occlusion=occl,
                                           flow_source_frames=[0], flow_target_frames=[1], coordinate_system='image')
        flow_observation = flow_observation.cast_image_coords_to_unit_coords()

        return flow_observation


class RAFTFlowProvider(FlowProvider):

    @staticmethod
    def get_flow_model():
        sys.path.append('repositories/RAFTPrinceton')
        from repositories.RAFTPrinceton.core.raft import RAFT

        args = Namespace(model='/mnt/personal/jelint19/weights/RAFT/raft-things.pth',
                         model_name='RAFTPrinceton',
                         path=None,
                         mixed_precision=True,
                         alternate_corr=False, small=False)

        model = torch.nn.DataParallel(RAFT(args))
        model = FlowProvider.prepare_model(args, model)

        return model

    def next_flow(self, source_image, target_image):
        padder = InputPadder(source_image.shape)

        image1, image2 = padder.pad(source_image, target_image)

        flow_low, flow_up = self.model(image1, image2, iters=12, test_mode=True)

        # flow_low = padder.unpad(flow_low)[None]
        flow_up = padder.unpad(flow_up)[None]

        occlusion = torch.zeros(1, 1, 1, *flow_up.shape[-2:]).to(flow_up.device)
        uncertainty = torch.zeros(1, 1, 1, *flow_up.shape[-2:]).to(flow_up.device)

        return flow_up, occlusion, uncertainty


class RoMaFlowProvider(FlowProvider):

    def __init__(self, config_name, **kwargs):
        super().__init__(**kwargs, get_model=False)
        self.add_to_path()
        from repositories.MFT_tracker.MFT.roma import RoMaWrapper
        self.flow_model: RoMaWrapper = self.get_flow_model(config_name)

    @staticmethod
    def add_to_path():
        if 'MFT_tracker' not in sys.path:
            sys.path.append('repositories/MFT_tracker')

    def init(self, template):
        pass

    def next_flow(self, source_image, target_image):
        source_image_mft = tensor_image_to_mft_format(source_image)
        target_image_mft = tensor_image_to_mft_format(target_image)

        flow, extra = self.flow_model.compute_flow(source_image_mft, target_image_mft, mode='flow')

        flow = flow.cuda()[None, None]
        occlusion = extra['occlusion'].cuda()[None, None]
        sigma = extra['sigma'].cuda()[None, None]

        return flow, occlusion, sigma

    def next_flow_raw(self, source_image, target_image, sample=None):
        source_image_roma = torchvision.transforms.functional.to_pil_image(source_image.squeeze())
        target_image_roma = torchvision.transforms.functional.to_pil_image(target_image.squeeze())

        warp, certainty = self.flow_model.model.match(source_image_roma, target_image_roma, device='cuda')
        if sample:
            warp, certainty = self.flow_model.model.sample(warp, certainty, sample)

        return warp, certainty

    def next_flow_roma_src_pts_xy(self, source_image, target_image, image_height, image_width, sample=None):

        matches, certainty = self.next_flow_raw(source_image, target_image, sample)
        src_pts_xy, dst_pts_xy = self.flow_model.model.to_pixel_coordinates(matches, image_height, image_width,
                                                                            image_height, image_width)

        return src_pts_xy, dst_pts_xy

    def get_flow_model(self, config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.roma import RoMaWrapper
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
            default_config.occlusion_threshold = self.tracker_config.occlusion_coef_threshold
            model = RoMaWrapper(default_config.flow_config)

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

    def __init__(self, config_name, **kwargs):
        super().__init__(**kwargs, get_model=False)
        self.add_to_path()
        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        self.need_to_init = True
        self.flow_model: MFTTracker = self.get_flow_model(config_name)

    @staticmethod
    def add_to_path():
        if 'MFT_tracker' not in sys.path:
            sys.path.append('repositories/MFT_tracker')

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

    def get_flow_model(self, config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
            default_config.occlusion_threshold = self.tracker_config.occlusion_coef_threshold
            model = MFTTracker(default_config)

        return model


class MFTIQFlowProvider(FlowProvider):

    def __init__(self, config_name, **kwargs):
        self.add_to_path()
        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        self.need_to_init = True
        self.flow_model: MFTTracker = self.get_flow_model(config_name)

    @staticmethod
    def add_to_path():
        if 'MFT_tracker' not in sys.path:
            sys.path.append('repositories/MFT_tracker')

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

    def get_flow_model(self, config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
            default_config.occlusion_threshold = self.tracker_config.occlusion_coef_threshold
            model = default_config.tracker_class(default_config)  #

        return model


class MFTIQSyntheticFlowProvider(MFTIQFlowProvider):

    def __init__(self, config_name, **kwargs):
        super().__init__(config_name, **kwargs, get_model=False)
        self.add_to_path()
        from repositories.MFT_tracker.MFT.MFTIQ import MFT as MFTTracker
        from repositories.MFT_tracker.MFT.gt_flow import GTFlowWrapper

        self.need_to_init = True
        self.config: TrackerConfig = kwargs['config']
        self.faces = kwargs['faces']
        self.gt_encoder = kwargs['gt_encoder']

        self.flow_model: MFTTracker = self.get_flow_model(config_name)
        # TODO this does not work for some reason
        # if not isinstance(self.flow_model.flower, GTFlowWrapper):
        #     breakpoint()
        #     raise ValueError("Something went wrong, the flower of MFT must be GTFlowWrapper")

    def init(self, template):
        template_mft = tensor_image_to_mft_format(template)
        self.flow_model.flower.initialize_renderer(self.config, self.gt_encoder, self.faces)
        self.flow_model.init(template_mft)


class MFTEnsembleFlowProvider(FlowProvider):
    def __init__(self, config_name, **kwargs):
        self.add_to_path()
        from repositories.MFT_tracker.MFT.MFT_ensemble import MFTEnsemble as MFTTracker
        self.need_to_init = True
        self.flow_model: MFTTracker = self.get_flow_model(config_name)

    @staticmethod
    def add_to_path():
        if 'MFT_tracker' not in sys.path:
            sys.path.append('repositories/MFT_tracker')

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

    def get_flow_model(self, config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT_ensemble import MFTEnsemble as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
            default_config.occlusion_threshold = self.tracker_config.occlusion_coef_threshold
            model = MFTTracker(default_config)

        return model
import contextlib
import importlib
import os
import sys

import torch

from argparse import Namespace
from abc import ABC, abstractmethod

from flow import tensor_image_to_mft_format
from repositories.GMA.core.utils.utils import InputPadder
from tracker_config import TrackerConfig


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

        return flow_up


class GMAFlowProvider(FlowProvider):

    @staticmethod
    def get_flow_model():
        sys.path.append('repositories/GMA')
        from repositories.GMA.core.network import RAFTGMA

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

    @staticmethod
    def get_flow_model(config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
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

    @staticmethod
    def get_flow_model(config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT import MFT as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
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

    @staticmethod
    def get_flow_model(config_name=None):
        MFTFlowProvider.add_to_path()

        from repositories.MFT_tracker.MFT.MFT_ensemble import MFTEnsemble as MFTTracker
        config_module = importlib.import_module(f'repositories.MFT_tracker.configs.{config_name}')

        with temporary_change_directory("repositories/MFT_tracker"):
            default_config = config_module.get_config()
            model = MFTTracker(default_config)

        return model

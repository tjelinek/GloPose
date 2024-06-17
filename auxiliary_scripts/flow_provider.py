from abc import ABC, abstractmethod

import torch

from flow import normalize_rendered_flows, MFTFlowProvider, FlowProvider


class FlowStrategy(ABC):
    @abstractmethod
    def compute_flow(self, *args, **kwargs):
        pass


class SyntheticFlowStrategy(FlowStrategy):
    @abstractmethod
    def compute_flow(self, flow_source_index, flow_target_index, backview=False):
        pass


class ExternalMethodFlowStrategy(FlowStrategy):
    @abstractmethod
    def compute_flow(self, flow_source_image, flow_target_image, backview=False):
        pass


class GenerateSyntheticFlowStrategy(SyntheticFlowStrategy):
    def __init__(self, renderer, renderer_backview, gt_encoder, shape):
        self.renderer = renderer
        self.renderer_backview = renderer_backview
        self.gt_encoder = gt_encoder
        self.shape = shape

    @torch.no_grad()
    def compute_flow(self, flow_source_frame, flow_target_frame, backview=False):
        keyframes = [flow_target_frame]
        flow_frames = [flow_source_frame]
        flow_arcs_indices = [(0, 0)]

        encoder_result, enc_flow = self.gt_encoder.frames_and_flow_frames_inference(keyframes, flow_frames)

        renderer = self.renderer_backview if backview else self.renderer
        observed_renderings = renderer.compute_theoretical_flow(encoder_result, enc_flow, flow_arcs_indices)

        observed_flow = observed_renderings.theoretical_flow.detach()
        observed_flow = normalize_rendered_flows(observed_flow, self.renderer.width, self.renderer.height,
                                                 self.shape[-1], self.shape[-2])
        occlusion = observed_renderings.rendered_flow_occlusion.detach()
        uncertainty = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)

        return observed_flow, occlusion, uncertainty


class ShortFlowStrategy(ExternalMethodFlowStrategy):
    def __init__(self, short_flow_model_name: str):

        if short_flow_model_name in short_flow_models:
            self.short_flow_model = self.short_flow_model = short_flow_models[short_flow_model_name]()
        else:
            # Default case or raise an error if you don't want a default FlowProvider
            raise ValueError(f"Unsupported short flow model: {self.config.short_flow_model}")

    def compute_flow(self, source_image: torch.Tensor, target_image: torch.Tensor, backview=False):
        source_image = source_image.float() * 255
        target_image = target_image.float() * 255

        observed_flow = self.short_flow_model.next_flow(source_image, target_image)

        occlusion = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)
        uncertainty = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)

        return observed_flow, occlusion, uncertainty


class LongFlowStrategy(ExternalMethodFlowStrategy):
    def __init__(self, long_flow_provider: MFTFlowProvider):
        self.long_flow_provider = long_flow_provider

    def compute_flow(self, source_image: torch.Tensor, target_image: torch.Tensor, backview=False):
        source_image = source_image.float() * 255
        target_image = target_image.float() * 255

        if self.long_flow_provider.need_to_init:
            self.long_flow_provider.init(source_image)
            self.long_flow_provider.need_to_init = False

        observed_flow, occlusion, uncertainty = self.long_flow_provider.next_flow(source_image, target_image)

        return observed_flow, occlusion, uncertainty

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
    def __init__(self, short_flow_model: FlowProvider):
        self.short_flow_model = short_flow_model

    def compute_flow(self, flow_source_frame, flow_target_frame, backview=False):
        image_new_x255 = flow_target_frame.observed_image.float() * 255
        image_prev_x255 = flow_source_frame.observed_image.float() * 255

        observed_flow = self.short_flow_model.next_flow(image_prev_x255, image_new_x255)
        occlusion = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)
        uncertainty = torch.zeros(1, 1, 1, *observed_flow.shape[-2:]).to(observed_flow.device)

        return observed_flow, occlusion, uncertainty


class LongFlowStrategy(ExternalMethodFlowStrategy):
    def __init__(self, long_flow_provider: MFTFlowProvider, long_flow_provider_backview: MFTFlowProvider):
        self.long_flow_provider = long_flow_provider
        self.long_flow_provider_backview = long_flow_provider_backview

    def compute_flow(self, flow_source_frame, flow_target_frame, backview=False):
        image_new_x255 = flow_target_frame.observed_image.float() * 255
        image_prev_x255 = flow_source_frame.observed_image.float() * 255

        if backview:
            template = image_prev_x255
            target = image_new_x255
            flow_provider = self.long_flow_provider_backview
        else:
            template = image_prev_x255
            target = image_new_x255
            flow_provider = self.long_flow_provider

        if flow_provider.need_to_init:
            flow_provider.init(template)
            flow_provider.need_to_init = False

        observed_flow, occlusion, uncertainty = flow_provider.next_flow(template, target)

        return observed_flow, occlusion, uncertainty

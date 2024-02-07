from dataclasses import field, dataclass
from typing import Optional, Tuple

import torch

from models.encoder import Encoder, EncoderResult
from models.loss import FMOLoss
from models.rendering import RenderingKaolin
from utils import qnorm_vectorized, normalize_rendered_flows


class LossFunctionWrapper(torch.nn.Module):
    @dataclass
    class CTX:
        x_world: Optional[torch.Tensor] = field(default=None)
        x_camera: Optional[torch.Tensor] = field(default=None)
        x_image: Optional[torch.Tensor] = field(default=None)
        x_prime_image: Optional[torch.Tensor] = field(default=None)
        gt_flow: Optional[torch.Tensor] = field(default=None)
        flow_sgd_indices: Optional[Tuple] = field(default=None)

    def __init__(self, encoder_result, encoder_result_flow_frames, encoder, rendering, flow_arcs_indices, loss_function,
                 observed_flows, observed_flows_segmentations, rendered_width,
                 rendered_height, original_width, original_height, custom_points_for_ransac=None):
        super().__init__()
        self.encoder_result = encoder_result
        self.encoder_result_flow_frames: EncoderResult = encoder_result_flow_frames
        self.rendering: RenderingKaolin = rendering
        self.encoder: Encoder = encoder
        self.flow_arcs_indices = flow_arcs_indices
        self.loss_function: FMOLoss = loss_function
        self.observed_flows = observed_flows
        self.observed_flows_segmentations = observed_flows_segmentations
        self.rendered_width = rendered_width
        self.rendered_height = rendered_height
        self.original_width = original_width
        self.original_height = original_height
        self.custom_points_for_ransac = custom_points_for_ransac
        self.ctx: Optional[LossFunctionWrapper.CTX] = None

        self.rendering_result_flow_frames = self.rendering.forward(self.encoder_result_flow_frames.translations,
                                                                   self.encoder_result_flow_frames.quaternions,
                                                                   self.encoder_result_flow_frames.vertices,
                                                                   self.encoder.face_features,
                                                                   self.encoder_result_flow_frames.texture_maps,
                                                                   self.encoder_result_flow_frames.lights)

    def forward(self, trans_quats):
        trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 7, 7))

        translations = trans_quats[None, ..., :3]
        quaternions = trans_quats[..., 3:]
        quaternions = qnorm_vectorized(quaternions)

        encoder_result = self.encoder_result._replace(translations=translations, quaternions=quaternions)

        self.ctx = None
        if self.rendering.config.use_custom_jacobian:
            self.ctx = self.CTX()
            self.ctx.gt_flow = self.observed_flows

        theoretical_flow_result = \
            self.rendering.compute_theoretical_flow_using_rendered_vertices(self.rendering_result_flow_frames,
                                                                            encoder_result,
                                                                            self.encoder_result_flow_frames,
                                                                            self.flow_arcs_indices,
                                                                            ctx=self.ctx)

        # TODO replace me with true occlusions
        mock_occlusion = torch.zeros(1, 1, 1, *self.observed_flows.shape[-2:]).cuda()

        # theoretical_flow = normalize_rendered_flows(theoretical_flow_result.theoretical_flow,
        #                                             self.rendered_width, self.rendered_height,
        #                                             self.original_width, self.original_height)
        theoretical_flow = theoretical_flow_result.theoretical_flow
        flow_segmentation = theoretical_flow_result.rendered_flow_segmentation

        loss_result, nonzero_points = self.loss_function.get_optical_flow_epes(self.observed_flows, mock_occlusion,
                                                                               self.observed_flows_segmentations,
                                                                               theoretical_flow, flow_segmentation)

        self.ctx.flow_sgd_indices = nonzero_points

        return loss_result.to(torch.float)

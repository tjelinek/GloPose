import torch

from models.encoder import Encoder
from models.loss import FMOLoss
from models.rendering import RenderingKaolin, infer_normalized_renderings


class LossFunctionWrapper(torch.nn.Module):

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

        self.rendering_result_flow_frames = self.rendering.forward(self.encoder_result.translations,
                                                                   self.encoder_result.quaternions,
                                                                   self.encoder_result.vertices,
                                                                   self.encoder.face_features,
                                                                   self.encoder_result.texture_maps,
                                                                   self.encoder_result.lights)

    def forward(self, trans_quats):
        trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 7, 7))

        translations = trans_quats[None, ..., :3]
        quaternions = trans_quats[..., 3:]
        quaternions = qnorm_vectorized(quaternions)

        encoder_result = self.encoder_result._replace(translations=translations, quaternions=quaternions)

        theoretical_flow_result = \
            self.rendering.compute_theoretical_flow_using_rendered_vertices(self.rendering_result_flow_frames,
                                                                            encoder_result,
                                                                            self.encoder_result_flow_frames,
                                                                            self.flow_arcs_indices)

        # TODO replace me with true occlusions
        mock_occlusion = torch.zeros(1, 1, 1, *self.observed_flows.shape[-2:]).cuda()

        loss_result = self.loss_function.get_optical_flow_epes(observed_flow=self.observed_flows,
                                                               observed_flow_occlusion=mock_occlusion,
                                                               observed_flow_segmentation=self.observed_flows_segmentations,
                                                               rendered_flow=theoretical_flow_result.theoretical_flow,
                                                               rendered_flow_segmentation=theoretical_flow_result.rendered_flow_segmentation)

        return loss_result.to(torch.float)

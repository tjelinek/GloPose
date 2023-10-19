import torch

from models.encoder import Encoder
from models.rendering import RenderingKaolin
from utils import infer_normalized_renderings


class LossFunctionWrapper(torch.nn.Module):

    def __init__(self, encoder_result, encoder_result_flow_frames, encoder, rendering, flow_arcs_indices, loss_function,
                 observed_images, observed_segmentations, observed_flows, observed_flows_segmentations, rendered_width,
                 rendered_height, original_width, original_height):
        super().__init__()
        self.encoder_result = encoder_result
        self.encoder_result_flow_frames = encoder_result_flow_frames
        self.rendering: RenderingKaolin = rendering
        self.encoder: Encoder = encoder
        self.flow_arcs_indices = flow_arcs_indices
        self.loss_function = loss_function
        self.observed_images = observed_images
        self.observed_segmentations = observed_segmentations
        self.observed_flows = observed_flows
        self.observed_flows_segmentations = observed_flows_segmentations
        self.rendered_width = rendered_width
        self.rendered_height = rendered_height
        self.original_width = original_width
        self.original_height = original_height
        self.tracking6d = tracking6d

    def forward(self, trans_quats):
        trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 6, 6))

        translations = trans_quats[None, ..., :3]
        quaternions = trans_quats[..., 3:]
        quaternions_weights = 1 - torch.linalg.vector_norm(quaternions, dim=-1).unsqueeze(-1)
        quaternions = torch.cat([quaternions_weights, quaternions], dim=-1)

        encoder_result = self.encoder_result._replace(translations=translations, quaternions=quaternions)

        inference_result = infer_normalized_renderings(self.rendering, self.encoder.face_features,
                                                       encoder_result, self.encoder_result_flow_frames,
                                                       self.flow_arcs_indices, self.original_width,
                                                       self.original_height)

        renders, theoretical_flow, rendered_flow_segmentation, occlusion_masks = inference_result

        rendered_silhouettes = renders[0, :, :, -1:]
        loss_result = self.loss_function.forward(rendered_images=renders, observed_images=self.observed_images,
                                                 rendered_silhouettes=rendered_silhouettes,
                                                 observed_silhouettes=self.observed_segmentations,
                                                 rendered_flow=theoretical_flow,
                                                 observed_flow=self.observed_flows,
                                                 observed_flow_segmentation=self.observed_flows_segmentations,
                                                 rendered_flow_segmentation=rendered_flow_segmentation,
                                                 observed_flow_occlusion=None, observed_flow_uncertainties=None,
                                                 keyframes_encoder_result=encoder_result,
                                                 last_keyframes_encoder_result=None,
                                                 return_end_point_errors=True)

        return loss_result.to(torch.float)

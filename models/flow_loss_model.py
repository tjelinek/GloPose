import torch

from utils import normalize_rendered_flows


class LossFunctionWrapper(torch.nn.Module):

    def __init__(self, encoder_result, encoder_result_flow_frames, encoder, rendering, loss_function, observed_images,
                 observed_segmentations, observed_flows, observed_flows_segmentations, rendered_width, rendered_height,
                 original_width, original_height):
        super().__init__()
        self.encoder_result = encoder_result
        self.encoder_result_flow_frames = encoder_result_flow_frames
        self.rendering = rendering
        self.encoder = encoder
        self.loss_function = loss_function
        self.observed_images = observed_images
        self.observed_segmentations = observed_segmentations
        self.observed_flows = observed_flows
        self.observed_flows_segmentations = observed_flows_segmentations
        self.rendered_width = rendered_width
        self.rendered_height = rendered_height
        self.original_width = original_width
        self.original_height = original_height

    def forward(self, trans_quats):
        trans_quats = trans_quats.unflatten(-1, (1, trans_quats.shape[-1] // 6, 6))

        translations = trans_quats[None, ..., :3]
        quaternions = trans_quats[..., 3:]
        quaternions_weights = 1 - torch.linalg.vector_norm(quaternions, dim=-1).unsqueeze(-1)
        quaternions = torch.cat([quaternions_weights, quaternions], dim=-1)

        encoder_result = self.encoder_result._replace(translations=translations, quaternions=quaternions)

        renders = self.rendering(translations, quaternions, encoder_result.vertices,
                                 self.encoder.face_features, encoder_result.texture_maps, None)

        flow_result = self.rendering.compute_theoretical_flow(encoder_result, self.encoder_result_flow_frames)
        theoretical_flow, rendered_flow_segmentation = flow_result
        rendered_flow_segmentation = rendered_flow_segmentation[None]

        # Renormalization compensating for the fact that we render into bounding box that is smaller than the
        # actual image
        theoretical_flow = normalize_rendered_flows(theoretical_flow, self.rendered_width, self.rendered_height,
                                                    self.original_width, self.original_height)

        rendered_silhouettes = renders[0, :, :, -1:]
        loss_result = self.loss_function.forward(rendered_images=renders, observed_images=self.observed_images,
                                                 rendered_silhouettes=rendered_silhouettes,
                                                 observed_silhouettes=self.observed_segmentations,
                                                 rendered_flow=theoretical_flow,
                                                 observed_flow=self.observed_flows,
                                                 observed_flow_segmentation=self.observed_flows_segmentations,
                                                 rendered_flow_segmentation=rendered_flow_segmentation,
                                                 keyframes_encoder_result=encoder_result,
                                                 last_keyframes_encoder_result=None,
                                                 return_end_point_errors=True)

        del renders
        del theoretical_flow
        del rendered_silhouettes

        return loss_result.to(torch.float)

        # def loss_function_wrapper(translations_quaternions_, encoder_result_, encoder_result_flow_frames_):
        #     # quaternions_, translations_ = se3_exp(translations_quaternions_)
        #     # translations_ = translations_[None]
        #
        #     translations_ = translations_quaternions_[None, ..., :3]
        #     quaternions_ = translations_quaternions_[..., 3:]
        #     quaternions_weights_ = 1 - torch.linalg.vector_norm(quaternions_, dim=-1).unsqueeze(-1)
        #     quaternions_ = torch.cat([quaternions_weights_, quaternions_], dim=-1)
        #
        #     encoder_result_ = encoder_result_._replace(translations=translations_, quaternions=quaternions_)
        #
        #     renders_ = self.rendering(translations_, quaternions_, encoder_result_.vertices,
        #                               self.encoder.face_features, encoder_result_.texture_maps, None)
        #
        #     flow_result_ = self.rendering.compute_theoretical_flow(encoder_result_, encoder_result_flow_frames_)
        #     theoretical_flow_, rendered_flow_segmentation_ = flow_result_
        #     rendered_flow_segmentation_ = rendered_flow_segmentation_[None]
        #
        #     # Renormalization compensating for the fact that we render into bounding box that is smaller than the
        #     # actual image
        #     theoretical_flow_ = normalize_rendered_flows(theoretical_flow_, self.rendering.width,
        #                                                  self.rendering.height, self.shape[-1], self.shape[-2])
        #
        #     rendered_silhouettes_ = renders_[0, :, :, -1:]
        #     loss_result_ = self.loss_function.forward(rendered_images=renders_, observed_images=observed_images,
        #                                               rendered_silhouettes=rendered_silhouettes_,
        #                                               observed_silhouettes=observed_segmentations,
        #                                               rendered_flow=theoretical_flow_,
        #                                               observed_flow=observed_flows,
        #                                               observed_flow_segmentation=observed_flows_segmentations,
        #                                               rendered_flow_segmentation=rendered_flow_segmentation_,
        #                                               keyframes_encoder_result=encoder_result_,
        #                                               last_keyframes_encoder_result=self.last_encoder_result,
        #                                               return_end_point_errors=True)
        #
        #     del renders_
        #     del theoretical_flow_
        #     del rendered_silhouettes_
        #
        #     return loss_result_.to(torch.float)

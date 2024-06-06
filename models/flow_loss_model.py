from dataclasses import field, dataclass
from typing import Optional, Tuple

import torch
from kornia.geometry import normalize_quaternion

from models.encoder import Encoder, EncoderResult
from models.loss import FMOLoss
from models.rendering import RenderingKaolin


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
        quaternions = normalize_quaternion(quaternions)

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

        if self.rendering.config.use_custom_jacobian:
            self.ctx.flow_sgd_indices = nonzero_points

        return loss_result.to(torch.float)

    def dx_dq(self, q, x0):
        # Given quaternion q=(q_w, q_i, q_j, q_k) and an initial point x=(x_0, x_1, x_2), computes the gradient
        # f(x0) = R(q)*x0, where R(q) is a transformation of the quaternion q to rotation matrix.
        # q: torch.Tensor of shape (B, 4)
        # x0: torch.Tensor of shape (B, 3)
        # where B is the batch size

        q_w = q[:, 0].unsqueeze(-1).unsqueeze(-1)
        q_i = q[:, 1].unsqueeze(-1).unsqueeze(-1)
        q_j = q[:, 2].unsqueeze(-1).unsqueeze(-1)
        q_k = q[:, 3].unsqueeze(-1).unsqueeze(-1)

        dx_dqw = torch.cat([
            torch.cat([torch.zeros_like(q_k), -2 * q_k, 2 * q_j], dim=2),
            torch.cat([2 * q_k, torch.zeros_like(q_k), -2 * q_i], dim=2),
            torch.cat([-2 * q_j, 2 * q_i, torch.zeros_like(q_i)], dim=2)
        ], dim=1) @ x0.unsqueeze(-1)

        dx_dqi = torch.cat([
            torch.cat([torch.zeros_like(q_j), 2 * q_j, 2 * q_k], dim=2),
            torch.cat([2 * q_j, -4 * q_i, -2 * q_w], dim=2),
            torch.cat([2 * q_k, 2 * q_w, -4 * q_i], dim=2)
        ], dim=1) @ x0.unsqueeze(-1)

        dx_dqj = torch.cat([
            torch.cat([-4 * q_j, 2 * q_i, 2 * q_w], dim=2),
            torch.cat([2 * q_i, -4 * q_j, 2 * q_k], dim=2),
            torch.cat([-2 * q_w, 2 * q_k, -4 * q_j], dim=2)
        ], dim=1) @ x0.unsqueeze(-1)

        dx_dqk = torch.cat([
            torch.cat([-4 * q_k, -2 * q_w, 2 * q_i], dim=2),
            torch.cat([2 * q_w, -4 * q_k, 2 * q_j], dim=2),
            torch.cat([2 * q_i, 2 * q_j, -4 * q_k], dim=2)
        ], dim=1) @ x0.unsqueeze(-1)

        # Reshape the output to match expected shape (100, 3)
        dx_dqw = dx_dqw.squeeze(-1)
        dx_dqi = dx_dqi.squeeze(-1)
        dx_dqj = dx_dqj.squeeze(-1)
        dx_dqk = dx_dqk.squeeze(-1)

        return dx_dqw, dx_dqi, dx_dqj, dx_dqk

    def dW_dx(self, x):
        # Computes gradient of perspective camera viewing transformation W(x) = (x_0 / x_2, x_1 / x_2)
        # x: torch.Tensor of shape (B, 3), where B us the batch size
        # Returns three Tensors of shape (B, 2)

        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]

        batch_zeros = torch.zeros(x0.shape).cuda()

        dW_dx0 = torch.stack((1 / x2, batch_zeros), dim=1)
        dW_dx1 = torch.stack((batch_zeros, 1 / x2), dim=1)
        dW_dx2 = torch.stack((x0 * torch.log(x2), x1 * torch.log(x2)), dim=1)

        return dW_dx0, dW_dx1, dW_dx2

    def dW_dq(self, q, x_world, x_camera):
        # Computes gradient of viewing transformation W with respect to individual quaternion components
        # q: torch.Tensor of shape (B, 4)
        # x_world: torch.Tensor of shape (B, 3) - the world mesh vertices coordinates
        # x_camera: torch.Tensor of shape (B, 3) - the camera mesh vertices coordinates

        dW_dx0, dW_dx1, dW_dx2 = self.dW_dx(x_camera)
        dW_dx = torch.stack((dW_dx0, dW_dx1, dW_dx2), dim=-1)
        # dW_dx<i> shape (B, 2, 3)

        dx_dqw, dx_dqi, dx_dqj, dx_dqk = self.dx_dq(q, x_world)
        dx_dq = torch.stack((dx_dqw, dx_dqi, dx_dqj, dx_dqk), dim=-1)
        # dx_dq shape (B, 3, 4)

        dW_dq = dW_dx @ dx_dq
        # dW_dq shape: (B, 2, 4)

        return dW_dq

    def dW_dt(self, x_camera):
        # Computes gradient of viewing transformation W with respect to individual quaternion components
        # t: torch.Tensor of shape (B, 4)
        # x_camera: torch.Tensor of shape (B, 3) - the camera mesh vertices coordinates

        x0 = x_camera[:, 0]
        x1 = x_camera[:, 1]
        x2 = x_camera[:, 2]

        batch_zeros = torch.zeros(x0.shape).cuda()

        dW_dt0 = torch.stack((1 / x2, batch_zeros), dim=1)
        dW_dt1 = torch.stack((batch_zeros, 1 / x2), dim=1)
        dW_dt2 = torch.stack((x0 * torch.log(x2), x1 * torch.log(x2)), dim=1)
        # dW_dti shape: (B, 2)

        dW_dt = torch.stack((dW_dt0, dW_dt1, dW_dt2), dim=-1)
        # dW_dti shape: (B, 2, 3)

        return dW_dt

    def dL_dq_dt(self, q, t, x_world, x_camera, x_image, x_prime_image, gt_flow):
        # x_world: torch.Tensor of shape (B, 3)
        # x_camera: torch.Tensor of shape (B, 3)
        # x_image: torch.Tensor of shape (B, 2)
        # x_prime_image: torch.Tensor of shape (B, 2)
        # gt_flow: torch.Tensor of shape (B, 2)

        flow_predicted = x_prime_image - x_image

        x_camera = x_camera * self.rendering.camera_proj.view(-1, 3)

        dW_dq = self.dW_dq(q, x_world, x_camera)
        # dW_dq shape: (B, 2, 4)

        dL = 2 * (flow_predicted - gt_flow).unsqueeze(1)
        # dL shape: (B, 2)
        dL_dq = (dL @ dW_dq).squeeze()
        # dL_dq shape: (B, 4)

        dW_dt = self.dW_dt(x_camera)
        # dW_dt shape: (B, 2, 3)

        dL_dt = (dL @ dW_dt).squeeze()
        # dL_dq shape: (B, 3)

        dL_dq_dt = torch.cat([dL_dt, dL_dq], dim=-1)
        # dL_dq_dt shape: (B, 7)

        return dL_dq_dt

    def compute_jacobian(self, trans_quats):
        trans_quats = trans_quats.unflatten(-1, (trans_quats.shape[-1] // 7, 7))
        translations = trans_quats[..., :3]
        quaternions = trans_quats[..., 3:]

        F = len(self.flow_arcs_indices)  # Total frames
        N = self.loss_function.config.flow_sgd_n_samples  # Samples per frame
        T = N * F  # Total samples

        J = torch.zeros(T, 7 * (F + 1)).cuda()

        for f1_i, f2_i in self.flow_arcs_indices:
            t = translations[[f2_i]]
            q = quaternions[[f1_i]]
            t_batch = t.repeat(N, 1)
            q_batch = q.repeat(N, 1)

            flow_sgd_indices = self.ctx.flow_sgd_indices[:, f1_i].nonzero().t()

            h_idxs = flow_sgd_indices[2]
            w_idxs = flow_sgd_indices[3]

            x_world = self.ctx.x_world[0, f1_i, :, h_idxs, w_idxs].transpose(0, 1)[f1_i * N: (f1_i + 1) * N]
            x_camera = self.ctx.x_camera[0, f1_i, :, h_idxs, w_idxs].transpose(0, 1)[f1_i * N: (f1_i + 1) * N]
            x_image = self.ctx.x_image[0, f1_i, :, h_idxs, w_idxs].transpose(0, 1)[f1_i * N: (f1_i + 1) * N]
            x_prime_image = self.ctx.x_prime_image[0, f1_i, :, h_idxs, w_idxs].transpose(0, 1)[f1_i * N: (f1_i + 1) * N]
            gt_flow = self.ctx.gt_flow[0, f1_i, :, h_idxs, w_idxs].transpose(0, 1)[f1_i * N: (f1_i + 1) * N]

            dL_dq_dt = self.dL_dq_dt(q_batch, t_batch, x_world, x_camera, x_image, x_prime_image, gt_flow)

            block_idx = f2_i
            J[(block_idx - 1) * N: (block_idx + 0) * N, block_idx * 7: (block_idx + 1) * 7] = dL_dq_dt

        return J

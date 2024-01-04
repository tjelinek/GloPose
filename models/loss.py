import numpy as np
import torch
import torch.nn as nn
from kornia.losses import total_variation

from models.encoder import EncoderResult
from utils import erode_segment_mask2


class FMOLoss(nn.Module):
    def __init__(self, config, ivertices, faces):
        super(FMOLoss, self).__init__()
        self.config = config
        if self.config.loss_laplacian_weight > 0:
            self.lapl_loss = LaplacianLoss(ivertices, faces)

    def forward(self,
                rendered_images, observed_images,
                rendered_silhouettes, observed_silhouettes,
                rendered_flow, observed_flow,
                observed_flow_segmentation, rendered_flow_segmentation,
                observed_flow_occlusion, rendered_flow_occlusion,
                observed_flow_uncertainties,
                keyframes_encoder_result, last_keyframes_encoder_result,
                return_end_point_errors=False,
                custom_points_for_ransac=None):
        """
        Forward pass of the model.

        Args:
            rendered_images (torch.Tensor): Rendered images.
            observed_images (torch.Tensor): Input batch of images.
            rendered_silhouettes (torch.Tensor): Rendered silhouettes (segmentation masks from rendered images).
            observed_silhouettes (torch.Tensor): Observed silhouettes (segmentation masks from input images).
            rendered_flow (torch.Tensor): Flow obtained from the tracking results.
            observed_flow (torch.Tensor): Observed flow between frames.
            observed_flow_segmentation (torch.Tensor): Flow masks for observed flow.
            rendered_flow_segmentation (torch.Tensor): Flow masks for rendered flow.
            observed_flow_occlusion (torch.Tensor): Occlusion mask in range [0, 1], where 1 indicates occlusion.
            rendered_flow_occlusion (torch.Tensor): Occlusion mask in range [0, 1], where 1 indicates occlusion.
            observed_flow_uncertainties (torch.Tensor): Flow values uncertainties in range [0, 1], where 1 indicated
                                                        being not certain about the true value.
            keyframes_encoder_result (EncoderResult): Encoder result for the current frame.
            last_keyframes_encoder_result (EncoderResult): Encoder result for the previous frame.
            return_end_point_errors (bool): Indicates whether to return concatenated end point errors for two
                                            consecutive flow frames rather than the mean of the end point errors.
            custom_points_for_ransac (list): List of custom points for opt. flow loss computation

        Returns:
            tuple: A tuple containing losses_all, losses, and total loss.

        """
        vertices = keyframes_encoder_result.vertices
        texture_maps = keyframes_encoder_result.texture_maps
        tdiff = keyframes_encoder_result.translation_difference
        qdiff = keyframes_encoder_result.quaternion_difference

        losses = {}
        losses_all = {}
        if self.config.loss_rgb_weight > 0:
            modelled_renders = (rendered_images * rendered_silhouettes)
            segments_our = (rendered_silhouettes > 0).to(rendered_images.dtype)

            track_segm_loss, t_all = fmo_model_loss(observed_images, modelled_renders, segments_our, self.config)
            losses["model"] = self.config.loss_rgb_weight * track_segm_loss
            losses_all["track_segm_loss"] = t_all
        if True or self.config.loss_iou_weight > 0:
            losses["silh"] = 0
            losses_all["silh"] = []
            denom = rendered_images.shape[1]
            for frmi in range(rendered_images.shape[1]):
                temp_loss = self.config.loss_iou_weight * iou_loss(rendered_silhouettes[:, [frmi]],
                                                                   observed_silhouettes[:, [frmi], [-1]])
                losses_all["silh"].append(temp_loss.tolist()[0])
                losses["silh"] = losses["silh"] + temp_loss / denom
        if self.config.optimize_shape and self.config.loss_laplacian_weight > 0:
            losses["lap"] = self.config.loss_laplacian_weight * self.lapl_loss(vertices)

        if self.config.loss_q_weight > 0:
            if qdiff[qdiff > 0].shape[0] > 0:
                losses["qdiff"] = qdiff[qdiff > 0][-1].mean()
            else:
                losses["qdiff"] = qdiff[-1].mean()
            losses["qdiff"] = self.config.loss_q_weight * qdiff[-1]

        if self.config.loss_t_weight > 0:
            if tdiff[tdiff > 0].shape[0] > 0:
                losses["tdiff"] = tdiff[tdiff > 0][-1].mean()
            else:
                losses["tdiff"] = tdiff[-1].mean()
            losses["tdiff"] = self.config.loss_t_weight * tdiff[-1]

        if self.config.loss_dist_weight > 0:
            dists = (observed_silhouettes[:, :, 0] * rendered_images[:, :, 0, -1])
            losses["dist"] = self.config.loss_dist_weight * (
                    dists.sum((0, 2, 3)) / rendered_images[:, :, 0, -1].sum((0, 2, 3))).mean()

        if self.config.loss_tv_weight > 0:
            texture_maps_rep = torch.cat((texture_maps[:, :, -1:], texture_maps, texture_maps[:, :, :1]), 2)
            texture_maps_rep = torch.cat(
                (texture_maps_rep[:, :, :, -1:], texture_maps_rep, texture_maps_rep[:, :, :, :1]), 3)
            texture_maps_rep = torch.cat((texture_maps_rep[:, :, -1:], texture_maps_rep, texture_maps_rep[:, :, :1]), 2)
            losses["tv"] = self.config.loss_tv_weight * total_variation(texture_maps_rep, reduction='sum') / (
                    3 * self.config.texture_size ** 2)
            losses["tv"] = losses["tv"].sum(dim=1)

        per_pixel_flow_loss = None
        if self.config.loss_flow_weight > 0:
            observed_flow_segmentation = observed_flow_segmentation[0, :, -1:].permute(1, 0, 2, 3)  # Shape (1, N, H, W)
            rendered_flow_segmentation = rendered_flow_segmentation[0].permute(1, 0, 2, 3)  # Shape (1, N, H, W)

            # observed_not_rendered_flow_segmentation = (observed_flow_segmentation - rendered_flow_segmentation > 0)\
            #     .to(observed_flow_segmentation.dtype)
            # observed_and_rendered_flow_segmentation = (observed_flow_segmentation * rendered_flow_segmentation > 0)\
            #     .to(observed_flow_segmentation.dtype)
            # not_observed_rendered_flow_segmentation = (rendered_flow_segmentation - observed_flow_segmentation > 0)\
            #     .to(observed_flow_segmentation.dtype)
            # observed_or_rendered_flow_segmentation = (rendered_flow_segmentation + observed_flow_segmentation > 0)\
            #     .to(observed_flow_segmentation.dtype)
            observed_flow_segmentation = (observed_flow_segmentation > 0).to(observed_flow_segmentation.dtype)
            rendered_flow_segmentation = (rendered_flow_segmentation > 0).to(rendered_flow_segmentation.dtype)

            # Perform erosion of the segmentation mask
            if self.config.segmentation_mask_erosion_iters:
                erosion_iterations = self.config.segmentation_mask_erosion_iters
                observed_flow_segmentation = erode_segment_mask2(erosion_iterations, observed_flow_segmentation)

                flow_from_tracking_tmp = rendered_flow[0].permute(0, 3, 1, 2)
                flow_from_tracking_tmp = erode_segment_mask2(flow_from_tracking_tmp, rendered_flow_segmentation)
                rendered_flow = flow_from_tracking_tmp.permute(0, 2, 3, 1)

            flow_from_tracking_clone = rendered_flow.clone()  # Size (1, N, 2, H, W)
            flow_from_tracking_clone = flow_from_tracking_clone.permute(0, 1, 3, 4, 2)  # Size (1, N, H, W, 2)

            observed_flow_clone = observed_flow.clone()  # Size (1, N, 2, H, W)
            observed_flow_clone = observed_flow_clone.permute(0, 1, 3, 4, 2)  # Size (1, N, 2, H, W)

            if self.config.flow_sgd:
                if custom_points_for_ransac is not None:
                    rendered_flow_segmentation[...] = False
                    rendered_flow_segmentation[custom_points_for_ransac] = True
                    image_area = custom_points_for_ransac[0].shape[0]
                else:
                    rendered_flow_segmentation = random_points_from_binary_mask(rendered_flow_segmentation[0],
                                                                                self.config.flow_sgd_n_samples)[None]

                    image_area = self.config.flow_sgd_n_samples
            else:
                image_area = rendered_images.shape[-2:].numel()

            # end_point_error_sqrt = torch.norm(end_point_error, dim=-1, p=0.5)
            # per_pixel_flow_loss = torch.where(end_point_error_magnitude < 1, end_point_error_magnitude,
            #                                   end_point_error_sqrt)

            # Compute the mean of the loss divided by the total object area to take into account different objects size
            end_point_error = observed_flow_clone - flow_from_tracking_clone
            end_point_error_l1_norm = torch.norm(end_point_error, dim=-1, p=2)

            # per_pixel_flow_loss_observed_not_rendered = (end_point_error_l1_norm *
            #                                              observed_not_rendered_flow_segmentation)
            # per_pixel_flow_loss_not_observed_rendered = (end_point_error_l1_norm *
            #                                              not_observed_rendered_flow_segmentation)
            # per_pixel_flow_loss_observed_and_rendered = (end_point_error_l1_norm *
            #                                              observed_and_rendered_flow_segmentation)
            # per_pixel_flow_loss_observed_or_rendered = (end_point_error_l1_norm *
            #                                             observed_or_rendered_flow_segmentation)
            # per_pixel_flow_loss_observed = (end_point_error_l1_norm * observed_flow_segmentation_binary)
            per_pixel_flow_loss_rendered = (end_point_error_l1_norm * rendered_flow_segmentation)

            per_pixel_flow_loss = per_pixel_flow_loss_rendered
            if return_end_point_errors:
                nonzero_indices = tuple(rendered_flow_segmentation.nonzero().t())
                per_pixel_flow_loss_nonzero = per_pixel_flow_loss[nonzero_indices]
                return per_pixel_flow_loss_nonzero.flatten()

            # per_pixel_mean_flow_loss_observed_not_rendered = (
            #         per_pixel_flow_loss_observed_not_rendered.sum(dim=(2, 3)) / image_area).mean(dim=(1,))
            # per_pixel_mean_flow_loss_not_observed_rendered = (
            #         per_pixel_flow_loss_not_observed_rendered.sum(dim=(2, 3)) / image_area).mean(dim=(1,))
            # per_pixel_mean_flow_loss_observed_and_rendered = (
            #         per_pixel_flow_loss_observed_and_rendered.sum(dim=(2, 3)) / image_area).mean(dim=(1,))
            # per_pixel_mean_flow_loss_observed_or_rendered = (
            #         per_pixel_flow_loss_observed_or_rendered.sum(dim=(2, 3)) / image_area).mean(dim=(1,))
            # per_pixel_mean_flow_loss_observed = (
            #         per_pixel_flow_loss_observed.sum(dim=(2, 3)) / image_area).mean(dim=(1,))
            per_pixel_mean_flow_loss_rendered = (
                    per_pixel_flow_loss_rendered.sum(dim=(2, 3)) / image_area).mean(dim=(1,))

            # losses["fl_obs_not_ren"] = (per_pixel_mean_flow_loss_observed_not_rendered *
            #                              self.config.loss_flow_weight)
            # losses["fl_not_obs_ren"] = (per_pixel_mean_flow_loss_not_observed_rendered *
            #                              self.config.loss_fl_not_obs_rend_weight)
            # losses["fl_obs_and_ren"] = (per_pixel_mean_flow_loss_observed_and_rendered *
            #                              self.config.loss_fl_obs_and_rend_weight)
            # losses["fl_obs_or_ren"] = (per_pixel_mean_flow_loss_observed_or_rendered *
            #                             self.config.loss_flow_weight)
            # losses["fl_obs"] = (per_pixel_mean_flow_loss_observed *
            #                             self.config.loss_flow_weight)
            # losses["fl_ren"] = (per_pixel_mean_flow_loss_rendered *
            #                             self.config.loss_flow_weight)

            # per_image_mean_flow = per_pixel_flow_loss.sum(dim=(2, 3)) / image_area
            # flow_loss = per_image_mean_flow.mean(dim=(1,))
            flow_loss = per_pixel_mean_flow_loss_rendered
            losses["flow_loss"] = flow_loss * self.config.loss_flow_weight

        if self.config.loss_texture_change_weight > 0:
            change_in_texture = (last_keyframes_encoder_result.texture_maps - texture_maps) ** 2
            change_in_texture = change_in_texture.squeeze().permute(1, 2, 0)
            change_in_texture = change_in_texture.sum(dim=-1).flatten()
            least_changed_90_pct, _ = (-change_in_texture).topk(k=int(change_in_texture.size().numel() * 0.9))
            least_changed_90_pct = -least_changed_90_pct
            loss_texture_change = least_changed_90_pct.mean()
            losses["texture_change"] = loss_texture_change * self.config.loss_texture_change_weight

        loss = 0
        for ls in losses:
            if ls not in ['fl_obs_not_rend', 'fl_not_obs_rend', 'fl_obs_and_rend', 'fl_obs_or_rend']:
                loss += losses[ls]
        return losses_all, losses, loss, per_pixel_flow_loss


def random_points_from_binary_mask(binary_mask, N):
    batch_size, height, width = binary_mask.shape
    binary_mask_new = binary_mask.clone() * False

    # Get the indices of True values in the binary mask for each slice along the first dimension
    indices_list = [torch.nonzero(binary_mask[i], as_tuple=False) for i in range(batch_size)]

    for i, indices in enumerate(indices_list):
        # Get the number of true points in the current slice
        num_true_points = indices.shape[0]

        selected_indices_i = indices[torch.randperm(num_true_points)[:N]]

        # Create a mask for the selected indices
        mask_i = torch.zeros_like(binary_mask[i])
        mask_i[selected_indices_i[:, 0], selected_indices_i[:, 1]] = True

        # Assign mask_i to the corresponding slice in binary_mask_new
        binary_mask_new[i] = mask_i

    return binary_mask_new


def iou_loss(YM, YpM):
    """

    :param YM: Segmentation of shape (1, N, 1, H, W)
    :param YpM: Segmentation of shape (1, N, 1, H, W)
    :return:
    """
    A_inter_B = YM * YpM
    A_union_B = (YM + YpM - A_inter_B)
    iou = 1 - (torch.sum(A_inter_B, [2, 3, 4]) / torch.sum(A_union_B, [2, 3, 4])).mean(1)
    return iou


def cauchy_loss(YpM, YM, YMb, scale=0.25):
    losses = nn.L1Loss(reduction='none')(YpM * YMb, YM * YMb)  # **2
    cauchy_losses = (scale ** 2) * torch.log(1 + losses / (scale ** 2))
    bloss = cauchy_losses.sum([1, 2, 3]) / (YMb.sum([1, 2, 3]) * YpM.shape[1] + 0.01)
    return bloss


def batch_loss(YpM, YM, YMb, do_mult=True, weights=None):
    if do_mult:
        losses = nn.L1Loss(reduction='none')(YpM * YMb, YM * YMb)
    else:
        losses = nn.L1Loss(reduction='none')(YpM, YM)
    if weights is not None:
        losses = weights * losses
        YMb = weights
    if len(losses.shape) > 4:
        bloss = losses.sum([1, 2, 3, 4]) / YMb.sum([1, 2, 3, 4])
    else:
        bloss = losses.sum([1, 2, 3]) / (YMb.sum([1, 2, 3]) + 0.01)
    return bloss


def fmo_model_loss(input_batch, renders, segments, config):
    segments = segments.type(renders.dtype)

    model_loss = 0
    model_loss_all = []
    for frmi in range(input_batch.shape[1]):
        if config.features == 'deep':
            temp_loss = cauchy_loss(renders[:, frmi], input_batch[:, frmi], segments[:, frmi])
        else:
            temp_loss = batch_loss(renders[:, frmi], input_batch[:, frmi], segments[:, frmi])
        model_loss_all.append(temp_loss.tolist()[0])
        model_loss = model_loss + temp_loss / input_batch.shape[1]
    return model_loss, model_loss_all


# Taken from
# https://github.com/ShichenLiu/SoftRas/blob/master/soft_renderer/losses.py

class LaplacianLoss(nn.Module):
    def __init__(self, vertex, faces, average=False):
        super(LaplacianLoss, self).__init__()
        self.nv = vertex.shape[0]
        self.nf = faces.shape[0]
        self.average = average
        laplacian = np.zeros([self.nv, self.nv]).astype(np.float32)

        laplacian[faces[:, 0], faces[:, 1]] = -1
        laplacian[faces[:, 1], faces[:, 0]] = -1
        laplacian[faces[:, 1], faces[:, 2]] = -1
        laplacian[faces[:, 2], faces[:, 1]] = -1
        laplacian[faces[:, 2], faces[:, 0]] = -1
        laplacian[faces[:, 0], faces[:, 2]] = -1

        r, c = np.diag_indices(laplacian.shape[0])
        laplacian[r, c] = -laplacian.sum(1)

        for i in range(self.nv):
            laplacian[i, :] /= laplacian[i, i]

        self.register_buffer('laplacian', torch.from_numpy(laplacian))

    def forward(self, x):
        batch_size = x.size(0)
        x = torch.matmul(self.laplacian, x)
        dims = tuple(range(x.ndimension())[1:])
        x = x.pow(2).mean(dims)
        if self.average:
            return x.sum() / batch_size
        else:
            return x

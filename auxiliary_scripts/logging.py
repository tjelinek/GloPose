from collections import namedtuple
from itertools import product

from typing import Dict, Iterable, Tuple, List

import os
import math

import torch
import cv2
import imageio
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from torch import nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from kornia.geometry.conversions import quaternion_to_axis_angle, axis_angle_to_quaternion
from pytorch3d.loss.chamfer import chamfer_distance

from keyframe_buffer import FrameObservation, FlowObservation, KeyframeBuffer
from models.loss import iou_loss, FMOLoss
from tracker_config import TrackerConfig
from utils import write_video, qnorm, quaternion_angular_difference, deg_to_rad, rad_to_deg
from models.rendering import infer_normalized_renderings, RenderingKaolin
from helpers.torch_helpers import write_renders
from models.kaolin_wrapper import write_obj_mesh
from models.encoder import EncoderResult, Encoder
from flow import visualize_flow_with_images, compare_flows_with_images, flow_unit_coords_to_image_coords, \
    source_coords_to_target_coords_np


class WriteResults:
    Metrics = namedtuple('Metrics', ['loss', 'rotation', 'translation',
                                     'gt_rotation', 'gt_translation'])

    def __init__(self, write_folder, shape, num_frames, tracking_config: TrackerConfig):
        self.all_input = cv2.VideoWriter(os.path.join(write_folder, 'all_input.avi'), cv2.VideoWriter_fourcc(*"MJPG"),
                                         10, (shape[1], shape[0]), True)
        self.all_segm = cv2.VideoWriter(os.path.join(write_folder, 'all_segm.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                        (shape[1], shape[0]), True)
        self.all_proj = cv2.VideoWriter(os.path.join(write_folder, 'all_proj.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                        (shape[1], shape[0]), True)
        self.all_proj_filtered = cv2.VideoWriter(os.path.join(write_folder, 'all_proj_filtered.avi'),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                                 (shape[1], shape[0]), True)
        self.tracking_config: TrackerConfig = tracking_config
        self.output_size: torch.Size = shape
        self.baseline_iou = -np.ones((num_frames - 1, 1))
        self.our_iou = -np.ones((num_frames - 1, 1))
        self.tracking_log = open(Path(write_folder) / "tracking_log.txt", "w")
        self.metrics_log = open(Path(write_folder) / "tracking_metrics_log.txt", "w")

        self.write_folder = Path(write_folder)

        self.flows_path = self.write_folder / Path('flows')
        self.gt_imgs_path = self.write_folder / Path('gt_imgs')
        self.occlusion_maps_path = self.write_folder / Path('mft_occlusions')

        self.flows_path.mkdir(exist_ok=True, parents=True)
        self.gt_imgs_path.mkdir(exist_ok=True, parents=True)
        self.occlusion_maps_path.mkdir(exist_ok=True, parents=True)

        self.metrics_writer = csv.writer(self.metrics_log)

        self.metrics_writer.writerow(["Frame", "mIoU", "lastIoU", "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
                                      "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"])

        self.tensorboard_log_dir = Path(write_folder) / Path("logs")
        self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_log = None

        self.logged_metrics: Dict[int, WriteResults.Metrics] = {}

    def __del__(self):
        self.all_input.release()
        self.all_segm.release()
        self.all_proj.release()
        self.all_proj_filtered.release()

        self.tracking_log.close()
        self.metrics_log.close()

    def visualize_loss_landscape(self, observations: FrameObservation, flow_observations: FlowObservation, tracking6d,
                                 stepi, relative_mode=False):
        """
        Visualizes the loss landscape by computing the joint losses across varying translations and rotations.

        Parameters:
        - tracking6d (Tracking6DClassType): The 6D tracking data, containing ground truth rotations, translations,
                                            and other relevant properties.
        - observations (FrameObservation): Observations.
        - flow_observations (FlowObservation): Flow observations.
        - stepi (int): Current step or iteration.

        - relative_mode (bool, optional): If True and stepi >= 1, it will compute the ground truth marker with respect
                                          the previous predicted value.

        Behavior:
        - The function computes joint losses over a grid defined by translations and rotations around different axes.
        - It visualizes these joint losses as a 2D heatmap, overlaying paths of SGD iterations and ground truth values.
        - The resultant visualization is saved as an EPS file in a 'loss_landscapes' directory under
          `tracking6d.write_folder`.

        Notes:
        - Only certain combinations of translation and rotation axes are considered.
        - The range of translations and rotations are derived from the ground truth values and are limited to specific
          intervals for visualization.
        - Ground truth, start and end points of the SGD path, and contour lines for the loss values are overlaid on the
        heatmap.

        Returns:
        None
        """

        num_translations = 25
        num_rotations = 25

        trans_axes = ['x', 'y', 'z']
        rot_axes = ['x', 'y', 'z']

        for trans_axis_idx, rot_axis_idx in product(range(len(trans_axes)), range(len(rot_axes))):

            if trans_axis_idx in [1, 2] or rot_axis_idx in [0, 2]:
                continue

            if relative_mode and stepi > 1:
                gt_rotation_deg_prev = rad_to_deg(tracking6d.gt_rotations[0, stepi - 1]).cpu()
                gt_translation_prev = tracking6d.gt_translations[0, 0, stepi - 1].cpu()

                gt_rotation_deg_current = rad_to_deg(tracking6d.gt_rotations[0, stepi]).cpu()
                gt_translation_current = tracking6d.gt_translations[0, 0, stepi].cpu()

                gt_translation_diff = gt_translation_current - gt_translation_prev
                gt_rotation_diff = gt_rotation_deg_current - gt_rotation_deg_prev

                gt_translation = tracking6d.logged_sgd_translations[0][0, 0, 0].detach().cpu() + \
                                 gt_translation_diff

                gt_rotation_quaternion = tracking6d.logged_sgd_quaternions[0].detach().cpu()
                gt_rotation_rad = quaternion_to_axis_angle(gt_rotation_quaternion)
                last_rotation_deg = rad_to_deg(gt_rotation_rad)[0, 0]

                gt_rotation_deg = last_rotation_deg + gt_rotation_diff
            else:
                gt_rotation_deg = rad_to_deg(tracking6d.gt_rotations[0, stepi]).cpu()
                gt_translation = tracking6d.gt_translations[0, 0, stepi].cpu()

            translations_space = np.linspace(gt_translation[trans_axis_idx] - 0.3,
                                             gt_translation[trans_axis_idx] + 0.3, num=num_translations)
            rotations_space = np.linspace(gt_rotation_deg[rot_axis_idx] - 7,
                                          gt_rotation_deg[rot_axis_idx] + 7, num=num_rotations)

            print(f"Visualizing loss landscape for translation axis {trans_axes[trans_axis_idx]} "
                  f"and rotation axis {rot_axes[rot_axis_idx]}")

            joint_losses: np.ndarray = self.compute_loss_landscape(flow_observations, observations, tracking6d,
                                                                   rotations_space, translations_space, trans_axis_idx,
                                                                   rot_axis_idx, stepi)

            min_val = np.min(joint_losses)
            max_val = np.max(joint_losses)

            plt.figure(figsize=(10, 8))
            plt.imshow(joint_losses.T,
                       extent=[translations_space[0], translations_space[-1], rotations_space[-1], rotations_space[0]],
                       aspect='auto', cmap='hot', interpolation='none', vmin=min_val, vmax=max_val)
            cbar = plt.colorbar(label='Joint Loss')
            ticks = list(cbar.get_ticks())
            ticks.append(min_val)
            ticks.append(max_val)
            ticks = sorted(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

            plt.xlabel(f'Translation ({trans_axes[trans_axis_idx]}-axis)')
            plt.ylabel(f'Rotation (degrees) ({rot_axes[rot_axis_idx]}-axis)')

            # Set y-ticks for all integral numbers based on the range of rotations_space
            plt.yticks(np.arange(int(rotations_space[0]), int(rotations_space[-1]) + 1, 1))

            # Set x-ticks at every 0.1 step based on the range of translations_space
            plt.xticks(np.arange(translations_space[0], translations_space[-1] + 0.1, 0.1))

            # Add grid
            plt.grid(True, which='both', linestyle='--', linewidth=0.25)

            prev_iteration_x_translation = None
            prev_iteration_y_rotation_deg = None

            for i in range(len(tracking6d.logged_sgd_translations)):
                iteration_translation = tracking6d.logged_sgd_translations[i][0, 0, -1, trans_axis_idx].detach().cpu()
                iteration_rotation_quaternion = tracking6d.logged_sgd_quaternions[i].detach().cpu()
                iteration_rotation_rad = quaternion_to_axis_angle(iteration_rotation_quaternion)
                iteration_rotation_deg = rad_to_deg(iteration_rotation_rad)[0, -1, rot_axis_idx]

                if i == 0:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='white', marker='x', label='Start')
                    plt.text(iteration_translation, iteration_rotation_deg, 'Start', verticalalignment='bottom',
                             color='white')
                elif i == len(tracking6d.logged_sgd_translations) - 1:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='yellow', marker='x', label='End')
                    plt.text(iteration_translation, iteration_rotation_deg, 'End', verticalalignment='bottom',
                             color='yellow')
                    plt.plot([prev_iteration_x_translation, iteration_translation],
                             [prev_iteration_y_rotation_deg, iteration_rotation_deg], color='yellow', linewidth=0.1,
                             linestyle='dotted')
                else:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='orange', marker='.', label=str(i))
                    plt.plot([prev_iteration_x_translation, iteration_translation],
                             [prev_iteration_y_rotation_deg, iteration_rotation_deg], color='yellow', linewidth=0.1,
                             linestyle='dotted')

                prev_iteration_x_translation = iteration_translation
                prev_iteration_y_rotation_deg = iteration_rotation_deg

            plt.scatter(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                        color='green', marker='x', label='GT')
            if relative_mode:
                plt.text(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                         'Relative optimum', verticalalignment='top', color='green')
            else:
                plt.text(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                         'GT', verticalalignment='top', color='green')

            # 2) Show contours of the values
            contours = plt.contour(translations_space, rotations_space, joint_losses.T, levels=20)
            plt.clabel(contours, inline=True, fontsize=10)

            # 3) Visualize the gradient
            # gradient = np.gradient(joint_losses.T)
            # plt.quiver(translations_space, rotations_space, -gradient[1], gradient[0], color='white', width=0.003)

            plt.title('Joint Losses')
            loss_landscape_folder = Path(tracking6d.write_folder / 'loss_landscapes')
            loss_landscape_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(loss_landscape_folder /
                        f'joint_loss_landscape_{stepi}_trans-{trans_axes[trans_axis_idx]}'
                        f'_rot-{rot_axes[rot_axis_idx]}.eps', format='eps')

    @staticmethod
    def compute_loss_landscape(flow_observations: FlowObservation, observations: FrameObservation, tracking6d,
                               rotation_space, translation_space, trans_axis_idx, rot_axis_idx, stepi):

        joint_losses = np.zeros((translation_space.shape[0], rotation_space.shape[0]))

        for i, translation in enumerate(translation_space):
            for j, rotation_deg in enumerate(rotation_space):
                translation_tensor = torch.Tensor([0, 0, 0])
                translation_tensor[trans_axis_idx] = translation
                sampled_translation = translation_tensor[None, None, None].cuda()

                rotation_tensor_deg = torch.Tensor([0, 0, 0])
                rotation_tensor_deg[rot_axis_idx] = rotation_deg
                rotation_tensor_rad = deg_to_rad(rotation_tensor_deg)
                rotation_tensor_quaternion = axis_angle_to_quaternion(rotation_tensor_rad)

                sampled_rotation = rotation_tensor_quaternion[None, None].cuda()

                encoder_result, encoder_result_flow_frames = \
                    tracking6d.frames_and_flow_frames_inference([stepi], [stepi - 1], encoder_type='deep_features')

                encoder_result = encoder_result._replace(translations=sampled_translation, quaternions=sampled_rotation)
                flow_arcs_indices = [(-1, -1)]

                inference_result = infer_normalized_renderings(tracking6d.rendering, tracking6d.encoder.face_features,
                                                               encoder_result, encoder_result_flow_frames,
                                                               flow_arcs_indices,
                                                               tracking6d.shape[-1], tracking6d.shape[-2])
                renders, rendered_silhouettes, rendered_flow_result = inference_result

                loss_function: FMOLoss = tracking6d.loss_function
                loss_result = loss_function.forward(rendered_images=renders,
                                                    observed_images=observations.observed_image_features,
                                                    rendered_silhouettes=rendered_silhouettes,
                                                    observed_silhouettes=observations.observed_segmentation,
                                                    rendered_flow=rendered_flow_result.theoretical_flow,
                                                    observed_flow=flow_observations.observed_flow,
                                                    observed_flow_segmentation=flow_observations.observed_flow_segmentation,
                                                    rendered_flow_segmentation=rendered_flow_result.rendered_flow_segmentation,
                                                    observed_flow_occlusion=flow_observations.observed_flow_occlusion,
                                                    rendered_flow_occlusion=rendered_flow_result.rendered_flow_occlusion,
                                                    observed_flow_uncertainties=flow_observations.observed_flow_uncertainty,
                                                    keyframes_encoder_result=encoder_result,
                                                    last_keyframes_encoder_result=encoder_result)

                losses_all, losses, joint_loss, per_pixel_error = loss_result

                joint_losses[i, j] = joint_loss

        return joint_losses

    def set_tensorboard_log_for_frame(self, frame_i):
        self.tensorboard_log = SummaryWriter(str(self.tensorboard_log_dir / f'Frame_{frame_i}'))

    def write_into_tensorboard_log(self, sgd_iter: int, values_dict: Dict):

        for field_name, value in values_dict.items():
            self.tensorboard_log.add_scalar(field_name, value, sgd_iter)

    def write_tensor_into_bbox(self, image, bounding_box):
        image_with_margins = torch.zeros(image.shape[:-2] + self.output_size).to(image.device)
        image_with_margins[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]] = \
            image[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
        return image_with_margins

    def write_results(self, bounding_box, our_losses, frame_i, encoder_result, tex, new_flow_arcs, frame_result,
                      active_keyframes: KeyframeBuffer, active_keyframes_backview: KeyframeBuffer,
                      logged_sgd_translations, logged_sgd_quaternions, deep_encoder: Encoder, rgb_encoder: Encoder,
                      renderer: RenderingKaolin, renderer_backview, best_model, observations: FrameObservation,
                      observations_backview: FrameObservation, gt_encoder: Encoder, gt_rotations, gt_translations):

        observed_images = observations.observed_image
        observed_image_features = observations.observed_image_features
        observed_segmentations = observations.observed_segmentation

        self.visualize_theoretical_flow(bounding_box=bounding_box, keyframe_buffer=active_keyframes,
                                        keyframe_buffer_backview=active_keyframes_backview, new_flow_arcs=new_flow_arcs,
                                        rgb_encoder=rgb_encoder, deep_encoder=deep_encoder, renderer=renderer,
                                        renderer_backview=renderer_backview)

        self.visualize_flow(active_keyframes, active_keyframes_backview, new_flow_arcs,
                            frame_result.per_pixel_flow_error)

        # FLOW BACKVIEW ERROR VISUALIZATION
        for new_flow_arc in new_flow_arcs:

            flow_arc_source, flow_arc_target = new_flow_arc

            values_frontview = self.get_values_for_matching(active_keyframes, flow_arc_source, flow_arc_target)
            (occlusion_mask_front, seg_mask_front, target_front,
             template_front, template_front_overlay, step, flow_frontview) = values_frontview

            display_bounds = (0, target_front.shape[0], 0, target_front.shape[1])

            fig, axs = plt.subplots(3, 2, figsize=(8, 8))

            for ax in axs.flat:
                ax.axis('off')

            darkening_factor = 0.5
            axs[0, 0].imshow(template_front * darkening_factor, extent=display_bounds)
            axs[0, 0].set_title('Template Front')

            axs[1, 0].imshow(template_front_overlay * darkening_factor, extent=display_bounds)
            axs[1, 0].set_title('Template Front Occlusion')

            axs[2, 0].imshow(target_front * darkening_factor, extent=display_bounds)
            axs[2, 0].set_title('Target Front')

            height, width = template_front.shape[:2]  # Assuming these are the dimensions of your images
            x, y = np.meshgrid(np.arange(width, step=step), np.arange(height, step=step))
            template_coords = np.stack((y, x), axis=0).reshape(2, -1)  # Shape: [2, height*width]

            # Plot lines on the target front and back view subplots
            occlusion_threshold = self.tracking_config.occlusion_coef_threshold

            flow_frontview_np = flow_frontview.numpy(force=True)

            if frame_result.inliers is not None and False:
                inliers = frame_result.inliers[new_flow_arc].numpy(force=True).T  # Ensure shape is (2, N)
                self.draw_cross_axes_flow_matches(inliers, flow_frontview_np, axs[1, 0], axs[2, 0], 'Greens')

            if frame_result.outliers is not None:
                outliers = frame_result.outliers[new_flow_arc].numpy(force=True).T  # Ensure shape is (2, N)
                self.draw_cross_axes_flow_matches(outliers, flow_frontview_np, axs[1, 0], axs[2, 0], 'Reds')

            self.plot_matched_lines(axs[1, 0], axs[2, 0], template_coords, occlusion_mask_front, occlusion_threshold,
                                    flow_frontview_np, cmap='spring', marker='o', segment_mask=seg_mask_front)

            if self.tracking_config.matching_target_to_backview:
                values_backview = self.get_values_for_matching(active_keyframes_backview, flow_arc_source,
                                                               flow_arc_target)
                (occlusion_mask_back, seg_mask_back, target_back, template_back,
                 template_back_overlay, step, flow_backview) = values_backview

                assert template_back_overlay.shape == template_front_overlay.shape
                assert target_back.shape == target_front.shape

                axs[0, 1].imshow(template_back * darkening_factor, extent=display_bounds)
                axs[0, 1].set_title('Template Back')

                axs[1, 1].imshow(template_back_overlay * darkening_factor, extent=display_bounds)
                axs[1, 1].set_title('Template Back Occlusion')

                axs[2, 1].imshow(target_front * darkening_factor, extent=display_bounds)
                axs[2, 1].set_title('Target Back')

                flow_backview_np = flow_backview.numpy(force=True)
                self.plot_matched_lines(axs[1, 1], axs[2, 1], template_coords, occlusion_mask_back, occlusion_threshold,
                                        flow_backview_np, cmap='cool', marker='x', segment_mask=seg_mask_back)

            destination_path = self.flows_path / f'matching_gt_flow_{flow_arc_source}_{flow_arc_target}.png'
            fig.savefig(str(destination_path), dpi=600, bbox_inches='tight')

        # FLOW BACKVIEW ERROR VISUALIZATION

        detached_result = EncoderResult(*[it.clone().detach() if type(it) is torch.Tensor else it
                                          for it in encoder_result])

        with torch.no_grad():

            gt_rotation_current_frame = gt_rotations[:, frame_i].squeeze()
            gt_translation_current_frame = gt_translations[:, :, frame_i].squeeze()

            last_logged_sgd_rotation = rad_to_deg(quaternion_to_axis_angle(logged_sgd_quaternions[-1].detach().cpu(),
                                                                           ))[0, -1]

            self.logged_metrics[frame_i] = self.Metrics(loss=frame_result.frame_losses[-1],
                                                        translation=logged_sgd_translations[-1][0, 0, -1],
                                                        rotation=last_logged_sgd_rotation,
                                                        gt_rotation=rad_to_deg(gt_rotation_current_frame),
                                                        gt_translation=gt_translation_current_frame)

            self.visualize_logged_metrics()

            self.visualize_rotations_per_epoch(logged_sgd_translations, logged_sgd_quaternions,
                                               frame_result.frame_losses, frame_i,
                                               gt_rotation=gt_rotation_current_frame,
                                               gt_translation=gt_translation_current_frame)

            print(f"Keyframes: {active_keyframes.keyframes}, "
                  f"flow arcs: {sorted(active_keyframes.G.edges, key=lambda x: x[::-1])}")

            self.tracking_log.write(f"Step {frame_i}:\n")
            self.tracking_log.write(f"Keyframes: {active_keyframes.keyframes}\n")

            self.write_keyframe_rotations(detached_result, active_keyframes.keyframes)
            self.write_all_encoder_rotations(deep_encoder, max(active_keyframes.keyframes) + 1)

            if self.tracking_config.features == 'rgb':
                tex = detached_result.texture_maps

            feat_renders_result = renderer.forward(detached_result.translations, detached_result.quaternions,
                                                   detached_result.vertices, deep_encoder.face_features,
                                                   detached_result.texture_maps, detached_result.lights)

            feat_renders = feat_renders_result.rendered_image

            rgb_renders_result = renderer.forward(detached_result.translations, detached_result.quaternions,
                                                  detached_result.vertices, deep_encoder.face_features,
                                                  tex, detached_result.lights)

            renders = rgb_renders_result.rendered_image
            rendered_silhouette = rgb_renders_result.rendered_image_segmentation

            renders_crop = renders[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
            feat_renders_crop = feat_renders[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]

            self.render_silhouette_overlap(rendered_silhouette[:, [-1]],
                                           observed_segmentations[:, [-1]], frame_i)

            write_renders(feat_renders[:, :, :3], self.write_folder, self.tracking_config.max_keyframes + 1, ids=0)
            write_renders(renders_crop, self.write_folder, self.tracking_config.max_keyframes + 1, ids=1)

            # write_renders(torch.cat((images[..., bounding_box[0]:bounding_box[1],
            #                          bounding_box[2]:bounding_box[3]], feat_renders[:, :, :, -1:]), 3),
            #                 self.write_folder, self.tracking_config.max_keyframes + 1, ids=2)

            write_obj_mesh(detached_result.vertices[0].cpu().numpy(), best_model["faces"],
                           deep_encoder.face_features[0].cpu().numpy(),
                           os.path.join(self.write_folder, f'mesh_{frame_i}.obj'), "model_" + str(frame_i) + ".mtl")
            save_image(detached_result.texture_maps[:, :3], os.path.join(self.write_folder, 'tex_deep.png'))
            save_image(tex, os.path.join(self.write_folder, f'tex_{frame_i}.png'))

            with open(self.write_folder / "model.mtl", "r") as file:
                lines = file.readlines()

            # Replace the last line
            lines[-1] = f"map_Kd tex_{frame_i}.png\n"

            # Write the result to a new file
            with open(self.write_folder / f"model_{frame_i}.mtl", "w") as file:
                file.writelines(lines)

            renders_np = renders.numpy(force=True)
            observed_images_numpy = observed_images.numpy(force=True)
            observed_segmentations_numpy = observed_segmentations.numpy(force=True)
            segmented_images_numpy = observed_images_numpy * observed_segmentations_numpy

            write_video(renders_np, os.path.join(self.write_folder, 'im_recon.avi'), fps=6)
            write_video(observed_images_numpy, os.path.join(self.write_folder, 'input.avi'), fps=6)

            write_video(segmented_images_numpy, os.path.join(self.write_folder, 'segments.avi'), fps=6)
            for tmpi in range(renders.shape[1]):
                img = observed_images[0, tmpi, :3, bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
                seg = observed_segmentations[0][tmpi, :, bounding_box[0]:bounding_box[1],
                      bounding_box[2]:bounding_box[3]].clone()
                save_image(seg, os.path.join(self.write_folder, 'imgs', 's{}.png'.format(tmpi)))
                seg[seg == 0] = 0.35
                save_image(img, os.path.join(self.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
                save_image(observed_image_features[0, tmpi, :3, bounding_box[0]:bounding_box[1],
                           bounding_box[2]:bounding_box[3]],
                           os.path.join(self.write_folder, 'imgs', 'if{}.png'.format(tmpi)))
                save_image(torch.cat((img, seg), 0),
                           os.path.join(self.write_folder, 'imgs', 'is{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, [3, 3, 3]],
                           os.path.join(self.write_folder, 'imgs', 'm{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, :],
                           os.path.join(self.write_folder, 'imgs', 'r{}.png'.format(tmpi)))
                save_image(feat_renders_crop[0, tmpi, 0, :],
                           os.path.join(self.write_folder, 'imgs', 'f{}.png'.format(tmpi)))

                segmentations_discrete = (observed_segmentations[:, -1:, [-1]] > 0).to(observed_segmentations.dtype)
                self.baseline_iou[frame_i - 1] = 1 - iou_loss(segmentations_discrete,
                                                              observed_segmentations[:, -1:, [-1]]).detach().cpu()
                self.our_iou[frame_i - 1] = 1 - iou_loss(rendered_silhouette[:, [-1]],
                                                         observed_segmentations[:, -1:, [-1]]).detach().cpu()

            print('Baseline IoU {}, our IoU {}'.format(self.baseline_iou[frame_i - 1], self.our_iou[frame_i - 1]))
            np.savetxt(os.path.join(self.write_folder, 'baseline_iou.txt'), self.baseline_iou, fmt='%.10f',
                       delimiter='\n')
            np.savetxt(os.path.join(self.write_folder, 'iou.txt'), self.our_iou, fmt='%.10f', delimiter='\n')
            np.savetxt(os.path.join(self.write_folder, 'losses.txt'), our_losses, fmt='%.10f', delimiter='\n')

            image_to_write = observed_images[0, :, :3].clamp(min=0, max=1).cpu().numpy()
            image_to_write = image_to_write.transpose(2, 3, 1, 0)
            image_to_write = image_to_write[:, :, [2, 1, 0], -1]
            image_to_write = (image_to_write * 255).astype(np.uint8)
            self.all_input.write(image_to_write)

            segmentation_to_write = (observed_images[0, :, :3] * observed_segmentations[0])
            segmentation_to_write = segmentation_to_write.clamp(min=0, max=1).cpu().numpy()
            segmentation_to_write = segmentation_to_write.transpose(2, 3, 1, 0)
            segmentation_to_write = segmentation_to_write[:, :, [2, 1, 0], -1]
            segmentation_to_write = (segmentation_to_write * 255).astype(np.uint8)
            self.all_segm.write(segmentation_to_write)

            rendered_silhouette = renders[0, :, :3].detach().clamp(min=0, max=1).cpu().numpy()
            rendered_silhouette = rendered_silhouette.transpose(2, 3, 1, 0)
            rendered_silhouette = rendered_silhouette[:, :, [2, 1, 0], -1]
            rendered_silhouette = (rendered_silhouette * 255).astype(np.uint8)
            self.all_proj.write(rendered_silhouette)

    @staticmethod
    def draw_cross_axes_flow_matches(source_coords, flow_frontview_np, axs1, axs2, cmap):
        max_points = 100
        total_points = source_coords.shape[1]

        if total_points > max_points:
            random_sample = np.random.permutation(total_points)[:max_points]
            source_coords = source_coords[:, random_sample]

        y2_f, x2_f = source_coords_to_target_coords_np(source_coords, flow_frontview_np)
        target_coords = np.vstack((x2_f, y2_f))
        norm = Normalize(vmin=0, vmax=source_coords.shape[1] - 1)
        cmap = plt.get_cmap(cmap)
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        for i in range(0, source_coords.shape[1]):
            color = mappable.to_rgba(i)
            xyA = source_coords[:, i]  # Source point
            xB, yB = target_coords[:, i]

            # Create a ConnectionPatch for each pair of sampled points
            con = ConnectionPatch(xyA=(xyA[1], xyA[0]), xyB=(xB, yB),
                                  coordsA='data', coordsB='data',
                                  axesA=axs1, axesB=axs2, color=color, lw=0.5)
            axs2.add_artist(con)

    def get_values_for_matching(self, active_keyframes, flow_arc_source, flow_arc_target):
        flow_observation_frontview = active_keyframes.get_flows_between_frames(flow_arc_source, flow_arc_target)
        flow_frontview = flow_observation_frontview.observed_flow
        occlusion_mask_front = self.convert_observation_to_numpy(flow_observation_frontview.observed_flow_occlusion)
        seg_mask_front = self.convert_observation_to_numpy(flow_observation_frontview.observed_flow_segmentation)
        flow_frontview = flow_unit_coords_to_image_coords(flow_frontview.clone())
        template_observation_frontview = active_keyframes.get_observations_for_keyframe(flow_arc_source)
        target_observation_frontview = active_keyframes.get_observations_for_keyframe(flow_arc_target)
        template_front = self.convert_observation_to_numpy(template_observation_frontview.observed_image)
        target_front = self.convert_observation_to_numpy(target_observation_frontview.observed_image)

        N = 20
        step = template_front.shape[0] // N

        template_front_overlay = self.overlay_occlusion(template_front, occlusion_mask_front >=
                                                        self.tracking_config.occlusion_coef_threshold)

        return (occlusion_mask_front, seg_mask_front, target_front,
                template_front, template_front_overlay, step, flow_frontview)

    def add_loss_plot(self, ax_, frame_losses_, indices=None):
        if indices is None:
            indices = range(len(frame_losses_))
        ax_loss = ax_.twinx()
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(indices, frame_losses_, color='red', label='Loss')
        ax_loss.spines.right.set_position(("axes", 1.15))
        ax_loss.legend(loc='upper left')

    def visualize_logged_metrics(self):
        fig, axs = plt.subplots(2, 1, figsize=(12, 15))
        fig.subplots_adjust(hspace=0.4)
        frame_indices = sorted(self.logged_metrics.keys())
        losses = [self.logged_metrics[frame].loss for frame in frame_indices]
        rotations = np.array([self.logged_metrics[frame].rotation.numpy(force=True) for frame in frame_indices])
        translations = np.array([self.logged_metrics[frame].translation.numpy(force=True) for frame in frame_indices])
        gt_rotations = np.array([self.logged_metrics[frame].gt_rotation.numpy(force=True) for frame in frame_indices])
        gt_translations = np.array([self.logged_metrics[frame].gt_translation.numpy(force=True)
                                    for frame in frame_indices])

        # Plot Rotation
        colors = ['yellow', 'green', 'blue']
        axs[0].set_xticks(list(frame_indices))
        axs[1].set_xticks(list(frame_indices))

        for i, axis_label in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
            axs[0].plot(frame_indices, rotations[:, i], label=f'{axis_label} Rotation', color=colors[i])
            axs[0].plot(frame_indices, gt_rotations[:, i], '--', label=f'GT {axis_label} Rotation', alpha=0.5,
                        color=colors[i])
        axs[0].set_title('Rotation per Frame')
        axs[0].set_xlabel('Frame Index')
        axs[0].set_ylabel('Rotation')
        axs[0].legend(loc='lower right')
        # Plot Translation
        for i, axis_label in enumerate(['X-axis', 'Y-axis', 'Z-axis']):
            axs[1].plot(frame_indices, translations[:, i], label=f'{axis_label} Translation', color=colors[i])
            axs[1].plot(frame_indices, gt_translations[:, i], '--', label=f'GT {axis_label} Translation', alpha=0.5,
                        color=colors[i])
        axs[1].set_title('Translation per Frame')
        axs[1].set_xlabel('Frame Index')
        axs[1].set_ylabel('Translation')
        axs[1].legend(loc='lower right')

        self.add_loss_plot(axs[0], losses, indices=frame_indices)
        self.add_loss_plot(axs[1], losses, indices=frame_indices)

        (Path(self.write_folder) / Path('rotations_by_epoch')).mkdir(exist_ok=True, parents=True)
        fig_path = Path(self.write_folder) / Path('rotations_by_epoch') / f'pose_per_frame.svg'
        plt.savefig(fig_path)
        plt.close()

    @staticmethod
    def convert_observation_to_numpy(observation):
        return observation[0, 0].permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def overlay_occlusion(image, occlusion_mask):
        """
        Overlay an occlusion mask on an image.

        Args:
        - image: The original image as a numpy array of shape (H, W, C).
        - occlusion_mask: The occlusion mask as a numpy array of shape (H, W, 1), values in [0, 1].

        Returns:
        - The image with the occlusion mask overlay as a numpy array of shape (H, W, C).
        """
        occlusion_mask = occlusion_mask.squeeze() * 0.2  # Remove the singleton dimension if present
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # Grayscale or single-channel
            image = np.dstack([image] * 3)  # Convert to 3-channel for coloring

        white_overlay = np.ones_like(image) * 255
        overlay_image = (1 - occlusion_mask[..., np.newaxis]) * image + occlusion_mask[..., np.newaxis] * white_overlay

        return overlay_image.astype(image.dtype)

    @staticmethod
    def plot_matched_lines(ax1, ax2, source_coords, occlusion_mask, occl_threshold, flow, cmap='jet', marker='o',
                           segment_mask=None, segment_threshold=0.99):
        """
        Draws lines from source coordinates in ax1 to target coordinates in ax2.
        Args:
        - ax1: The matplotlib axis to draw source points on.
        - ax2: The matplotlib axis to draw target points on.
        - source_coords: Source coordinates as a 2xN numpy array (N is the number of points).
        - target_coords: Target coordinates as a 2xN numpy array.
        - cmap: Colormap used for plotting.
        - marker: Marker style for the points.
        """
        # Assuming flow is [2, H, W] and source_coords is [2, N]
        y1, x1 = source_coords
        y2_f, x2_f = source_coords_to_target_coords_np(source_coords, flow)

        # Apply masks
        valid_mask = occlusion_mask[-y1.astype(int), x1.astype(int), 0] <= occl_threshold
        if segment_mask is not None:
            valid_mask &= segment_mask[-y1.astype(int), x1.astype(int), 0] > segment_threshold

        # Filter coordinates and colors based on valid_mask
        x1, y1, x2_f, y2_f = x1[valid_mask], y1[valid_mask], x2_f[valid_mask], y2_f[valid_mask]

        # Normalize and map colors
        norm = Normalize(vmin=0, vmax=np.sum(valid_mask) - 1)
        cmap = plt.get_cmap(cmap)
        colors = cmap(norm(range(np.sum(valid_mask))))

        # Create LineCollection for valid lines
        lines = np.stack((x1, y1, x2_f, y2_f), axis=1).reshape(-1, 2, 2)
        lc = LineCollection(lines, colors=colors, linewidths=0.5, alpha=0.8)
        ax2.add_collection(lc)

        # Scatter plot for source and target points
        ax1.scatter(x1, y1, color=colors, marker=marker, alpha=0.8, s=1.5)
        ax2.scatter(x2_f, y2_f, color=colors, marker=marker, alpha=0.8, s=1.5)

    def evaluate_metrics(self, stepi, tracking6d, keyframes, predicted_vertices, predicted_quaternion,
                         predicted_translation, predicted_mask, gt_vertices=None, gt_rotation=None, gt_translation=None,
                         gt_object_mask=None):

        with torch.no_grad():
            encoder_result_all_frames, _ = self.encoder_result_all_frames(tracking6d.encoder, max(keyframes) + 1)

            chamfer_dist = "NA"
            iou_3d = "NA"
            miou_2d = "NA"
            last_iou_2d = "NA"
            mTransAll = "NA"
            mTransKF = "NA"
            transLast = "NA"
            mAngDiffAll = "NA"
            mAngDiffKF = "NA"
            angDiffLast = "NA"

            if gt_vertices is not None:
                chamfer_dist = float(chamfer_distance(predicted_vertices, gt_vertices)[0])

            if gt_rotation is not None:
                gt_quaternion = axis_angle_to_quaternion(gt_rotation)

                pred_quaternion_all_frames = encoder_result_all_frames.quaternions
                gt_quaternion_all_frames = gt_quaternion[:, :stepi + 1]

                pred_quaternion_keyframes = predicted_quaternion
                gt_quaternion_keyframes = gt_quaternion[:, keyframes]

                pred_quaternion_last = predicted_quaternion[None, :, -1]
                gt_quaternion_last = gt_quaternion[None, :, stepi]

                ang_diff_all_frames = quaternion_angular_difference(pred_quaternion_all_frames,
                                                                    gt_quaternion_all_frames)
                mAngDiffAll = float(ang_diff_all_frames.mean())

                ang_diff_keyframes = quaternion_angular_difference(pred_quaternion_keyframes,
                                                                   gt_quaternion_keyframes)
                mAngDiffKF = float(ang_diff_keyframes.mean())

                ang_diff_last_frame = quaternion_angular_difference(pred_quaternion_last,
                                                                    gt_quaternion_last)
                angDiffLast = float(ang_diff_last_frame.mean())

            if gt_translation is not None:
                pred_translation_all_frames = encoder_result_all_frames.translations
                gt_translation_all_frames = gt_translation[:, :, :stepi + 1]

                pred_translation_keyframes = predicted_translation
                gt_translation_keyframes = gt_translation[:, :, keyframes]

                pred_translation_last = predicted_translation[None, :, :, -1]
                gt_translation_last = gt_translation[None, :, :, stepi]

                # Compute L2 norm for all frames
                translation_l2_diff_all_frames = torch.norm(pred_translation_all_frames - gt_translation_all_frames,
                                                            dim=-1)
                mTransAll = float(translation_l2_diff_all_frames.mean())

                # Compute L2 norm for keyframes
                translation_l2_diff_keyframes = torch.norm(pred_translation_keyframes - gt_translation_keyframes,
                                                           dim=-1)
                mTransKF = float(translation_l2_diff_keyframes.mean())

                # Compute L2 norm for the last frame
                translation_l2_diff_last = torch.norm(pred_translation_last - gt_translation_last, dim=-1)
                transLast = float(translation_l2_diff_last.mean())

            if gt_object_mask is not None:

                ious = torch.zeros(gt_object_mask.shape[1])
                for frame_i in range(gt_object_mask.shape[1]):
                    frame_iou = 1 - iou_loss(predicted_mask[None, None, :, frame_i],
                                             gt_object_mask[None, None, :, frame_i])
                    ious[frame_i] = frame_iou

                last_iou_2d = float(frame_iou)
                miou_2d = float(torch.mean(ious))

            # ["Frame", "mIoU", "lastIoU" "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
            #  "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"]
            row_results = [stepi, miou_2d, last_iou_2d, iou_3d, chamfer_dist, mTransAll, mTransKF, transLast,
                           mAngDiffAll, mAngDiffKF, angDiffLast]

            row_results_rounded = [round(res, 3) if type(res) is float else res for res in row_results]

            self.metrics_writer.writerow(row_results_rounded)
            self.metrics_log.flush()

    def visualize_rotations_per_epoch(self, logged_sgd_translations, logged_sgd_quaternions, frame_losses, frame_i,
                                      gt_rotation=None, gt_translation=None):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 12))  # Adjusted for two subplots, one above the other
        fig.subplots_adjust(left=0.25, right=0.85, hspace=0.5)

        # Current rotation and translation values
        translation_tensors = [t[0, 0, -1].detach().cpu() for t in logged_sgd_translations]
        rotation_tensors = [
            rad_to_deg(quaternion_to_axis_angle(q.detach().cpu()))[0, -1]
            for q in logged_sgd_quaternions
        ]

        # Plot rotations
        axis_labels = ['X-axis rotation', 'Y-axis rotation', 'Z-axis rotation']
        colors = ['yellow', 'green', 'blue']
        for i in range(3):
            values = [tensor[i].item() for tensor in rotation_tensors]
            ax1.plot(range(len(rotation_tensors)), values, label=axis_labels[i], color=colors[i])
            # Plot GT rotations if provided
            if gt_rotation is not None:
                gt_rotation_deg = rad_to_deg(gt_rotation)
                gt_rotation_values = gt_rotation_deg.numpy(force=True)[i].repeat(len(rotation_tensors))
                ax1.plot(range(gt_rotation_values.shape[0]), gt_rotation_values, linestyle='dashed',
                         label=f"GT {axis_labels[i]}", alpha=0.5, color=colors[i])

        ax1.set_xlabel('Gradient descend iteration')
        ax1.set_ylabel('Rotation [degrees]')
        ax1.legend(loc='lower left')

        # Plot translations
        translation_axis_labels = ['X-axis translation', 'Y-axis translation', 'Z-axis translation']
        for i in range(3):
            values = [tensor[i].item() for tensor in translation_tensors]
            ax3.plot(range(len(translation_tensors)), values, label=translation_axis_labels[i], color=colors[i])
            # Plot GT translations if provided
            if gt_translation is not None:
                gt_translation_values = gt_translation.numpy(force=True)[i].repeat(len(translation_tensors))
                ax3.plot(range(gt_translation_values.shape[0]), gt_translation_values, linestyle='dashed',
                         label=f"GT {translation_axis_labels[i]}", alpha=0.5, color=colors[i])

        ax3.set_xlabel('Gradient descend iteration')
        ax3.set_ylabel('Translation')
        ax3.legend(loc='lower left')

        # Adjust the loss plot on the rotation axis for clarity
        self.add_loss_plot(ax1, frame_losses)
        self.add_loss_plot(ax3, frame_losses)

        # Saving the figure
        (Path(self.write_folder) / Path('rotations_by_epoch')).mkdir(exist_ok=True, parents=True)
        fig_path = (Path(self.write_folder) / Path('rotations_by_epoch') / f'rotations_by_epoch_frame_{frame_i}.svg')
        plt.savefig(fig_path)
        plt.close()

    def render_silhouette_overlap(self, last_rendered_silhouette, last_segment_mask, frame_idx):
        last_rendered_silhouette_binary = last_rendered_silhouette[0] > 0.5
        last_segment_mask_binary = last_segment_mask[0] > 0.5
        silh_overlap_image = torch.zeros(1, *last_segment_mask.shape[-2:], 3)
        R = torch.tensor([255.0, 0, 0])
        G = torch.tensor([0, 255.0, 0])
        Y = R + G
        # Set yellow where there is last_rendered_silhouette and last_segment_mask
        indicesG = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesG[:, 0], indicesG[:, 1]] = Y
        # Set red where there is last_rendered_silhouette and not last_segment_mask
        indicesR = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary <= 0))
        silh_overlap_image[0, indicesR[:, 0], indicesR[:, 1]] = R
        # Set green where there is not last_rendered_silhouette and last_segment_mask
        indicesB = torch.nonzero((last_segment_mask_binary <= 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesB[:, 0], indicesB[:, 1]] = G

        silh_overlap_image_np = silh_overlap_image[0].cpu().to(torch.uint8).numpy()
        (self.write_folder / Path('silhouette_overlap')).mkdir(exist_ok=True, parents=True)
        silhouette_overlap_path = self.write_folder / 'silhouette_overlap' / Path(f"silhouette_overlap_{frame_idx}.png")
        imageio.imwrite(silhouette_overlap_path, silh_overlap_image_np)

    def write_keyframe_rotations(self, detached_result, keyframes):
        quaternions = detached_result.quaternions[0]  # Assuming shape is (1, N, 4)
        for k in range(quaternions.shape[0]):
            quaternions[k] = qnorm(quaternions[k])
        # Convert quaternions to Euler angles
        angles_rad = quaternion_to_axis_angle(quaternions)
        # Convert radians to degrees
        angles_deg = angles_rad * 180.0 / math.pi
        rot_axes = ['X-axis: ', 'Y-axis: ', 'Z-axis: ']
        for k in range(angles_rad.shape[0]):
            rotations = [rot_axes[i] + str(round(float(angles_deg[k, i]), 3))
                         for i in range(3)]
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} rotation: " + str(rotations) + '\n')
        for k in range(detached_result.quaternions.shape[1]):
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} translation: str{detached_result.translations[0, 0, k]}\n")
        self.tracking_log.write('\n')
        self.tracking_log.flush()

    def write_all_encoder_rotations(self, encoder: Encoder, last_keyframe_idx):
        self.tracking_log.write("============================================\n")
        self.tracking_log.write("Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n")
        encoder_result_prime, keyframes_prime = self.encoder_result_all_frames(encoder, last_keyframe_idx)
        self.write_keyframe_rotations(encoder_result_prime, keyframes_prime)
        self.tracking_log.write("============================================\n")
        self.tracking_log.write("END of Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n\n\n")

    @staticmethod
    def encoder_result_all_frames(encoder: Encoder, last_keyframe_idx: int):
        keyframes_prime = list(range(last_keyframe_idx))
        encoder_result_prime = encoder(keyframes_prime)
        return encoder_result_prime, keyframes_prime

    def visualize_theoretical_flow(self, bounding_box, keyframe_buffer: KeyframeBuffer, keyframe_buffer_backview,
                                   new_flow_arcs: List[Tuple[int, int]], rgb_encoder: Encoder, deep_encoder: Encoder,
                                   renderer: RenderingKaolin, renderer_backview):
        with torch.no_grad():
            for flow_arc_idx, flow_arc in enumerate(new_flow_arcs):

                source_frame = flow_arc[0]
                target_frame = flow_arc[1]

                # Get optical flow frames
                keyframes = [source_frame, target_frame]
                flow_frames = [source_frame, target_frame]

                # Compute estimated shape
                encoder_result, encoder_result_flow_frames = deep_encoder.frames_and_flow_frames_inference(keyframes,
                                                                                                           flow_frames)

                # Get texture map
                tex_rgb = nn.Sigmoid()(rgb_encoder.texture_map)

                # Render keyframe images
                rendering_result = renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                                    encoder_result.vertices, deep_encoder.face_features, tex_rgb,
                                                    encoder_result.lights)

                rendering_rgb = rendering_result.rendered_image

                rendered_flow_result = renderer.compute_theoretical_flow(encoder_result, encoder_result_flow_frames,
                                                                         flow_arcs_indices=[(0, 1)])

                rendered_keyframe_images = self.write_tensor_into_bbox(rendering_rgb, bounding_box)

                # Extract current and previous rendered images
                source_rendered_image_rgb = rendered_keyframe_images[0, -2]
                target_rendered_image_rgb = rendered_keyframe_images[0, -1]

                # Prepare file paths
                theoretical_flow_paths = self.write_folder / Path('flows')
                renderings_path = self.write_folder / Path('renderings')
                occlusion_maps_path = self.write_folder / Path('rendered_occlusions')

                theoretical_flow_paths.mkdir(exist_ok=True, parents=True)
                renderings_path.mkdir(exist_ok=True, parents=True)
                occlusion_maps_path.mkdir(exist_ok=True, parents=True)

                theoretical_flow_path = theoretical_flow_paths / Path(
                    f"predicted_flow_{source_frame}_{target_frame}.png")
                flow_difference_path = theoretical_flow_paths / Path(
                    f"flow_difference_{source_frame}_{target_frame}.png")
                rendering_path = renderings_path / Path(f"rendering_{target_frame}.png")
                occlusion_path = occlusion_maps_path / Path(f"occlusion_{source_frame}_{target_frame}.png")

                visualize_occlusions(occlusion_path, source_rendered_image_rgb,
                                     rendered_flow_result.rendered_flow_occlusion, alpha=0.5)

                # Convert tensors to NumPy arrays
                target_rendered_image_np = (target_rendered_image_rgb * 255).detach().cpu().numpy().transpose(1, 2, 0)
                target_rendered_image_np = target_rendered_image_np.astype('uint8')

                # Save rendered images
                if flow_arc_idx == 0:
                    imageio.imwrite(rendering_path, target_rendered_image_np)

                # Adjust (0, 1) range to pixel range
                theoretical_flow = rendered_flow_result.theoretical_flow[0, -1].detach().clone().cpu()
                theoretical_flow[0, ...] *= theoretical_flow.shape[-1]
                theoretical_flow[1, ...] *= theoretical_flow.shape[-2]

                # Adjust (0, 1) range to pixel range
                observed_flow = keyframe_buffer.get_flows_between_frames(source_frame, target_frame).observed_flow
                observed_flow = observed_flow.squeeze().detach().clone().cpu()
                observed_flow[0, ...] *= observed_flow.shape[-1]
                observed_flow[1, ...] *= observed_flow.shape[-2]

                # Obtain flow and image illustrations
                # theoretical_flow = self.write_tensor_into_bbox(theoretical_flow, bounding_box)
                # observed_flow = self.write_tensor_into_bbox(observed_flow, bounding_box)

                observed_flow_np = observed_flow.permute(1, 2, 0).numpy()
                theoretical_flow_np = theoretical_flow.permute(1, 2, 0).numpy()

                target_frame_observation = keyframe_buffer.get_observations_for_keyframe(target_frame)
                source_frame_observation = keyframe_buffer.get_observations_for_keyframe(source_frame)

                target_image_segmentation = target_frame_observation.observed_segmentation[
                    0, 0, 0].detach().clone().cpu()
                source_image_segmentation = source_frame_observation.observed_segmentation[
                    0, 0, 0].detach().clone().cpu()

                rendered_flow_occlusion_mask = rendered_flow_result.rendered_flow_occlusion[
                    0, 0, 0].detach().clone().cpu()

                # Visualize flow and flow difference
                flow_illustration = visualize_flow_with_images([source_rendered_image_rgb],
                                                               target_rendered_image_rgb, observed_flows=None,
                                                               gt_flows=theoretical_flow_np,
                                                               gt_silhouette_current=target_image_segmentation,
                                                               gt_silhouettes_prev=[source_image_segmentation],
                                                               flow_occlusion_masks=[rendered_flow_occlusion_mask])

                flow_difference_illustration = compare_flows_with_images([source_rendered_image_rgb],
                                                                         target_rendered_image_rgb,
                                                                         [observed_flow_np], theoretical_flow_np,
                                                                         gt_silhouette_current=target_image_segmentation,
                                                                         gt_silhouette_prev=[source_image_segmentation])

                # Save flow illustrations
                imageio.imwrite(theoretical_flow_path, flow_illustration)
                imageio.imwrite(flow_difference_path, flow_difference_illustration)

    def visualize_flow(self, keyframe_buffer: KeyframeBuffer, keyframe_buffer_backview, flow_arcs,
                       per_pixel_flow_error):
        for flow_arcs in flow_arcs:

            source_frame = flow_arcs[0]
            target_frame = flow_arcs[1]

            flow_observation = keyframe_buffer.get_flows_between_frames(source_frame, target_frame)
            source_frame_observation = keyframe_buffer_backview.get_observations_for_keyframe(source_frame)
            target_frame_observation = keyframe_buffer.get_observations_for_keyframe(target_frame)

            observed_flow = flow_observation.observed_flow.cpu()
            observed_flow_occlusions = flow_observation.observed_flow_occlusion.cpu()
            source_frame_image = source_frame_observation.observed_image.cpu()
            source_frame_segment = source_frame_observation.observed_segmentation.cpu()

            target_frame_image = target_frame_observation.observed_image.cpu()
            target_frame_segment = target_frame_observation.observed_segmentation.cpu()

            observed_flow = flow_unit_coords_to_image_coords(observed_flow)
            observed_flow_reordered = observed_flow.squeeze().permute(1, 2, 0).numpy()

            source_image_discrete: torch.Tensor = (source_frame_image * 255).to(torch.uint8).squeeze()
            target_image_discrete: torch.Tensor = (target_frame_image * 255).to(torch.uint8).squeeze()

            source_frame_segment_squeezed = source_frame_segment.squeeze()
            target_frame_segment_squeezed = target_frame_segment.squeeze()
            observed_flow_occlusions_squeezed = observed_flow_occlusions.squeeze()

            flow_illustration = visualize_flow_with_images([source_image_discrete], target_image_discrete,
                                                           [observed_flow_reordered], None,
                                                           gt_silhouette_current=source_frame_segment_squeezed,
                                                           gt_silhouettes_prev=[target_frame_segment_squeezed],
                                                           flow_occlusion_masks=[observed_flow_occlusions_squeezed])

            # Define output file paths
            new_image_path = self.gt_imgs_path / Path(f'gt_img_{source_frame}_{target_frame}.png')
            flow_image_path = self.flows_path / Path(f'flow_{source_frame}_{target_frame}.png')
            occlusion_path = self.occlusion_maps_path / Path(f"occlusion_{source_frame}_{target_frame}.png")

            visualize_occlusions(occlusion_path, source_frame_image.squeeze(),
                                 observed_flow_occlusions_squeezed, alpha=0.5)

            transform = transforms.ToPILImage()
            target_image_PIL = transform(target_image_discrete)

            # Save the images to disk
            imageio.imwrite(new_image_path, target_image_PIL)
            imageio.imwrite(flow_image_path, flow_illustration)

            # PER PIXEL FLOW ERROR VISUALIZATION
            if per_pixel_flow_error is not None:
                per_pixel_flow_loss_np = per_pixel_flow_error[:, -1].squeeze().detach().cpu().numpy()

                # Normalize values for visualization (optional)
                per_pixel_flow_loss_np_norm = (per_pixel_flow_loss_np - per_pixel_flow_loss_np.min()) / \
                                              (per_pixel_flow_loss_np.max() - per_pixel_flow_loss_np.min()) * 255

                # Convert to uint8 and save using imageio
                output_loss_viz = self.write_folder / Path('losses')
                output_loss_viz.mkdir(parents=True, exist_ok=True)
                output_filename = output_loss_viz / f'end_point_error_frame_{source_frame}_{target_frame}.png'
                imageio.imwrite(output_filename, per_pixel_flow_loss_np_norm.astype('uint8'))


def visualize_occlusions(occlusion_path, source_rendered_image_rgb, rendered_flow_occlusion, alpha):
    occlusion_mask = rendered_flow_occlusion[:, -1:].detach().clone().repeat(1, 1, 3, 1, 1)
    blended_image = alpha * occlusion_mask + (1 - alpha) * source_rendered_image_rgb
    blended_image_np = blended_image.squeeze().cpu().numpy()
    blended_image_np = (blended_image_np * 255).astype(np.uint8).transpose(1, 2, 0)
    imageio.imwrite(occlusion_path, blended_image_np)


def load_gt_annotations_file(file_path):
    # Initialize empty lists to store the data
    frames = []
    rotations_degrees = []
    translations = []

    # Load the CSV file
    with open(file_path, 'r') as csvfile:
        reader: Iterable[Dict] = csv.DictReader(csvfile)
        for row in reader:
            # Append frame number
            frames.append(int(row['frame']))

            # Append rotations in degrees
            rotations_degrees.append([
                float(row['rot_x']),
                float(row['rot_y']),
                float(row['rot_z'])
            ])

            # Append translations
            translations.append([
                float(row['trans_x']),
                float(row['trans_y']),
                float(row['trans_z'])
            ])

    # Convert rotations from degrees to radians
    rotations_radians = [[math.radians(rot[0]), math.radians(rot[1]), math.radians(rot[2])] for rot in
                         rotations_degrees]

    # Create tensors
    rotations_tensor = torch.tensor(rotations_radians).unsqueeze(0)
    translations_tensor = torch.tensor(translations).unsqueeze(0).unsqueeze(1)

    return frames, rotations_tensor, translations_tensor

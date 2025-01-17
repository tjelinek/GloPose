from itertools import product
from pathlib import Path

import numpy as np
import torch
from kornia.geometry import axis_angle_to_quaternion, quaternion_to_axis_angle
from matplotlib import pyplot as plt

from data_structures.keyframe_buffer import FlowObservation, FrameObservation
from models.loss import FMOLoss
from models.rendering import infer_normalized_renderings


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
            rotation_tensor_rad = torch.deg2rad(rotation_tensor_deg)
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
                                                keyframes_encoder_result=encoder_result)

            losses_all, losses, joint_loss, per_pixel_error = loss_result

            joint_losses[i, j] = joint_loss

    return joint_losses


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
            gt_rotation_deg_prev = torch.rad2deg(tracking6d.gt_cam_to_obj_rotations[0, stepi - 1]).cpu()
            gt_translation_prev = tracking6d.gt_cam_to_obj_translations[0, 0, stepi - 1].cpu()

            gt_rotation_deg_current = torch.rad2deg(tracking6d.gt_cam_to_obj_rotations[0, stepi]).cpu()
            gt_translation_current = tracking6d.gt_cam_to_obj_translations[0, 0, stepi].cpu()

            gt_translation_diff = gt_translation_current - gt_translation_prev
            gt_rotation_diff = gt_rotation_deg_current - gt_rotation_deg_prev

            gt_translation = tracking6d.logged_sgd_translations[0][0, 0, 0].detach().cpu() + gt_translation_diff

            gt_rotation_quaternion = tracking6d.logged_sgd_quaternions[0].detach().cpu()
            gt_rotation_rad = quaternion_to_axis_angle(gt_rotation_quaternion)
            last_rotation_deg = torch.rad2deg(gt_rotation_rad)[0, 0]

            gt_rotation_deg = last_rotation_deg + gt_rotation_diff
        else:
            gt_rotation_deg = torch.rad2deg(tracking6d.gt_cam_to_obj_rotations[0, stepi]).cpu()
            gt_translation = tracking6d.gt_cam_to_obj_translations[0, 0, stepi].cpu()

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
                   extent=(float(translations_space[0]), float(translations_space[-1]),
                           float(rotations_space[-1]), float(rotations_space[0])),
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
            iteration_rotation_deg = float(torch.rad2deg(iteration_rotation_rad)[0, -1, rot_axis_idx])

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
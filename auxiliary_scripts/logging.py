from typing import Dict

import os
import math

import torch
import cv2
import imageio
import csv
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from kornia.geometry.conversions import quaternion_to_angle_axis, QuaternionCoeffOrder, angle_axis_to_quaternion
from pytorch3d.loss.chamfer import chamfer_distance

from models.loss import fmo_loss
from segmentations import create_mask_from_string, get_bbox, pad_image
from utils import write_video, segment2bbox, qnorm, quaternion_angular_difference
from helpers.torch_helpers import write_renders
from models.kaolin_wrapper import write_obj_mesh
from models.encoder import EncoderResult
from flow import visualize_flow_with_images, compare_flows_with_images


def visualize_flow(observed_flow, image, image_new, image_prev, segment, stepi, output_dir, per_pixel_flow_error):
    """
    Visualize optical flow between two images and save the results as image files.

    Args:
        observed_flow (torch.Tensor): Optical flow w.r.t. the image coordinate system [-1, 1]. Tensor must be detached.
        image (torch.Tensor): Original image tensor.
        image_new (torch.Tensor): New (second) image tensor.
        image_prev (torch.Tensor): Previous (first) image tensor.
        segment (torch.Tensor): Segmentation mask tensor.
        stepi (int): Index of the current step in the frame sequence.
        output_dir (Path): Flow output directory
        per_pixel_flow_error: Error mask

    Returns:
        None. The function saves multiple visualization images to the disk.
    """
    flow_video_up = observed_flow.detach().clone()
    flow_video_up[:, 0, ...] = flow_video_up[:, 0, ...] * flow_video_up.shape[-1]
    flow_video_up[:, 1, ...] = flow_video_up[:, 1, ...] * flow_video_up.shape[-2]
    flow_video_up = flow_video_up[0].permute(1, 2, 0).cpu().numpy()

    # flow_image = transforms.ToTensor()(flow_viz.flow_to_image(flow_video_up))
    # image_small_dims = image.shape[-2], image.shape[-1]
    # flow_image_small = transforms.Resize(image_small_dims)(flow_image)
    # segmentation_mask = segment[0, 0, -1, :, :].to(torch.bool).unsqueeze(0).repeat(3, 1, 1).cpu().detach()
    # flow_image_segmented = flow_image_small.mul(segmentation_mask)
    image_prev_reformatted: torch.Tensor = image_prev.to(torch.uint8)[0]
    image_new_reformatted: torch.Tensor = image_new.to(torch.uint8)[0]

    flow_illustration = visualize_flow_with_images(image_prev_reformatted, image_new_reformatted, flow_video_up)
    transform = transforms.ToPILImage()
    # image_pure_flow_segmented = transform(flow_image_segmented)
    image_new_pil = transform(image_new[0] / 255.0)
    # image_old_pil = transform(image_prev[0] / 255.0)
    (output_dir / Path('flows')).mkdir(exist_ok=True, parents=True)
    (output_dir / Path('gt_imgs')).mkdir(exist_ok=True, parents=True)

    # Define output file paths
    # prev_image_path = output_dir / Path('gt_imgs') / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_1.png')
    new_image_path = output_dir / Path('gt_imgs') / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_2.png')
    # flow_segm_path = output_dir / Path('flow_segmented_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_image_path = output_dir / Path('flows') / Path('flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')

    # Save the images to disk
    # imageio.imwrite(flow_segm_path, image_pure_flow_segmented)
    imageio.imwrite(new_image_path, image_new_pil)
    # imageio.imwrite(prev_image_path, image_old_pil)
    imageio.imwrite(flow_image_path, flow_illustration)

    # PER PIXEL FLOW ERROR VISUALIZATION
    numpy_array = per_pixel_flow_error.squeeze().detach().cpu().numpy()

    # Normalize values for visualization (optional)
    normalized_array = (numpy_array - numpy_array.min()) / (numpy_array.max() - numpy_array.min()) * 255

    # Convert to uint8 and save using imageio
    output_loss_viz = output_dir / Path('losses')
    output_loss_viz.mkdir(parents=True, exist_ok=True)
    output_filename = output_loss_viz / f'end_point_error_frame_{stepi}.png'
    imageio.imwrite(output_filename, normalized_array.astype('uint8'))


class WriteResults:

    def __init__(self, write_folder, images, num_frames):
        self.all_input = cv2.VideoWriter(os.path.join(write_folder, 'all_input.avi'), cv2.VideoWriter_fourcc(*"MJPG"),
                                         10, (images.shape[4], images.shape[3]), True)
        self.all_segm = cv2.VideoWriter(os.path.join(write_folder, 'all_segm.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                        (images.shape[4], images.shape[3]), True)
        self.all_proj = cv2.VideoWriter(os.path.join(write_folder, 'all_proj.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                        (images.shape[4], images.shape[3]), True)
        self.all_proj_filtered = cv2.VideoWriter(os.path.join(write_folder, 'all_proj_filtered.avi'),
                                                 cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                                 (images.shape[4], images.shape[3]), True)
        self.baseline_iou = -np.ones((num_frames - 1, 1))
        self.our_iou = -np.ones((num_frames - 1, 1))
        self.tracking_log = open(Path(write_folder) / "tracking_log.txt", "w")
        self.metrics_log = open(Path(write_folder) / "tracking_metrics_log.txt", "w")
        self.metrics_writer = csv.writer(self.metrics_log)

        self.metrics_writer.writerow(["Frame", "mIoU", "lastIoU", "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
                                      "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"])

        self.tensorboard_log_dir = Path(write_folder) / Path("logs")
        self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_log = None

    def __del__(self):
        self.all_input.release()
        self.all_segm.release()
        self.all_proj.release()
        self.all_proj_filtered.release()

        self.tracking_log.close()
        self.metrics_log.close()

    def set_tensorboard_log_for_frame(self, frame_i):
        self.tensorboard_log = SummaryWriter(str(self.tensorboard_log_dir / f'Frame_{frame_i}'))

    def write_into_tensorboard_log(self, sgd_iter: int, values_dict: Dict):

        for field_name, value in values_dict.items():
            self.tensorboard_log.add_scalar(field_name, value, sgd_iter)

    def write_results(self, tracking6d, b0, bboxes, our_losses, silh_losses, stepi, encoder_result,
                      ground_truth_segments, images, images_feat, tex, frame_losses):

        detached_result = EncoderResult(*[it.clone().detach() if type(it) is torch.Tensor else it
                                          for it in encoder_result])
        if tracking6d.gt_texture is not None:
            tex = tracking6d.gt_texture

        with torch.no_grad():

            self.visualize_rotations_per_epoch(tracking6d, frame_losses, stepi)

            stochastically_added_keyframes = list(set(tracking6d.all_keyframes.keyframes) -
                                                  set(tracking6d.active_keyframes.keyframes))
            print("Keyframes:", tracking6d.active_keyframes.keyframes)
            print("Stochastically added keyframes: ", stochastically_added_keyframes)

            self.tracking_log.write(f"Step {stepi}:\n")
            self.tracking_log.write(f"Keyframes: {tracking6d.all_keyframes.keyframes}\n")
            self.tracking_log.write(f"Stochastically added keyframes: "
                                    f"{stochastically_added_keyframes}\n")

            self.write_keyframe_rotations(detached_result, tracking6d.active_keyframes.keyframes)

            self.write_all_encoder_rotations(tracking6d)

            if tracking6d.config.features == 'rgb':
                tex = detached_result.texture_maps
            feat_renders_crop = tracking6d.get_rendered_image_features(detached_result.lights,
                                                                       detached_result.quaternions,
                                                                       detached_result.texture_maps,
                                                                       detached_result.translations,
                                                                       detached_result.vertices)

            renders, renders_crop = tracking6d.get_rendered_image(b0, detached_result.lights,
                                                                  detached_result.quaternions,
                                                                  tex,
                                                                  detached_result.translations,
                                                                  detached_result.vertices)

            rendered_silhouette = renders[:, :, 0, -1:]
            last_rendered_silhouette = rendered_silhouette[0, -1]
            last_segment = ground_truth_segments[:, -1:]
            last_segment_reshaped = last_segment[:, 0]
            last_segment_mask = last_segment_reshaped[:, 1]

            self.render_silhouette_overlap(last_rendered_silhouette, last_segment_mask,
                                           stepi, tracking6d)

            write_renders(feat_renders_crop, tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=0)
            write_renders(renders_crop, tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=1)
            write_renders(torch.cat(
                (images[:, :, None, :, b0[0]:b0[1], b0[2]:b0[3]], feat_renders_crop[:, :, :, -1:]), 3),
                tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=2)
            write_obj_mesh(detached_result.vertices[0].cpu().numpy(), tracking6d.best_model["faces"],
                           tracking6d.encoder.face_features[0].cpu().numpy(),
                           os.path.join(tracking6d.write_folder, f'mesh_{stepi}.obj'), "model_" + str(stepi) + ".mtl")
            save_image(detached_result.texture_maps[:, :3], os.path.join(tracking6d.write_folder, 'tex_deep.png'))
            save_image(tex, os.path.join(tracking6d.write_folder, f'tex_{stepi}.png'))

            with open(tracking6d.write_folder / "model.mtl", "r") as file:
                lines = file.readlines()

            # Replace the last line
            lines[-1] = f"map_Kd tex_{stepi}.png\n"

            # Write the result to a new file
            with open(tracking6d.write_folder / f"model_{stepi}.mtl", "w") as file:
                file.writelines(lines)

            write_video(renders[0, :, 0, :3].detach().cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(tracking6d.write_folder, 'im_recon.avi'), fps=6)
            write_video(images[0, :, :3].cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(tracking6d.write_folder, 'input.avi'), fps=6)
            write_video(
                (images[0, :, :3] * ground_truth_segments[0, :, 1:2]).cpu().numpy().transpose(2, 3, 1, 0),
                os.path.join(tracking6d.write_folder, 'segments.avi'), fps=6)
            for tmpi in range(renders.shape[1]):
                img = images[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]]
                seg = ground_truth_segments[0, :, 1:2][tmpi, :, b0[0]:b0[1], b0[2]:b0[3]].clone()
                save_image(seg, os.path.join(tracking6d.write_folder, 'imgs', 's{}.png'.format(tmpi)))
                seg[seg == 0] = 0.35
                save_image(img, os.path.join(tracking6d.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
                save_image(images_feat[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]],
                           os.path.join(tracking6d.write_folder, 'imgs', 'if{}.png'.format(tmpi)))
                save_image(torch.cat((img, seg), 0),
                           os.path.join(tracking6d.write_folder, 'imgs', 'is{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, [3, 3, 3]],
                           os.path.join(tracking6d.write_folder, 'imgs', 'm{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, :],
                           os.path.join(tracking6d.write_folder, 'imgs', 'r{}.png'.format(tmpi)))
                save_image(feat_renders_crop[0, tmpi, 0, :],
                           os.path.join(tracking6d.write_folder, 'imgs', 'f{}.png'.format(tmpi)))

            if type(bboxes) is dict or (bboxes[stepi][0] == 'm'):
                gt_segm = None
                if (not type(bboxes) is dict) and bboxes[stepi][0] == 'm':
                    m_, offset_ = create_mask_from_string(bboxes[stepi][1:].split(','))
                    gt_segm = last_segment[0, 0, -1] * 0
                    gt_segm[offset_[1]:offset_[1] + m_.shape[0], offset_[0]:offset_[0] + m_.shape[1]] = \
                        torch.from_numpy(m_)
                elif stepi in bboxes:
                    gt_segm = tracking6d.tracker.process_segm(bboxes[stepi])[0].to(tracking6d.device)
                    gt_segm = pad_image(gt_segm)
                if gt_segm is not None:
                    self.baseline_iou[stepi - 1] = float((last_segment[0, 0, -1] * gt_segm > 0).sum()) / float(
                        ((last_segment[0, 0, -1] + gt_segm) > 0).sum() + 0.00001)
                    self.our_iou[stepi - 1] = float((renders[0, -1, 0, 3] * gt_segm > 0).sum()) / float(
                        ((renders[0, -1, 0, 3] + gt_segm) > 0).sum() + 0.00001)
            elif bboxes is not None:
                bbox = tracking6d.config.image_downsample * torch.tensor(
                    [bboxes[stepi] + [0, 0, bboxes[stepi][0], bboxes[stepi][1]]])
                self.baseline_iou[stepi - 1] = bops.box_iou(bbox,
                                                            torch.tensor([segment2bbox(last_segment[0, 0, -1])],
                                                                         dtype=torch.float64))
                self.our_iou[stepi - 1] = bops.box_iou(bbox, torch.tensor([segment2bbox(renders[0, -1, 0, 3])],
                                                                          dtype=torch.float64))
            print('Baseline IoU {}, our IoU {}'.format(self.baseline_iou[stepi - 1], self.our_iou[stepi - 1]))
            np.savetxt(os.path.join(tracking6d.write_folder, 'baseline_iou.txt'), self.baseline_iou, fmt='%.10f',
                       delimiter='\n')
            np.savetxt(os.path.join(tracking6d.write_folder, 'iou.txt'), self.our_iou, fmt='%.10f', delimiter='\n')
            np.savetxt(os.path.join(tracking6d.write_folder, 'losses.txt'), our_losses, fmt='%.10f', delimiter='\n')
            self.all_input.write(
                (images[0, :, :3].clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1, 0)[:, :,
                 [2, 1, 0], -1] * 255).astype(np.uint8))
            self.all_segm.write(((images[0, :, :3] * ground_truth_segments[0, :, 1:2]).clamp(min=0,
                                                                                             max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))
            self.all_proj.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1,
                                                                                                           0)[:, :,
                                 [2, 1, 0], -1] * 255).astype(np.uint8))
            if silh_losses[-1] > 0.3:
                renders[0, -1, 0, 3] = last_segment[0, 0, -1]
                renders[0, -1, 0, :3] = images[0, -1, :3] * last_segment[0, 0, -1]
            self.all_proj_filtered.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))

    def evaluate_metrics(self, stepi, tracking6d, keyframes, predicted_vertices, predicted_quaternion,
                         predicted_translation, predicted_mask, gt_vertices=None, gt_rotation=None, gt_translation=None,
                         gt_object_mask=None):

        with torch.no_grad():
            encoder_result_all_frames, _ = self.encoder_result_all_frames(tracking6d)

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
                gt_quaternion = angle_axis_to_quaternion(gt_rotation, order=QuaternionCoeffOrder.WXYZ)

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
                    frame_iou = 1 - fmo_loss(gt_object_mask[None, None, :, frame_i],
                                             predicted_mask[None, None, :, frame_i])
                    ious[frame_i] = frame_iou

                last_iou_2d = float(frame_iou)
                miou_2d = float(torch.max(ious))

            # ["Frame", "mIoU", "lastIoU" "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
            #  "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"]
            row_results = [stepi, miou_2d, last_iou_2d, iou_3d, chamfer_dist, mTransAll, mTransKF, transLast,
                           mAngDiffAll, mAngDiffKF, angDiffLast]

            row_results_rounded = [round(res, 3) if type(res) is float else res for res in row_results]

            self.metrics_writer.writerow(row_results_rounded)
            self.metrics_log.flush()

    @staticmethod
    def visualize_rotations_per_epoch(tracking6d, frame_losses, stepi):
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(left=0.25, right=0.85)

        rotation_tensors = tracking6d.encoder.rotation_by_gd_iter
        translation_tensors = tracking6d.encoder.translation_by_gd_iter

        axis_labels = ['X-axis rotation', 'Y-axis rotation', 'Z-axis rotation']
        for i in range(3):
            values = [tensor[i].item() for tensor in rotation_tensors]
            ax1.plot(range(len(rotation_tensors)), values, label=axis_labels[i])

        ax1.set_xlabel('Gradient descend iteration')
        ax1.set_ylabel('Rotation [degrees]')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss')
        ax2.plot(range(len(frame_losses)), frame_losses, color='red', label='Loss')

        ax3 = ax1.twinx()
        ax3.set_ylabel('Translation')
        translation_axis_labels = ['X-axis translation', 'Y-axis translation', 'Z-axis translation']
        for i in range(3):
            values = [tensor[i].item() for tensor in translation_tensors]
            ax3.plot(range(len(translation_tensors)), values, label=translation_axis_labels[i], linestyle='dashed')
        ax3.spines.left.set_position(("axes", -0.2))
        ax3.spines.left.set_visible(True)
        ax3.spines.right.set_visible(False)
        ax3.yaxis.set_label_position('left')
        ax3.yaxis.set_ticks_position('left')

        # Create a joint legend
        handles, labels = [], []
        for ax in [ax1, ax2, ax3]:
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)

            ax_min, ax_max = ax.get_ylim()
            ax_range = max(abs(ax_min), abs(ax_max))
            ax.set_ylim((-ax_range, ax_range))

        fig.legend(handles, labels, loc='upper right')

        (Path(tracking6d.write_folder) / Path('rotations_by_epoch')).mkdir(exist_ok=True, parents=True)
        fig_path = Path(tracking6d.write_folder) / Path('rotations_by_epoch') / \
                   ('rotations_by_epoch_frame_' + str(stepi) + '.png')
        plt.savefig(fig_path)
        plt.close()

    @staticmethod
    def render_silhouette_overlap(last_rendered_silhouette, last_segment_mask, stepi, tracking6d):
        last_rendered_silhouette_binary = last_rendered_silhouette[0] > 0
        last_segment_mask_binary = last_segment_mask[0] > 0
        silh_overlap_image = torch.zeros(1, *last_segment_mask.shape[-2:], 3)
        R = torch.tensor([255.0, 0, 0])
        G = torch.tensor([0, 255.0, 0])
        Y = R + G
        # Set green where there is silhouette1 and silhouette2
        indicesG = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesG[:, 0], indicesG[:, 1]] = G
        # Set red where there is silhouette1 and not silhouette2
        indicesR = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary <= 0))
        silh_overlap_image[0, indicesR[:, 0], indicesR[:, 1]] = R
        # Set yellow where there is not silhouette1 and silhouette2
        indicesB = torch.nonzero((last_segment_mask_binary <= 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesB[:, 0], indicesB[:, 1]] = Y

        silh_overlap_image_np = silh_overlap_image[0].cpu().to(torch.uint8).numpy()
        (tracking6d.write_folder / Path('silhouette_overlap')).mkdir(exist_ok=True, parents=True)
        silhouette_overlap_path = tracking6d.write_folder / Path('silhouette_overlap') / \
                                  Path(f"silhouette_overlap_{stepi}.png")
        imageio.imwrite(silhouette_overlap_path, silh_overlap_image_np)

    def write_keyframe_rotations(self, detached_result, keyframes):
        quaternions = detached_result.quaternions[0]  # Assuming shape is (1, N, 4)
        for k in range(quaternions.shape[0]):
            quaternions[k] = qnorm(quaternions[k])
        # Convert quaternions to Euler angles
        angles_rad = quaternion_to_angle_axis(quaternions, order=QuaternionCoeffOrder.WXYZ)
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

    def write_all_encoder_rotations(self, tracking6d):
        self.tracking_log.write("============================================\n")
        tracking6d.write_results.tracking_log.write("Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n")
        encoder_result_prime, keyframes_prime = self.encoder_result_all_frames(tracking6d)
        self.write_keyframe_rotations(encoder_result_prime, keyframes_prime)
        self.tracking_log.write("============================================\n")
        self.tracking_log.write("END of Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n\n\n")

    def encoder_result_all_frames(self, tracking6d):
        keyframes_prime = list(range(max(tracking6d.all_keyframes.keyframes) + 1))
        encoder_result_prime = tracking6d.encoder(keyframes_prime)
        return encoder_result_prime, keyframes_prime


def visualize_theoretical_flow(tracking6d, theoretical_flow, bounding_box, observed_flow, opt_frames, stepi):
    """
    Visualizes the theoretical flow and related images for a given step.

    Args:
        tracking6d (Tracking6D): The Tracking6D instance.
        theoretical_flow (torch.Tensor): Theoretical flow tensor with shape (B, H, W, 2) w.r.t. the [-1, 1]
                                         image coordinates.
        bounding_box (torch.Tensor: Bounding box of the observed object
        observed_flow (torch.Tensor): Observed flow tensor with shape (B, 2, H, W) w.r.t. the [-1, 1] image
                                      coordinates.
        opt_frames (list): List of optical flow frames.
        stepi (int): Step index.

    Returns:
        None
    """
    with torch.no_grad():
        # Get optical flow frames
        opt_frames_prime = [max(opt_frames) - 1, max(opt_frames)]

        # Compute estimated shape
        enc_result_prime, _ = tracking6d.frames_and_flow_frames_inference(opt_frames_prime, opt_frames_prime)

        # Get texture map
        tex_rgb = nn.Sigmoid()(tracking6d.rgb_encoder.texture_map) if tracking6d.gt_texture is None \
            else tracking6d.gt_texture

        # Render keyframe images
        rendered_keyframe_images, _ = tracking6d.get_rendered_image(bounding_box, enc_result_prime.lights,
                                                                    enc_result_prime.quaternions, tex_rgb,
                                                                    enc_result_prime.translations,
                                                                    enc_result_prime.vertices)

        # Extract current and previous rendered images
        current_rendered_image_rgba = rendered_keyframe_images[0, -1, ...]
        previous_rendered_image_rgba = rendered_keyframe_images[0, -2, ...]
        current_rendered_image_rgb = current_rendered_image_rgba[:, :3, ...]
        previous_rendered_image_rgb = previous_rendered_image_rgba[:, :3, ...]

        # Prepare file paths
        (tracking6d.write_folder / Path('flows')).mkdir(exist_ok=True, parents=True)
        (tracking6d.write_folder / Path('renderings')).mkdir(exist_ok=True, parents=True)

        theoretical_flow_path = tracking6d.write_folder / Path('flows') / \
                                Path(f"predicted_flow_{stepi}_{stepi + 1}.png")
        flow_difference_path = tracking6d.write_folder / Path('flows') / \
                               Path(f"flow_difference_{stepi}_{stepi + 1}.png")
        rendering_1_path = tracking6d.write_folder / Path('renderings') / \
                           Path(f"rendering_{stepi}_{stepi + 1}_1.png")
        rendering_2_path = tracking6d.write_folder / Path('renderings') / \
                           Path(f"rendering_{stepi}_{stepi + 1}_2.png")

        # Convert tensors to NumPy arrays
        previous_rendered_image_np = (previous_rendered_image_rgb[0] * 255).detach().cpu().numpy().transpose(1, 2, 0)
        previous_rendered_image_np = previous_rendered_image_np.astype('uint8')
        current_rendered_image_np = (current_rendered_image_rgb[0] * 255).detach().cpu().numpy().transpose(1, 2, 0)
        current_rendered_image_np = current_rendered_image_np.astype('uint8')

        # Save rendered images
        imageio.imwrite(rendering_1_path, previous_rendered_image_np)
        imageio.imwrite(rendering_2_path, current_rendered_image_np)

        # Clone theoretical flow and adjust coordinates
        adjusted_theoretical_flow = theoretical_flow.clone().detach()
        adjusted_theoretical_flow[..., 0] *= adjusted_theoretical_flow.shape[-2]
        adjusted_theoretical_flow[..., 1] *= adjusted_theoretical_flow.shape[-3]

        # Prepare observed flow
        adjusted_observed_flow = observed_flow.clone().permute(0, 2, 3, 1)
        adjusted_observed_flow[..., 0] *= observed_flow.shape[-1]
        adjusted_observed_flow[..., 1] *= observed_flow.shape[-2]

        # Obtain flow and image illustrations
        flow_up = adjusted_theoretical_flow[:, -1].cpu()[0].permute(2, 0, 1)
        theoretical_flow_up = tracking6d.write_image_into_bbox(bounding_box, flow_up)
        observed_flow_up = tracking6d.write_image_into_bbox(bounding_box, adjusted_observed_flow[0].permute(2, 0, 1))
        observed_flow_np = observed_flow_up.permute(1, 2, 0).cpu().numpy()
        theoretical_flow_np = theoretical_flow_up.detach().cpu().numpy()
        theoretical_flow_np = theoretical_flow_np.transpose(1, 2, 0)

        # Visualize flow and flow difference
        flow_illustration = visualize_flow_with_images(previous_rendered_image_rgb[0], current_rendered_image_rgb[0],
                                                       theoretical_flow_np)
        flow_difference_illustration = compare_flows_with_images(previous_rendered_image_rgb[0],
                                                                 current_rendered_image_rgb[0],
                                                                 observed_flow_np, theoretical_flow_np)

        # Save flow illustrations
        imageio.imwrite(theoretical_flow_path, flow_illustration)
        imageio.imwrite(flow_difference_path, flow_difference_illustration)


def load_gt_annotations_file(file_path):
    # Initialize empty lists to store the data
    frames = []
    rotations_degrees = []
    translations = []

    # Load the CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
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

import os
import math

import imageio
import torch
import cv2
import imageio
import numpy as np
from torch import nn
from pathlib import Path
from torchvision import transforms
from torchvision.utils import save_image

from utils import euler_from_quaternion, write_video
from helpers.torch_helpers import write_renders
from models.kaolin_wrapper import write_obj_mesh
from GMA.core.utils import flow_viz
from models.encoder import EncoderResult
from flow import visualize_flow_with_images


def visualize_flow(flow_video_up, image, image_new, image_prev, segment, stepi, output_dir):
    """
    Visualize optical flow between two images and save the results as image files.

    Args:
        flow_video_up (torch.Tensor): Upsampled optical flow tensor.
        image (torch.Tensor): Original image tensor.
        image_new (torch.Tensor): New (second) image tensor.
        image_prev (torch.Tensor): Previous (first) image tensor.
        segment (torch.Tensor): Segmentation mask tensor.
        stepi (int): Index of the current step in the frame sequence.
        output_dir (Path): Flow output directory

    Returns:
        None. The function saves multiple visualization images to the disk.
    """
    flow_image = transforms.ToTensor()(flow_viz.flow_to_image(flow_video_up))
    image_small_dims = image.shape[-2], image.shape[-1]
    flow_image_small = transforms.Resize(image_small_dims)(flow_image)
    segmentation_mask = segment[0, 0, -1, :, :].to(torch.bool).unsqueeze(0).repeat(3, 1, 1).cpu().detach()
    flow_image_segmented = flow_image_small.mul(segmentation_mask)
    image_prev_reformatted: torch.Tensor = image_prev.to(torch.uint8)[0]
    image_new_reformatted: torch.Tensor = image_new.to(torch.uint8)[0]

    flow_illustration = visualize_flow_with_images(image_prev_reformatted, image_new_reformatted, flow_video_up)
    transform = transforms.ToPILImage()
    image_pure_flow_segmented = transform(flow_image_segmented)
    image_new_pil = transform(image_new[0] / 255.0)
    image_old_pil = transform(image_prev[0] / 255.0)

    # Define output file paths
    prev_image_path = output_dir / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_1.png')
    new_image_path = output_dir / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_2.png')
    flow_segm_path = output_dir / Path('flow_segmented_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_image_path = output_dir / Path('flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')

    # Save the images to disk
    imageio.imwrite(flow_segm_path, image_pure_flow_segmented)
    imageio.imwrite(new_image_path, image_new_pil)
    imageio.imwrite(prev_image_path, image_old_pil)
    imageio.imwrite(flow_image_path, flow_illustration)


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

    def __del__(self):
        self.all_input.release()
        self.all_segm.release()
        self.all_proj.release()
        self.all_proj_filtered.release()

    def write_results(self, tracking6d, b0, bboxes, our_losses, segment, silh_losses, stepi, observed_flows,
                      encoder_result):

        detached_result = EncoderResult(*[it.detach() if type(it) is torch.Tensor else it for it in encoder_result])

        if tracking6d.config.features == 'deep':
            tracking6d.rgb_apply(tracking6d.images[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                 tracking6d.segments[:, :, :, b0[0]:b0[1], b0[2]:b0[3]], observed_flows,
                                 tracking6d.keyframes)
            tex = nn.Sigmoid()(tracking6d.rgb_encoder.texture_map)
        if tracking6d.gt_texture is not None:
            tex = tracking6d.gt_texture

        with torch.no_grad():
            # encoder_result = tracking6d.encoder(tracking6d.keyframes)

            last_quaternion = detached_result.quaternions[0, -1]
            prev_quaternion = detached_result.quaternions[0, -2]
            first_quaternion = detached_result.quaternions[0, 0]
            euler_angles_last = euler_from_quaternion(last_quaternion[0], last_quaternion[1], last_quaternion[2],
                                                      last_quaternion[3])
            euler_angles_first = euler_from_quaternion(first_quaternion[0], first_quaternion[1], first_quaternion[2],
                                                       first_quaternion[3])
            euler_angles_prev = euler_from_quaternion(prev_quaternion[0], prev_quaternion[1], prev_quaternion[2],
                                                      prev_quaternion[3])

            print("Keyframes:", tracking6d.keyframes)

            print("Last estimated rotation:", [(float(euler_angles_last[i]) * 180 / math.pi -
                                                float(euler_angles_first[i]) * 180 / math.pi) % 360
                                               for i in range(len(euler_angles_last))])
            print("Previous estimated rotation:", [(float(euler_angles_prev[i]) * 180 / math.pi -
                                                    float(euler_angles_first[i]) * 180 / math.pi) % 360
                                                   for i in range(len(euler_angles_last))])

            self.tracking_log.write(
                "Last estimated rotation:" + str([(float(euler_angles_last[i]) * 180 / math.pi -
                                                   float(
                                                       euler_angles_first[i]) * 180 / math.pi) % 360
                                                  for i in range(len(euler_angles_last))]))
            self.tracking_log.write('\n')
            self.tracking_log.flush()

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

            write_renders(feat_renders_crop, tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=0)
            write_renders(renders_crop, tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=1)
            write_renders(torch.cat(
                (tracking6d.images[:, :, None, :, b0[0]:b0[1], b0[2]:b0[3]], feat_renders_crop[:, :, :, -1:]), 3),
                tracking6d.write_folder, tracking6d.config.max_keyframes + 1, ids=2)
            write_obj_mesh(detached_result.vertices[0].cpu().numpy(), tracking6d.best_model["faces"],
                           tracking6d.encoder.face_features[0].cpu().numpy(),
                           os.path.join(tracking6d.write_folder, f'mesh_{stepi}.obj'))
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
            write_video(tracking6d.images[0, :, :3].cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(tracking6d.write_folder, 'input.avi'), fps=6)
            write_video(
                (tracking6d.images[0, :, :3] * tracking6d.segments[0, :, 1:2]).cpu().numpy().transpose(2, 3, 1, 0),
                os.path.join(tracking6d.write_folder, 'segments.avi'), fps=6)
            for tmpi in range(renders.shape[1]):
                img = tracking6d.images[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]]
                seg = tracking6d.segments[0, :, 1:2][tmpi, :, b0[0]:b0[1], b0[2]:b0[3]].clone()
                save_image(seg, os.path.join(tracking6d.write_folder, 'imgs', 's{}.png'.format(tmpi)))
                seg[seg == 0] = 0.35
                save_image(img, os.path.join(tracking6d.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
                save_image(tracking6d.images_feat[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]],
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
                    gt_segm = segment[0, 0, -1] * 0
                    gt_segm[offset_[1]:offset_[1] + m_.shape[0], offset_[0]:offset_[0] + m_.shape[1]] = \
                        torch.from_numpy(m_)
                elif stepi in bboxes:
                    gt_segm = tracking6d.tracker.process_segm(bboxes[stepi])[0].to(tracking6d.device)
                if gt_segm is not None:
                    self.baseline_iou[stepi - 1] = float((segment[0, 0, -1] * gt_segm > 0).sum()) / float(
                        ((segment[0, 0, -1] + gt_segm) > 0).sum() + 0.00001)
                    self.our_iou[stepi - 1] = float((renders[0, -1, 0, 3] * gt_segm > 0).sum()) / float(
                        ((renders[0, -1, 0, 3] + gt_segm) > 0).sum() + 0.00001)
            elif bboxes is not None:
                bbox = tracking6d.config.image_downsample * torch.tensor(
                    [bboxes[stepi] + [0, 0, bboxes[stepi][0], bboxes[stepi][1]]])
                self.baseline_iou[stepi - 1] = bops.box_iou(bbox, torch.tensor([segment2bbox(segment[0, 0, -1])],
                                                                               dtype=torch.float64))
                self.our_iou[stepi - 1] = bops.box_iou(bbox, torch.tensor([segment2bbox(renders[0, -1, 0, 3])],
                                                                          dtype=torch.float64))
            print('Baseline IoU {}, our IoU {}'.format(self.baseline_iou[stepi - 1], self.our_iou[stepi - 1]))
            np.savetxt(os.path.join(tracking6d.write_folder, 'baseline_iou.txt'), self.baseline_iou, fmt='%.10f',
                       delimiter='\n')
            np.savetxt(os.path.join(tracking6d.write_folder, 'iou.txt'), self.our_iou, fmt='%.10f', delimiter='\n')
            np.savetxt(os.path.join(tracking6d.write_folder, 'losses.txt'), our_losses, fmt='%.10f', delimiter='\n')
            self.all_input.write(
                (tracking6d.images[0, :, :3].clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1, 0)[:, :,
                 [2, 1, 0], -1] * 255).astype(np.uint8))
            self.all_segm.write(((tracking6d.images[0, :, :3] * tracking6d.segments[0, :, 1:2]).clamp(min=0,
                                                                                                      max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))
            self.all_proj.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1,
                                                                                                           0)[:, :,
                                 [2, 1, 0], -1] * 255).astype(np.uint8))
            if silh_losses[-1] > 0.3:
                renders[0, -1, 0, 3] = segment[0, 0, -1]
                renders[0, -1, 0, :3] = tracking6d.images[0, -1, :3] * segment[0, 0, -1]
            self.all_proj_filtered.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))

import copy
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import imageio
import kaolin
import numpy as np
import torch
import torchvision.ops.boxes as bops
from torch import nn
from torchvision import transforms
from torchvision.utils import save_image

from GMA.core.utils import flow_viz
from OSTrack.S2DNet.s2dnet import S2DNet
from helpers.torch_helpers import write_renders
from main_settings import g_ext_folder
from models.encoder import Encoder, qmult, qnorm
from models.initial_mesh import generate_face_features
from models.kaolin_wrapper import load_obj, write_obj_mesh
from models.loss import FMOLoss
from models.rendering import RenderingKaolin
from segmentations import PrecomputedTracker, CSRTrack, OSTracker, MyTracker, get_bbox, create_mask_from_string
from utils import segment2bbox, write_video, euler_from_quaternion
from flow import get_flow_from_images, visualize_flow_with_images, load_image, get_flow_from_images_raft
from flow_raft import get_flow_model
from cfg import FLOW_OUT_DEFAULT_DIR

BREAK_AFTER_ITERS_WITH_NO_CHANGE = 10

ALLOW_BREAK_AFTER = 50

TRAINING_PRINT_STATUS_FREQUENCY = 20

SILHOUETTE_LOSS_THRESHOLD = 0.3


@dataclass
class TrackerConfig:

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    # General settings
    tracker_type: str = 'ostrack'
    features: str = 'deep'
    verbose: bool = True
    write_results: bool = True
    write_intermediate: bool = True

    # Frame and keyframe settings
    input_frames: int = 0
    max_keyframes: int = 0
    keyframes: int = None
    fmo_steps: int = 1

    # Shape settings
    shapes: List = list
    init_shape: str = 'sphere'
    predict_vertices: bool = None

    # Mesh settings
    mesh_size: int = None
    mesh_normalize: bool = None
    texture_size: int = None
    use_lights: bool = None

    # Camera settings
    camera_distance: float = None
    max_width: int = 1024
    image_downsample: float = 1.0
    grabcut: bool = False

    # Tracking settings
    tran_init: float = None
    rot_init: List[float] = None
    inc_step: float = None
    learning_rate: float = None
    iterations: int = None
    stop_value: float = None
    rgb_iters: int = None
    project_coin: bool = None
    connect_frames: bool = None
    accumulate: bool = None
    weight_by_gradient: bool = None
    mot_opt_all: bool = None
    motion_only_last: bool = None

    # Loss function coefficients
    loss_laplacian_weight: float = 0.0
    loss_tv_weight: float = 1.0
    loss_iou_weight: float = 0.0
    loss_dist_weight: float = 0.0
    loss_qt_weight: float = 0.0
    loss_rgb_weight: float = 0.0
    loss_flow_weight: float = 1.0

    # Additional settings
    sigmainv: float = None
    factor: float = None
    mask_iou_th: float = None
    erode_renderer_mask: int = None
    rotation_divide: int = None
    sequence: str = None


def visualize_flow(flow_video_up, image, image_new, image_prev, segment, stepi):
    """
    Visualize optical flow between two images and save the results as image files.

    Args:
        flow_video_up (torch.Tensor): Upsampled optical flow tensor.
        image (torch.Tensor): Original image tensor.
        image_new (torch.Tensor): New (second) image tensor.
        image_prev (torch.Tensor): Previous (first) image tensor.
        segment (torch.Tensor): Segmentation mask tensor.
        stepi (int): Index of the current step in the frame sequence.

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
    image_pure_flow = transform(flow_image)
    image_pure_flow_small = transform(flow_image_small)
    image_pure_flow_segmented = transform(flow_image_segmented)
    image_new_pil = transform(image_new[0] / 255.0)
    image_old_pil = transform(image_prev[0] / 255.0)

    # Define output file paths
    prev_image_path = FLOW_OUT_DEFAULT_DIR / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_1.png')
    new_image_path = FLOW_OUT_DEFAULT_DIR / Path('gt_img_' + str(stepi) + '_' + str(stepi + 1) + '_2.png')
    pure_flow_path = FLOW_OUT_DEFAULT_DIR / Path('pure_flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_small_path = FLOW_OUT_DEFAULT_DIR / Path('flow_small_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_segm_path = FLOW_OUT_DEFAULT_DIR / Path('flow_segmented_' + str(stepi) + '_' + str(stepi + 1) + '.png')
    flow_image_path = FLOW_OUT_DEFAULT_DIR / Path('flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')

    # Save the images to disk
    imageio.imwrite(pure_flow_path, image_pure_flow)
    imageio.imwrite(flow_small_path, image_pure_flow_small)
    imageio.imwrite(flow_segm_path, image_pure_flow_segmented)
    imageio.imwrite(new_image_path, image_new_pil)
    imageio.imwrite(prev_image_path, image_old_pil)
    imageio.imwrite(flow_image_path, flow_illustration)


class Tracking6D:
    def __init__(self, config, device, write_folder, file0, bbox0, init_mask=None):
        self.write_folder = write_folder

        self.config = TrackerConfig(**config)
        self.config.fmo_steps = 1
        self.device = device
        self.model_flow = get_flow_model()

        torch.backends.cudnn.benchmark = True
        if type(bbox0) is dict:
            self.tracker = PrecomputedTracker(self.config.image_downsample,
                                              self.config.max_width, bbox0,
                                              self.config.grabcut)
        else:
            if self.config.tracker_type == 'csrt':
                self.tracker = CSRTrack(self.config.image_downsample, self.config.max_width,
                                        self.config.grabcut)
            elif self.config.tracker_type == 'ostrack':
                self.tracker = OSTracker(self.config.image_downsample, self.config.max_width,
                                         self.config.grabcut)
            else:  # d3s
                self.tracker = MyTracker(self.config.image_downsample, self.config.max_width,
                                         self.config.grabcut)
        if self.config.features == 'deep':
            self.net = S2DNet(device=device, checkpoint_path=g_ext_folder).to(device)
            self.feat = lambda x: self.net(x[0])[0][None][:, :, :64]
            self.feat_rgb = lambda x: x
        else:
            self.feat = lambda x: x
        self.images, self.segments, self.config.image_downsample = self.tracker.init_bbox(file0, bbox0, init_mask)
        self.images, self.segments = self.images[None].to(self.device), self.segments[None].to(self.device)
        self.images_high_resolution = load_image(file0)

        self.images_high_resolution: torch.Tensor = self.images_high_resolution[None].to(self.device)
        self.flows: torch.Tensor = torch.zeros(self.images.shape, dtype=self.images.dtype)
        self.flows_high_resolution: torch.Tensor = torch.zeros(self.images_high_resolution.shape,
                                                               dtype=self.images_high_resolution.dtype)
        self.images_feat = self.feat(self.images).detach()

        shape = self.segments.shape
        prot = self.config.shapes[0]
        if self.config.init_shape:
            mesh = load_obj(self.config.init_shape)
            ivertices = mesh.vertices.numpy()
            ivertices = ivertices - ivertices.mean(0)
            ivertices = ivertices / ivertices.max()
            faces = mesh.faces.numpy().copy()
            iface_features = generate_face_features(ivertices, faces)
        else:
            mesh = load_obj(os.path.join('./prototypes', prot + '.obj'))
            ivertices = mesh.vertices.numpy()
            faces = mesh.faces.numpy().copy()
            iface_features = mesh.uvs[mesh.face_uvs_idx].numpy()

        self.faces = faces
        self.rendering = RenderingKaolin(self.config, self.faces, shape[-1], shape[-2]).to(self.device)
        self.encoder = Encoder(self.config, ivertices, faces, iface_features, shape[-1], shape[-2],
                               self.images_feat.shape[2]).to(self.device)
        all_parameters = list(self.encoder.parameters())
        self.optimizer = torch.optim.Adam(all_parameters, lr=self.config.learning_rate)
        self.encoder.train()
        self.loss_function = FMOLoss(self.config, ivertices, faces).to(self.device)
        if self.config.features == 'deep':
            config = copy.deepcopy(self.config)
            config.features = 'rgb'
            self.rgb_encoder = Encoder(config, ivertices, faces, iface_features, shape[-1], shape[-2], 3).to(
                self.device)
            rgb_parameters = list(self.rgb_encoder.parameters())[-1:]
            self.rgb_optimizer = torch.optim.Adam(rgb_parameters, lr=self.config.learning_rate)
            self.rgb_encoder.train()
            config.loss_laplacian_weight = 0
            config.loss_tv_weight = 1.0
            config.loss_iou_weight = 0
            config.loss_dist_weight = 0
            config.loss_qt_weight = 0
            config.loss_flow_weight = 1.0
            self.rgb_loss_function = FMOLoss(config, ivertices, faces).to(self.device)
        if self.config.verbose:
            print('Total params {}'.format(sum(p.numel() for p in self.encoder.parameters())))
        self.best_model = {"value": 100,
                           "face_features": self.encoder.face_features.detach().clone(),
                           "faces": faces,
                           "encoder": None}
        self.keyframes = [0]

    def run_tracking(self, files, bboxes):
        # We canonically adapt the bboxes so that their keys are their order number, ordered from 1
        if type(bboxes) is dict:
            sorted_bb_keys = sorted(list(bboxes.keys()))
            bboxes = {i: bboxes[sorted_bb_keys[i]] for i, key in zip(range(len(bboxes)), sorted_bb_keys)}

        all_input = cv2.VideoWriter(os.path.join(self.write_folder, 'all_input.avi'), cv2.VideoWriter_fourcc(*"MJPG"),
                                    10, (self.images.shape[4], self.images.shape[3]), True)
        all_segm = cv2.VideoWriter(os.path.join(self.write_folder, 'all_segm.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                   (self.images.shape[4], self.images.shape[3]), True)
        all_proj = cv2.VideoWriter(os.path.join(self.write_folder, 'all_proj.avi'), cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                   (self.images.shape[4], self.images.shape[3]), True)
        all_proj_filtered = cv2.VideoWriter(os.path.join(self.write_folder, 'all_proj_filtered.avi'),
                                            cv2.VideoWriter_fourcc(*"MJPG"), 10,
                                            (self.images.shape[4], self.images.shape[3]), True)
        baseline_iou = -np.ones((files.shape[0] - 1, 1))
        our_iou = -np.ones((files.shape[0] - 1, 1))
        our_losses = -np.ones((files.shape[0] - 1, 1))
        self.config.loss_rgb_weight = 0
        removed_count = 0

        b0 = None
        for stepi in range(1, self.config.input_frames):
            image_raw, segment = self.tracker.next(files[stepi])

            image, segment = image_raw[None].to(self.device), segment[None].to(self.device)
            if b0 is not None:
                segment_clean = segment * 0
                segment_clean[:, :, :, b0[0]:b0[1], b0[2]:b0[3]] = segment[:, :, :, b0[0]:b0[1], b0[2]:b0[3]]
                segment_clean[:, :, 0] = segment[:, :, 0]
                segment = segment_clean
            self.images = torch.cat((self.images, image), 1)

            image_prev = (self.images[:, -2, :, :, :]).float() * 255
            image_new = (self.images[:, -1, :, :, :]).float() * 255

            # This gives the deep features of the image
            image_feat = self.feat(image).detach()

            self.images_feat = torch.cat((self.images_feat, image_feat), 1)

            self.segments = torch.cat((self.segments, segment), 1)
            self.keyframes.append(stepi)
            start = time.time()
            b0 = get_bbox(self.segments)

            with torch.no_grad():
                flow_video_low, flow_video_up = get_flow_from_images(image_prev, image_new, self.model_flow)
                flow_video_up = flow_video_up
                flow_video_up_np = flow_video_up[0].detach().cpu().permute(1, 2, 0).numpy()
                observed_flow = flow_video_up[..., b0[0]:b0[1], b0[2]:b0[3]].permute(0, 2, 3, 1)

            # Visualize flow we get from the video
            visualize_flow(flow_video_up_np, image, image_new, image_prev, segment, stepi)

            self.rendering = RenderingKaolin(self.config, self.faces, b0[3] - b0[2], b0[1] - b0[0]).to(self.device)

            self.encoder.offsets[:, :, stepi, :3] = (
                    self.encoder.used_tran[:, :, stepi - 1] + self.encoder.offsets[:, :, stepi - 1, :3])
            self.encoder.offsets[:, 0, stepi, 3:] = qmult(qnorm(self.encoder.used_quat[:, stepi - 1]),
                                                          qnorm(self.encoder.offsets[:, 0, stepi - 1, 3:]))

            theoretical_flow = self.apply(self.images_feat[:, :, :, b0[0]:b0[1], b0[2]:b0[3]],
                                          self.segments[:, :, :, b0[0]:b0[1], b0[2]:b0[3]], observed_flow,
                                          self.keyframes, step_i=stepi)

            silh_losses = np.array(self.best_model["losses"]["silh"])

            our_losses[stepi - 1] = silh_losses[-1]
            print('Elapsed time in seconds: ', time.time() - start, "Frame ", stepi, "out of",
                  self.config.input_frames)
            if silh_losses[-1] < 0.8:
                self.encoder.used_tran[:, :, stepi] = self.encoder.translation[:, :, stepi].detach()
                self.encoder.used_quat[:, stepi] = self.encoder.quaternion[:, stepi].detach()

            if self.config.write_results:
                qdiff, renders, tdiff, texture_maps, vertices = self.write_results(all_input, all_proj,
                                                                                   all_proj_filtered, all_segm, b0,
                                                                                   baseline_iou, bboxes, our_iou,
                                                                                   our_losses, segment, silh_losses,
                                                                                   stepi, observed_flow)
            keep_keyframes = (silh_losses <= 0.8)
            if silh_losses[-1] > SILHOUETTE_LOSS_THRESHOLD:
                keep_keyframes[-1] = False
            keep_keyframes[np.argmin(silh_losses)] = True
            self.keyframes = (np.array(self.keyframes)[keep_keyframes]).tolist()
            self.images = self.images[:, keep_keyframes]
            self.images_feat = self.images_feat[:, keep_keyframes]
            self.segments = self.segments[:, keep_keyframes]

            self.update_keyframes(image, image_feat, qdiff, removed_count, renders, segment, stepi, tdiff, texture_maps,
                                  vertices, observed_flow, theoretical_flow)
        all_input.release()
        all_segm.release()
        all_proj.release()
        all_proj_filtered.release()

        return self.best_model

    def update_keyframes(self, image, image_feat, qdiff, removed_count, renders, segment, stepi, tdiff, texture_maps,
                         vertices, observed_flow, flow_from_tracking):
        """
        Updates the keyframes, images, image features, and segments based on the current step and loss values.

        Parameters:
        image (Tensor): The current image tensor.
        image_feat (Tensor): The current image features tensor.
        qdiff (List[float]): The quaternion difference between consecutive frames.
        removed_count (int): The number of removed keyframes.
        renders (Tensor): The renders tensor.
        segment (Tensor): The current segment tensor.
        stepi (int): The current step iteration.
        tdiff (List[float]): The translation difference between consecutive frames.
        texture_maps (List[Tensor]): The list of texture maps for the objects.
        vertices (List[Tensor]): The list of vertices for the objects.

        Returns:
        None
        :param observed_flow:
        :param flow_from_tracking:
        """
        if len(self.keyframes) >= 3:
            l1, _, _ = self.loss_function(renders[:, -1:], renders[:, -2:-1, 0][:, :, [-1, -1]],
                                          renders[:, -2:-1, 0, :3], vertices, texture_maps, tdiff, qdiff,
                                          observed_flow, flow_from_tracking)
            l2, _, _ = self.loss_function(renders[:, -2:-1], renders[:, -3:-2, 0][:, :, [-1, -1]],
                                          renders[:, -3:-2, 0, :3], vertices, texture_maps, tdiff, qdiff,
                                          observed_flow, flow_from_tracking)
            if l1["silh"][-1] < 0.7 and l2["silh"][-1] < 0.7 and removed_count < 30:
                removed_count += 1
                # self.keyframes = self.keyframes[:-2] + [stepi]
                self.keyframes = self.keyframes[-2:-1] + [stepi]
                self.images = torch.cat((self.images[:, :-2], image), 1)
                self.images_feat = torch.cat((self.images_feat[:, :-2], image_feat), 1)
                self.segments = torch.cat((self.segments[:, :-2], segment), 1)
        # if len(self.keyframes) > self.config.max_keyframes:
        #     self.keyframes = self.keyframes[-self.config.max_keyframes:]
        #     self.images = self.images[:, -self.config.max_keyframes:]
        #     self.images_feat = self.images_feat[:, -self.config.max_keyframes:]
        #     self.segments = self.segments[:, -self.config.max_keyframes:]

    def write_results(self, all_input, all_proj, all_proj_filtered, all_segm, b0, baseline_iou, bboxes, our_iou,
                      our_losses, segment, silh_losses, stepi, observed_flow):
        if self.config.features == 'deep':
            tex = nn.Sigmoid()(self.rgb_encoder.texture_map)

        with torch.no_grad():
            translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.encoder(self.keyframes)

            last_quaternion = quaternion[0, -1]
            first_quaternion = quaternion[0, 0]
            euler_angles_last = euler_from_quaternion(last_quaternion[0], last_quaternion[1], last_quaternion[1],
                                                      last_quaternion[2])
            euler_angles_first = euler_from_quaternion(first_quaternion[0], first_quaternion[1],
                                                       first_quaternion[1], first_quaternion[2])
            print("Last estimated rotation:", [(float(euler_angles_last[i]) * 180 / math.pi -
                                                float(euler_angles_first[i]) * 180 / math.pi) % 360
                                               for i in range(len(euler_angles_last))])

            if self.config.features == 'rgb':
                tex = texture_maps
            feat_renders_crop = self.get_rendered_image_features(lights, quaternion, texture_maps, translation,
                                                                 vertices)

            renders, renders_crop = self.get_rendered_image(b0, lights, quaternion, tex, translation, vertices)
            # breakpoint()
            write_renders(feat_renders_crop, self.write_folder, self.config.max_keyframes + 1, ids=0)
            write_renders(renders_crop, self.write_folder, self.config.max_keyframes + 1, ids=1)
            write_renders(torch.cat(
                (self.images[:, :, None, :, b0[0]:b0[1], b0[2]:b0[3]], feat_renders_crop[:, :, :, -1:]), 3),
                self.write_folder, self.config.max_keyframes + 1, ids=2)
            write_obj_mesh(vertices[0].cpu().numpy(), self.best_model["faces"],
                           self.encoder.face_features[0].cpu().numpy(),
                           os.path.join(self.write_folder, 'mesh.obj'))
            save_image(texture_maps[:, :3], os.path.join(self.write_folder, 'tex_deep.png'))
            save_image(tex, os.path.join(self.write_folder, 'tex.png'))
            write_video(renders[0, :, 0, :3].detach().cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(self.write_folder, 'im_recon.avi'), fps=6)
            write_video(self.images[0, :, :3].cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(self.write_folder, 'input.avi'), fps=6)
            write_video((self.images[0, :, :3] * self.segments[0, :, 1:2]).cpu().numpy().transpose(2, 3, 1, 0),
                        os.path.join(self.write_folder, 'segments.avi'), fps=6)
            for tmpi in range(renders.shape[1]):
                img = self.images[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]]
                seg = self.segments[0, :, 1:2][tmpi, :, b0[0]:b0[1], b0[2]:b0[3]].clone()
                save_image(seg, os.path.join(self.write_folder, 'imgs', 's{}.png'.format(tmpi)))
                seg[seg == 0] = 0.35
                save_image(img, os.path.join(self.write_folder, 'imgs', 'i{}.png'.format(tmpi)))
                save_image(self.images_feat[0, tmpi, :3, b0[0]:b0[1], b0[2]:b0[3]],
                           os.path.join(self.write_folder, 'imgs', 'if{}.png'.format(tmpi)))
                save_image(torch.cat((img, seg), 0),
                           os.path.join(self.write_folder, 'imgs', 'is{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, [3, 3, 3]],
                           os.path.join(self.write_folder, 'imgs', 'm{}.png'.format(tmpi)))
                save_image(renders_crop[0, tmpi, 0, :],
                           os.path.join(self.write_folder, 'imgs', 'r{}.png'.format(tmpi)))
                save_image(feat_renders_crop[0, tmpi, 0, :],
                           os.path.join(self.write_folder, 'imgs', 'f{}.png'.format(tmpi)))
            if type(bboxes) is dict or (bboxes[stepi][0] == 'm'):
                gt_segm = None
                if (not type(bboxes) is dict) and bboxes[stepi][0] == 'm':
                    m_, offset_ = create_mask_from_string(bboxes[stepi][1:].split(','))
                    gt_segm = segment[0, 0, -1] * 0
                    gt_segm[offset_[1]:offset_[1] + m_.shape[0],
                    offset_[0]:offset_[0] + m_.shape[1]] = torch.from_numpy(m_)
                elif stepi in bboxes:
                    gt_segm = self.tracker.process_segm(bboxes[stepi])[0].to(self.device)
                if gt_segm is not None:
                    baseline_iou[stepi - 1] = float((segment[0, 0, -1] * gt_segm > 0).sum()) / float(
                        ((segment[0, 0, -1] + gt_segm) > 0).sum() + 0.00001)
                    our_iou[stepi - 1] = float((renders[0, -1, 0, 3] * gt_segm > 0).sum()) / float(
                        ((renders[0, -1, 0, 3] + gt_segm) > 0).sum() + 0.00001)
            elif bboxes is not None:
                bbox = self.config.image_downsample * torch.tensor(
                    [bboxes[stepi] + [0, 0, bboxes[stepi][0], bboxes[stepi][1]]])
                baseline_iou[stepi - 1] = bops.box_iou(bbox, torch.tensor([segment2bbox(segment[0, 0, -1])],
                                                                          dtype=torch.float64))
                our_iou[stepi - 1] = bops.box_iou(bbox, torch.tensor([segment2bbox(renders[0, -1, 0, 3])],
                                                                     dtype=torch.float64))
            print('Baseline IoU {}, our IoU {}'.format(baseline_iou[stepi - 1], our_iou[stepi - 1]))
            np.savetxt(os.path.join(self.write_folder, 'baseline_iou.txt'), baseline_iou, fmt='%.10f',
                       delimiter='\n')
            np.savetxt(os.path.join(self.write_folder, 'iou.txt'), our_iou, fmt='%.10f', delimiter='\n')
            np.savetxt(os.path.join(self.write_folder, 'losses.txt'), our_losses, fmt='%.10f', delimiter='\n')
            all_input.write((self.images[0, :, :3].clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1, 0)[:, :,
                             [2, 1, 0], -1] * 255).astype(np.uint8))
            all_segm.write(((self.images[0, :, :3] * self.segments[0, :, 1:2]).clamp(min=0,
                                                                                     max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))
            all_proj.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(2, 3, 1,
                                                                                                      0)[:, :,
                            [2, 1, 0], -1] * 255).astype(np.uint8))
            if silh_losses[-1] > 0.3:
                renders[0, -1, 0, 3] = segment[0, 0, -1]
                renders[0, -1, 0, :3] = self.images[0, -1, :3] * segment[0, 0, -1]
            all_proj_filtered.write((renders[0, :, 0, :3].detach().clamp(min=0, max=1).cpu().numpy().transpose(
                2, 3, 1, 0)[:, :, [2, 1, 0], -1] * 255).astype(np.uint8))
        return qdiff, renders, tdiff, texture_maps, vertices

    def get_rendered_image_features(self, lights, quaternion, texture_maps, translation, vertices):
        feat_renders_crop, theoretical_flow, texture_flow = self.rendering(translation, quaternion, vertices,
                                                                           self.encoder.face_features,
                                                                           texture_maps, lights, True)
        feat_renders_crop = feat_renders_crop[:, :, :, :-1]
        feat_renders_crop = torch.cat((feat_renders_crop[:, :, :, :3], feat_renders_crop[:, :, :, -1:]), 3)
        return feat_renders_crop

    def get_rendered_image(self, bounding_box, lights, quaternion, texture, translation, vertices):
        """

        :param bounding_box: [y0, y1, x0, x1] location of the bounding box
        :param lights:
        :param quaternion: Rotation of the mesh
        :param texture: Mesh texture
        :param translation: Mesh translation
        :param vertices: Mesh vertices
        :return:
            renders_cropped: [4, 272, 224] RGB-A rendering of the image
            renders: [4, 300, 225] RGB-A rendering of the image, where the rendering is put inside the bounding box
        """
        renders_crop, theoretical_flow, texture_flow = self.rendering(translation, quaternion, vertices,
                                                                      self.encoder.face_features, texture, lights)
        renders_crop = torch.cat((renders_crop[:, :, :, :3], renders_crop[:, :, :, -1:]), 3)
        renders = self.write_image_into_bbox(bounding_box, renders_crop)
        return renders, renders_crop

    def write_image_into_bbox(self, bounding_box, renders_crop):
        """

        :param bounding_box: List specifying the bounding box starts, resp. end
        :param renders_crop: Image of shape ..., C, H, W
        :return:
        """
        renders = torch.zeros(renders_crop.shape[:-2] + self.images_feat.shape[-2:]).to(self.device)
        renders[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]] = renders_crop
        return renders

    def apply(self, input_batch, segments, observed_flow, opt_frames=None, step_i=0):
        if self.config.write_results:
            save_image(input_batch[0, :, :3], os.path.join(self.write_folder, 'im.png'),
                       nrow=self.config.max_keyframes + 1)
            save_image(torch.cat((input_batch[0, :, :3], segments[0, :, [1]]), 1),
                       os.path.join(self.write_folder, 'segments.png'), nrow=self.config.max_keyframes + 1)
            if self.config.weight_by_gradient:
                save_image(torch.cat((segments[0, :, [0, 0, 0]], 0 * input_batch[0, :, :1] + 1), 1),
                           os.path.join(self.write_folder, 'weights.png'))

        self.best_model["value"] = 100
        self.best_model["losses"] = None
        iters_without_change = 0

        for epoch in range(self.config.iterations):
            translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.encoder(
                opt_frames)

            renders, theoretical_flow, texture_flow = self.rendering(translation, quaternion, vertices,
                                                                     self.encoder.face_features,
                                                                     texture_maps, lights)

            losses_all, losses, jloss = self.loss_function(renders, segments, input_batch, vertices, texture_maps,
                                                           tdiff, qdiff, observed_flow, theoretical_flow)

            if "model" in losses:
                model_loss = losses["model"].mean().item()
            else:
                model_loss = losses["silh"].mean().item()
            if self.config.verbose and epoch % TRAINING_PRINT_STATUS_FREQUENCY == 0:
                print("Epoch {:4d}".format(epoch + 1), end=" ")
                for ls in losses:
                    print(", {} {:.3f}".format(ls, losses[ls].mean().item()), end=" ")
                print("; joint {:.3f}".format(jloss.item()))

            if model_loss < self.best_model["value"]:
                iters_without_change = 0
                self.best_model["value"] = model_loss
                self.best_model["losses"] = losses_all
                self.best_model["encoder"] = copy.deepcopy(self.encoder.state_dict())
                if self.config.write_intermediate:
                    write_renders(torch.cat((renders[:, :, :, :3], renders[:, :, :, -1:]), 3), self.write_folder,
                                  self.config.max_keyframes + 1)
            else:
                iters_without_change += 1

            if self.config.loss_rgb_weight == 0:
                if epoch > 100 or model_loss < 0.1:
                    self.config.loss_rgb_weight = 1.0
                    self.best_model["value"] = 100
            else:
                if epoch > ALLOW_BREAK_AFTER and self.best_model["value"] < self.config.stop_value and \
                        iters_without_change > BREAK_AFTER_ITERS_WITH_NO_CHANGE:
                    break
            if epoch < self.config.iterations - 1:
                jloss = jloss.mean()
                self.optimizer.zero_grad()
                jloss.backward()
                self.optimizer.step()

        self.visualize_theoretical_flow(texture_flow, theoretical_flow, opt_frames, step_i)

        self.encoder.load_state_dict(self.best_model["encoder"])

        return theoretical_flow

    def visualize_theoretical_flow(self, texture_flow, theoretical_flow, opt_frames, stepi):
        b0 = get_bbox(self.segments)
        opt_frames_prime = [max(opt_frames) - 1, max(opt_frames)]
        translation_prime, quaternion_prime, vertices_prime, \
            texture_maps_prime, lights_prime, tdiff_prime, qdiff_prime = self.encoder(opt_frames_prime)
        tex_rgb = nn.Sigmoid()(self.rgb_encoder.texture_map)
        # Render the image given the estimated shape of it
        rendered_keyframe_images, _ = self.get_rendered_image(b0, lights_prime, quaternion_prime, tex_rgb,
                                                              translation_prime, vertices_prime)
        # The rendered images return renders of all keyframes, the previous and the current image
        current_rendered_image_rgba = rendered_keyframe_images[0, -1, ...]
        previous_rendered_image_rgba = rendered_keyframe_images[0, -2, ...]
        current_rendered_image_rgb = current_rendered_image_rgba[:, :3, ...]
        previous_rendered_image_rgb = previous_rendered_image_rgba[:, :3, ...]
        theoretical_flow_path = FLOW_OUT_DEFAULT_DIR / \
                                Path('theoretical_flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')
        texture_flow_path = FLOW_OUT_DEFAULT_DIR / \
                            Path('texture_flow_' + str(stepi) + '_' + str(stepi + 1) + '.png')
        rendering_1_path = FLOW_OUT_DEFAULT_DIR / \
                           Path('rendering_' + str(stepi) + '_' + str(stepi + 1) + '_1.png')
        rendering_2_path = FLOW_OUT_DEFAULT_DIR / \
                           Path('rendering_' + str(stepi) + '_' + str(stepi + 1) + '_2.png')

        prev_img_np = (previous_rendered_image_rgb[0] * 255).detach().cpu().numpy().transpose(1, 2, 0).astype(
            'uint8')
        new_img_np = (current_rendered_image_rgb[0] * 255).detach().cpu().numpy().transpose(1, 2, 0).astype(
            'uint8')
        imageio.imwrite(rendering_1_path, prev_img_np)
        imageio.imwrite(rendering_2_path, new_img_np)

        flow_render_up_ = theoretical_flow.detach().cpu()[0].permute(2, 0, 1)
        theoretical_flow_up_ = self.write_image_into_bbox(b0, flow_render_up_)
        # Convert the resized tensor back to a NumPy array and remove the batch dimension
        theoretical_flow_up_ = theoretical_flow_up_.detach().cpu().numpy()  # Remove batch dimension
        # Select the first channel
        theoretical_flow_up_ = theoretical_flow_up_.transpose(1, 2, 0)
        flow_illustration = visualize_flow_with_images(previous_rendered_image_rgb[0],
                                                       current_rendered_image_rgb[0], theoretical_flow_up_)
        texture_flow_up_ = self.write_image_into_bbox(b0, texture_flow[0].permute(2, 0, 1))
        texture_flow_up_ = texture_flow_up_.detach().cpu().numpy()
        texture_flow_up_ = texture_flow_up_.transpose(1, 2, 0)
        texture_flow_illustration = visualize_flow_with_images(previous_rendered_image_rgb[0],
                                                               current_rendered_image_rgb[0], texture_flow_up_)
        imageio.imwrite(theoretical_flow_path, flow_illustration)
        imageio.imwrite(texture_flow_path, texture_flow_illustration)

    def rgb_apply(self, input_batch, segments, observed_flow, opt_frames):
        self.best_model["value"] = 100
        model_state = self.rgb_encoder.state_dict()
        pretrained_dict = self.best_model["encoder"]
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k != "texture_map"}
        model_state.update(pretrained_dict)
        self.rgb_encoder.load_state_dict(model_state)
        for epoch in range(self.config.rgb_iters):
            translation, quaternion, vertices, texture_maps, lights, tdiff, qdiff = self.rgb_encoder(opt_frames)
            renders, theoretical_flow, texture_flow = self.rendering(translation, quaternion, vertices,
                                                                     self.encoder.face_features, texture_maps, lights)
            losses_all, losses, jloss = self.rgb_loss_function(renders, segments, input_batch, vertices, texture_maps,
                                                               tdiff, qdiff, observed_flow, theoretical_flow)
            if self.best_model["value"] < 0.1 and iters_without_change > 10:
                break
            if epoch < self.config.iterations - 1:
                jloss = jloss.mean()
                self.rgb_optimizer.zero_grad()
                jloss.backward()
                self.rgb_optimizer.step()

    def apply_incremental(self, input_batch, segments):
        # raise AssertionError("This method should never have been called")
        input_batch, segments = input_batch[None].to(self.device), segments[None].to(self.device)
        for stepi in range(int(self.config.input_frames / self.config.inc_step)):
            if self.config.accumulate:
                st = 0
            else:
                st = stepi * self.config.inc_step
            en = (stepi + 1) * self.config.inc_step
            opt_frames = self.keyframes + list(range(st, en))

            self.apply(input_batch[:, opt_frames], segments[:, opt_frames][:, :, 0, :2], None, opt_frames)
            self.encoder = self.best_model["encoder"]
            all_parameters = list(self.encoder.parameters())
            self.optimizer = torch.optim.Adam(all_parameters, lr=self.config.learning_rate)
            self.encoder.train()
            if self.config.write_results:
                with torch.no_grad():
                    translation, quaternion, vertices, texture_maps, lights, _, _ = self.encoder(list(range(0, en)))
                    renders, theoretical_flow = self.rendering(translation, quaternion, vertices,
                                                               self.encoder.face_features, texture_maps,
                                                               lights)
                    write_renders(renders, self.write_folder, self.config.inc_step, en)
                    write_obj_mesh(vertices[0].cpu().numpy(), self.best_model["faces"],
                                   self.encoder.face_features[0].cpu().numpy(),
                                   os.path.join(self.write_folder, 'mesh.obj'))
                    save_image(texture_maps, os.path.join(self.write_folder, 'tex.png'))
                    write_video(renders[0, :, 0, :3].detach().cpu().numpy().transpose(2, 3, 1, 0),
                                os.path.join(self.write_folder, 'im_recon' + '{}.avi'.format(en)), fps=6)
                    write_video(input_batch[0, 0:en, :3].cpu().numpy().transpose(2, 3, 1, 0),
                                os.path.join(self.write_folder, 'input.avi'), fps=6)
                    write_video((input_batch[0, 0:en, :3] * segments[0, 0:en, 1:2]).cpu().numpy().transpose(2, 3, 1, 0),
                                os.path.join(self.write_folder, 'segments.avi'), fps=6)

            if self.config.keyframes and not self.config.accumulate:
                self.keyframes.append(st)

        return self.best_model

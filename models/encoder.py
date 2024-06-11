from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
from kornia.geometry import normalize_quaternion
from kornia.geometry.conversions import axis_angle_to_quaternion
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.liegroup import Se3, So3
from pytorch3d.transforms import quaternion_multiply

from utils import mesh_normalize, comp_tran_diff
from auxiliary_scripts.math_utils import qmult, qdist

EncoderResult = namedtuple('EncoderResult', ['translations', 'quaternions', 'vertices', 'texture_maps',
                                             'lights', 'translation_difference', 'quaternion_difference'])


class Encoder(nn.Module):
    def __init__(self, config, ivertices, face_features, width, height, n_feat, texture_maps_init=None):
        super(Encoder, self).__init__()
        self.config = config

        # Translation initialization
        translation_init = torch.zeros(1, 1, config.input_frames, 3)
        translation_init[:, :, 0, 2] = self.config.tran_init

        # Quaternion initialization
        qinit = torch.zeros(1, config.input_frames, 4)
        qinit[:, :, 0] = 1.0
        init_angle = torch.Tensor(self.config.rot_init)
        qinit[:, 0, :] = axis_angle_to_quaternion(init_angle)

        init_axis_angle = torch.zeros(1, config.input_frames, 3)
        init_axis_angle[:, 0] = init_angle

        # Initial rotations and translations
        self.register_buffer('initial_translation', translation_init.clone())
        self.register_buffer('initial_quaternion', qinit.clone())
        self.register_buffer('initial_axis_angle', init_axis_angle.clone())

        # Offsets initialization
        quaternion_offsets = torch.zeros(1, config.input_frames, 4)
        quaternion_offsets[:, :, 0] = 1.0
        self.register_buffer('quaternion_offsets', quaternion_offsets)

        translation_offsets = torch.zeros(1, 1, config.input_frames, 3)
        self.register_buffer('translation_offsets', translation_offsets)

        self.translation = nn.Parameter(torch.zeros(translation_offsets.shape))
        quat = torch.zeros(1, config.input_frames, 4)
        quat[:, :, 0] = 1.0

        self.quaternion_w = nn.Parameter(quat[..., 0, None])
        self.quaternion_x = nn.Parameter(quat[..., 1, None])
        self.quaternion_y = nn.Parameter(quat[..., 2, None])
        self.quaternion_z = nn.Parameter(quat[..., 3, None])

        # Lights initialization
        if self.config.use_lights:
            lights = torch.zeros(1, 3, 9)
            lights[:, :, 0] = 0.5
            self.lights = nn.Parameter(lights)
        else:
            self.lights = None

        # Vertices initialization
        if self.config.optimize_shape:
            self.vertices = nn.Parameter(torch.zeros(1, ivertices.shape[0], 3))

        # Face features and texture map
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        if texture_maps_init is None:
            self.texture_map = nn.Parameter(torch.ones(1, n_feat, self.config.texture_size, self.config.texture_size))
        else:
            self.texture_map = nn.Parameter(texture_maps_init.clone())

        # Normalize and register ivertices
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)

        # Aspect ratio
        self.aspect_ratio = height / width

    def set_grad_mesh(self, req_grad):
        self.texture_map.requires_grad = req_grad
        if self.config.optimize_shape:
            self.vertices.requires_grad = req_grad
        if self.config.use_lights:
            self.lights.requires_grad = req_grad

    def forward(self, opt_frames):

        if self.config.optimize_shape:
            vertices = self.ivertices + self.vertices
            if self.config.mesh_normalize:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:, None, :]  # make center of mass in origin
        else:
            vertices = self.ivertices

        translation = self.get_total_translation_at_frame_vectorized()
        quaternion = self.get_total_rotation_at_frame_vectorized()
        translation[:, :, 0] = translation[:, :, 0].detach()
        quaternion[:, 0] = quaternion[:, 0].detach()

        noopt = list(set(range(translation.shape[2])) - set(opt_frames))

        translation[:, :, noopt] = translation[:, :, noopt].detach()
        quaternion[:, noopt] = quaternion[:, noopt].detach()

        quaternion = quaternion[:, :opt_frames[-1] + 1]
        translation = translation[:, :, :opt_frames[-1] + 1]

        if self.config.features == 'deep':
            texture_map = self.texture_map
        else:
            texture_map = nn.Sigmoid()(self.texture_map)

        # Computes differences of consecutive translations and rotations
        tdiff, qdiff = self.compute_tdiff_qdiff(opt_frames, quaternion[:, -1], quaternion, translation)

        result = EncoderResult(translations=translation,
                               quaternions=quaternion,
                               vertices=vertices,
                               texture_maps=texture_map,
                               lights=self.lights,
                               translation_difference=tdiff,
                               quaternion_difference=qdiff)
        return result

    @staticmethod
    def compute_tdiff_qdiff(opt_frames, quaternion0, quaternion, translation):
        weights = (torch.Tensor(opt_frames) - torch.Tensor(opt_frames[:1] + opt_frames[:-1])).to(translation.device)
        # Temporal distance between consecutive items in opt_frames, i.e. weight grows linearly with distance
        tdiff = weights * comp_tran_diff(translation[0, 0, opt_frames])
        key_dists = []
        for frmi in opt_frames[1:]:
            key_dists.append(qdist(quaternion[:, frmi - 1], quaternion[:, frmi]))
        qdiff = weights * (torch.stack([qdist(quaternion0, quaternion0)] + key_dists, 0).contiguous())
        return tdiff, qdiff

    def get_total_rotation_at_frame_vectorized(self):
        offset_initial_quaternion = quaternion_multiply(normalize_quaternion(self.initial_quaternion),
                                                        normalize_quaternion(self.quaternion_offsets))
        quaternion = torch.cat([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], dim=-1)
        quaternions = quaternion_multiply(offset_initial_quaternion,
                                          normalize_quaternion(quaternion))
        total_rotation_quaternion = normalize_quaternion(quaternions)

        return total_rotation_quaternion

    def get_total_translation_at_frame_vectorized(self):
        # The formula is initial_translation * translation_offsets * translation
        return self.initial_translation + self.translation_offsets + self.translation

    def compute_next_offset(self, stepi):
        self.initial_translation[:, :, stepi] = self.initial_translation[:, :, stepi - 1]
        self.initial_quaternion[:, stepi] = self.initial_quaternion[:, stepi - 1]
        self.initial_axis_angle[:, stepi] = self.initial_axis_angle[:, stepi - 1]

        self.translation_offsets[:, :, stepi] = self.translation_offsets[:, :, stepi - 1] + \
                                                self.translation[:, :, stepi - 1].detach()
        quaternion = torch.cat([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], dim=-1)
        self.quaternion_offsets[:, stepi] = qmult(normalize_quaternion(self.quaternion_offsets[:, stepi - 1]),
                                                  normalize_quaternion(quaternion[:, stepi - 1]).detach())

    def frames_and_flow_frames_inference(self, keyframes, flow_frames) -> Tuple[EncoderResult, EncoderResult]:

        joined_frames = sorted(set(keyframes + flow_frames))
        not_optimized_frames = set(flow_frames) - set(keyframes)
        optimized_frames = list(sorted(set(joined_frames) - not_optimized_frames))

        joined_frames_idx = {frame: idx for idx, frame in enumerate(joined_frames)}

        frames_join_idx = [joined_frames_idx[frame] for frame in keyframes]
        flow_frames_join_idx = [joined_frames_idx[frame] for frame in flow_frames]

        joined_encoder_result: EncoderResult = self.forward(optimized_frames)

        optimized_translations = joined_encoder_result.translations[:, :, joined_frames]
        optimized_quaternions = joined_encoder_result.quaternions[:, joined_frames]

        keyframes_translations = optimized_translations[:, :, frames_join_idx]
        keyframes_quaternions = optimized_quaternions[:, frames_join_idx]
        flow_frames_translations = optimized_translations[:, :, flow_frames_join_idx]
        flow_frames_quaternions = optimized_quaternions[:, flow_frames_join_idx]

        keyframes_tdiff, keyframes_qdiff = self.compute_tdiff_qdiff(keyframes, optimized_quaternions[:, -1],
                                                                    joined_encoder_result.quaternions,
                                                                    joined_encoder_result.translations)
        flow_frames_tdiff, flow_frames_qdiff = self.compute_tdiff_qdiff(flow_frames, optimized_quaternions[:, -1],
                                                                        joined_encoder_result.quaternions,
                                                                        joined_encoder_result.translations)

        encoder_result = EncoderResult(translations=keyframes_translations,
                                       quaternions=keyframes_quaternions,
                                       vertices=joined_encoder_result.vertices,
                                       texture_maps=joined_encoder_result.texture_maps,
                                       lights=joined_encoder_result.lights,
                                       translation_difference=keyframes_tdiff,
                                       quaternion_difference=keyframes_qdiff)

        encoder_result_flow_frames = EncoderResult(translations=flow_frames_translations,
                                                   quaternions=flow_frames_quaternions,
                                                   vertices=joined_encoder_result.vertices,
                                                   texture_maps=joined_encoder_result.texture_maps,
                                                   lights=joined_encoder_result.lights,
                                                   translation_difference=flow_frames_tdiff,
                                                   quaternion_difference=flow_frames_qdiff)

        return encoder_result, encoder_result_flow_frames

from collections import namedtuple
from typing import Tuple

import torch
import torch.nn as nn
from kornia.geometry import normalize_quaternion, So3, Se3
from kornia.geometry.conversions import axis_angle_to_quaternion
from kornia.geometry.quaternion import Quaternion

from utils import mesh_normalize

EncoderResult = namedtuple('EncoderResult', ['translations', 'quaternions', 'vertices', 'texture_maps',
                                             'lights'])


class Encoder(nn.Module):
    def __init__(self, config, ivertices, face_features, width, height, n_feat, texture_maps_init=None):
        super(Encoder, self).__init__()
        self.config = config

        # Translation initialization
        translation_init = torch.Tensor(self.config.tran_init).repeat(config.input_frames, 1)

        # Quaternion initialization
        qinit = Quaternion.from_axis_angle(torch.Tensor(self.config.rot_init).repeat(config.input_frames, 1)).q

        # Initial rotations and translations
        self.register_buffer('initial_translation', translation_init.clone())
        self.register_buffer('initial_quaternion', qinit.clone())

        # Offsets initialization
        quaternion_offsets = Quaternion.identity(config.input_frames)
        self.register_buffer('quaternion_offsets', qinit)

        translation_offsets = torch.zeros(config.input_frames, 3)
        self.register_buffer('translation_offsets', translation_init)

        self.translation = nn.Parameter(torch.zeros_like(translation_offsets))
        quat = Quaternion.identity(config.input_frames)

        self.quaternion = nn.Parameter(quat.q)

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
        translation[0] = translation[0].detach()
        quaternion[0] = quaternion[0].detach()

        noopt = list(set(range(self.config.input_frames)) - set(opt_frames))

        translation[noopt] = translation[noopt].detach()
        quaternion[noopt] = quaternion[noopt].detach()

        quaternion = quaternion[:opt_frames[-1] + 1]
        translation = translation[:opt_frames[-1] + 1]

        if self.config.features == 'deep':
            texture_map = self.texture_map
        else:
            texture_map = nn.Sigmoid()(self.texture_map)

        result = EncoderResult(translations=translation,
                               quaternions=quaternion,
                               vertices=vertices,
                               texture_maps=texture_map,
                               lights=self.lights)
        return result

    def set_encoder_poses(self, rotations: torch.Tensor, translations: torch.Tensor):
        rotation_quaternion = axis_angle_to_quaternion(rotations)

        self.quaternion = torch.nn.Parameter(rotation_quaternion)
        self.translation = torch.nn.Parameter(translations)

    def get_total_rotation_at_frame_vectorized(self):
        q_so3 = So3(Quaternion(self.quaternion))
        q_so3_offset = So3(Quaternion(self.quaternion_offsets))

        offset_initial_quaternion = q_so3_offset * q_so3

        total_rotation_quaternion = normalize_quaternion(offset_initial_quaternion.q.q)

        return total_rotation_quaternion

    def get_se3_at_frame_vectorized(self) -> Se3:

        so3_offset = So3(Quaternion(self.quaternion_offsets))
        so3_optimized = So3(Quaternion(normalize_quaternion(self.quaternion)))

        total_rotation = so3_offset * so3_optimized

        translation_at_frame_vectorized = self.translation_offsets + self.translation

        total_se3 = Se3(total_rotation, translation_at_frame_vectorized)

        return total_se3

    def get_total_translation_at_frame_vectorized(self):
        return self.translation_offsets + self.translation

    def compute_next_offset(self, stepi):

        self.translation_offsets[stepi] = self.translation_offsets[stepi - 1] + self.translation[stepi - 1].detach()

        so3_offset = So3(Quaternion(normalize_quaternion(self.quaternion_offsets[stepi - 1])))
        so3_optim = So3(Quaternion(normalize_quaternion(self.quaternion[stepi - 1]).detach()))
        so3_new_offset = so3_offset * so3_optim

        self.quaternion_offsets[stepi] = so3_new_offset.q.q

    def frames_and_flow_frames_inference(self, keyframes, flow_frames) -> Tuple[EncoderResult, EncoderResult]:

        joined_frames = sorted(set(keyframes + flow_frames))
        not_optimized_frames = set(flow_frames) - set(keyframes)
        optimized_frames = list(sorted(set(joined_frames) - not_optimized_frames))

        joined_frames_idx = {frame: idx for idx, frame in enumerate(joined_frames)}

        frames_join_idx = [joined_frames_idx[frame] for frame in keyframes]
        flow_frames_join_idx = [joined_frames_idx[frame] for frame in flow_frames]

        joined_encoder_result: EncoderResult = self.forward(optimized_frames)

        optimized_translations = joined_encoder_result.translations[joined_frames]
        optimized_quaternions = joined_encoder_result.quaternions[joined_frames]

        keyframes_translations = optimized_translations[frames_join_idx]
        keyframes_quaternions = optimized_quaternions[frames_join_idx]
        flow_frames_translations = optimized_translations[flow_frames_join_idx]
        flow_frames_quaternions = optimized_quaternions[flow_frames_join_idx]

        encoder_result = EncoderResult(translations=keyframes_translations,
                                       quaternions=keyframes_quaternions,
                                       vertices=joined_encoder_result.vertices,
                                       texture_maps=joined_encoder_result.texture_maps,
                                       lights=joined_encoder_result.lights)

        encoder_result_flow_frames = EncoderResult(translations=flow_frames_translations,
                                                   quaternions=flow_frames_quaternions,
                                                   vertices=joined_encoder_result.vertices,
                                                   texture_maps=joined_encoder_result.texture_maps,
                                                   lights=joined_encoder_result.lights)

        return encoder_result, encoder_result_flow_frames

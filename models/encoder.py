from collections import namedtuple

import math
import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_quaternion, QuaternionCoeffOrder, quaternion_to_angle_axis
from pytorch3d.transforms import quaternion_multiply

from utils import mesh_normalize, comp_tran_diff, qnorm, qmult, qdist, qnorm_vectorized

EncoderResult = namedtuple('EncoderResult', ['translations', 'quaternions', 'vertices', 'texture_maps',
                                             'lights', 'translation_difference', 'quaternion_difference'])


class Encoder(nn.Module):
    def __init__(self, config, ivertices, faces, face_features, width, height, n_feat):
        super(Encoder, self).__init__()
        self.config = config

        # Translation initialization
        translation_init = torch.zeros(1, 1, config.input_frames, 3)
        translation_init[:, :, 0, 2] = self.config.tran_init

        # Quaternion initialization
        qinit = torch.zeros(1, config.input_frames, 4)
        qinit[:, :, 0] = 1.0
        init_angle = torch.Tensor(self.config.rot_init)
        init_quat = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ)
        self.register_buffer('init_quat', init_quat)
        qinit[:, 0, :] = init_quat.clone()

        # Used translation and quaternion
        self.register_buffer('initial_translation', translation_init.clone())
        self.register_buffer('initial_quaternion', qinit.clone())

        # The offsets store the
        quaternion_offsets = torch.zeros(1, config.input_frames, 4)
        quaternion_offsets[:, :, 0] = 1.0
        translation_offsets = torch.zeros(1, 1, config.input_frames, 3)
        self.register_buffer('quaternion_offsets', quaternion_offsets)
        self.register_buffer('translation_offsets', translation_offsets)

        self.translation = nn.Parameter(torch.zeros(translation_offsets.shape))
        quat = torch.zeros(1, config.input_frames, 4)
        quat[:, :, 0] = 1.0
        self.quaternion = nn.Parameter(quat)

        # For logging purposes, store the values of development of estimated rotations and translations thorough the
        # gradient descent iterations
        self.rotation_by_gd_iter = []
        self.translation_by_gd_iter = []

        # Lights initialization
        if self.config.use_lights:
            lights = torch.zeros(1, 3, 9)
            lights[:, :, 0] = 0.5
            self.lights = nn.Parameter(lights)
        else:
            self.lights = None

        # Vertices initialization
        if self.config.predict_vertices:
            self.vertices = nn.Parameter(torch.zeros(1, ivertices.shape[0], 3))

        # Face features and texture map
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        self.texture_map = nn.Parameter(torch.ones(1, n_feat, self.config.texture_size, self.config.texture_size))

        # Normalize and register ivertices
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)

        # Aspect ratio
        self.aspect_ratio = height / width

    def set_grad_mesh(self, req_grad):
        self.texture_map.requires_grad = req_grad
        if self.config.predict_vertices:
            self.vertices.requires_grad = req_grad
        if self.config.use_lights:
            self.lights.requires_grad = req_grad

    def forward(self, opt_frames):

        if self.config.predict_vertices:
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

        self.log_rotation_and_translation(opt_frames, quaternion, translation)

        result = EncoderResult(translations=translation,
                               quaternions=quaternion,
                               vertices=vertices,
                               texture_maps=texture_map,
                               lights=self.lights,
                               translation_difference=tdiff,
                               quaternion_difference=qdiff)
        return result

    def compute_tdiff_qdiff(self, opt_frames, quaternion0, quaternion, translation):
        weights = (torch.Tensor(opt_frames) - torch.Tensor(opt_frames[:1] + opt_frames[:-1])).to(translation.device)
        # Temporal distance between consecutive items in opt_frames, i.e. weight grows linearly with distance
        tdiff = weights * comp_tran_diff(translation[0, 0, opt_frames])
        key_dists = []
        for frmi in opt_frames[1:]:
            key_dists.append(qdist(quaternion[:, frmi - 1], quaternion[:, frmi]))
        qdiff = weights * (torch.stack([qdist(quaternion0, quaternion0)] + key_dists, 0).contiguous())
        return tdiff, qdiff

    def get_total_rotation_at_frame(self, stepi):
        # The formula is initial_quaternion * quaternion_offsets * quaternion

        offset_initial_quaternion = qmult(qnorm(self.initial_quaternion[:, stepi]),
                                          qnorm(self.quaternion_offsets[:, stepi]))
        total_rotation_quaternion = qnorm(qmult(offset_initial_quaternion, qnorm(self.quaternion[:, stepi])))

        return total_rotation_quaternion

    def get_total_rotation_at_frame_vectorized(self):
        offset_initial_quaternion = quaternion_multiply(qnorm_vectorized(self.initial_quaternion),
                                                        qnorm_vectorized(self.quaternion_offsets))
        total_rotation_quaternion = qnorm_vectorized(quaternion_multiply(offset_initial_quaternion,
                                                                         qnorm_vectorized(self.quaternion)))

        return total_rotation_quaternion

    def get_total_translation_at_frame(self, stepi):
        # The formula is initial_translation * translation_offsets * translation
        return self.initial_translation[:, :, stepi] + self.translation_offsets[:, :, stepi] + \
            self.translation[:, :, stepi]

    def get_total_translation_at_frame_vectorized(self):
        # The formula is initial_translation * translation_offsets * translation
        return self.initial_translation + self.translation_offsets + self.translation

    def compute_next_offset(self, stepi):
        self.initial_translation[:, :, stepi] = self.initial_translation[:, :, stepi - 1]
        self.initial_quaternion[:, stepi] = self.initial_quaternion[:, stepi - 1]

        self.translation_offsets[:, :, stepi] = self.translation_offsets[:, :, stepi - 1] + \
                                                self.translation[:, :, stepi - 1].detach()
        self.quaternion_offsets[:, stepi] = qmult(qnorm(self.quaternion_offsets[:, stepi - 1]),
                                                  qnorm(self.quaternion[:, stepi - 1]).detach())

    def log_rotation_and_translation(self, opt_frames, quaternion, translation):
        angles_rad = quaternion_to_angle_axis(quaternion[0], order=QuaternionCoeffOrder.WXYZ)
        angles_deg = angles_rad * 180.0 / math.pi
        last_optimized = max(opt_frames)
        self.rotation_by_gd_iter.append(angles_deg[last_optimized].detach())
        self.translation_by_gd_iter.append(translation[0, 0, last_optimized].detach())

    def clear_logs(self):
        self.rotation_by_gd_iter = []
        self.translation_by_gd_iter = []

    def forward_normalize(self):
        exp = 0
        if self.config.connect_frames:
            exp = nn.Sigmoid()(self.exposure_fraction)
        thr = self.config.camera_distance - 2
        thrn = thr * 4
        translation_all = []
        quaternion_all = []
        for frmi in range(self.translation.shape[1]):
            translation = nn.Tanh()(self.translation[:, frmi, None, :])

            translation = translation.view(translation.shape[:2] + torch.Size([1, 2, 3]))
            translation_new = translation.clone()
            translation_new[:, :, :, :, 2][translation[:, :, :, :, 2] > 0] = translation[:, :, :, :, 2][
                                                                                 translation[:, :, :, :, 2] > 0] * thr
            translation_new[:, :, :, :, 2][translation[:, :, :, :, 2] < 0] = translation[:, :, :, :, 2][
                                                                                 translation[:, :, :, :, 2] < 0] * thrn
            translation_new[:, :, :, :, :2] = translation[:, :, :, :, :2] * (
                        (self.config.camera_distance - translation_new[:, :, :, :, 2:]) / 2)
            translation = translation_new
            translation[:, :, :, :, 1] = self.aspect_ratio * translation_new[:, :, :, :, 1]

            if frmi > 0 and self.config.connect_frames:
                translation[:, :, :, 1, :] = translation_all[-1][:, :, :, 1, :] + (1 + exp) * \
                                             translation_all[-1][:, :, :, 0, :]

            translation[:, :, :, 0, :] = translation[:, :, :, 0, :] - translation[:, :, :, 1, :]

            quaternion = self.quaternion[:, frmi]
            quaternion = quaternion.view(quaternion.shape[:1] + torch.Size([1, 2, 4]))

            translation_all.append(translation)
            quaternion_all.append(quaternion)

        translation = torch.stack(translation_all, 2).contiguous()[:, :, :, 0]
        quaternion = torch.stack(quaternion_all, 1).contiguous()[:, :, 0]
        if self.config.predict_vertices:
            vertices = self.ivertices + self.vertices
            if self.config.mesh_normalize:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:, None, :]  # make center of mass in origin
        else:
            vertices = self.ivertices

        return translation, quaternion, vertices, self.texture_map, exp

from collections import namedtuple

import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_quaternion, QuaternionCoeffOrder
from kornia.geometry.quaternion import Quaternion
from kornia.geometry.liegroup import Se3, So3
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
        qinit[:, 0, :] = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ)

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

        axis_angle_offsets = torch.zeros(1, config.input_frames, 3)
        self.register_buffer('axis_angle_offsets', axis_angle_offsets)

        self.translation = nn.Parameter(torch.zeros(translation_offsets.shape))
        quat = torch.zeros(1, config.input_frames, 4)
        quat[:, :, 0] = 1.0

        self.quaternion_w = nn.Parameter(quat[..., 0, None])
        self.quaternion_x = nn.Parameter(quat[..., 1, None])
        self.quaternion_y = nn.Parameter(quat[..., 2, None])
        self.quaternion_z = nn.Parameter(quat[..., 3, None])

        axis_angles = torch.zeros(1, config.input_frames, 3)
        self.axis_angle_x = nn.Parameter(axis_angles[..., 0, None])
        self.axis_angle_y = nn.Parameter(axis_angles[..., 1, None])
        self.axis_angle_z = nn.Parameter(axis_angles[..., 2, None])

        se3_algebra_init = Se3(So3(Quaternion(qinit[0])), translation_init[0, 0]).log()
        se3_algebra_offsets = Se3(So3(Quaternion(quaternion_offsets[0])), translation_offsets[0, 0]).log()

        self.register_buffer('se3_algebra_init', se3_algebra_init)
        self.register_buffer('se3_algebra_offsets', se3_algebra_offsets)
        self.se3_algebra = nn.Parameter(torch.zeros(se3_algebra_init.shape))

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
        self.texture_map = nn.Parameter(torch.ones(1, n_feat, self.config.texture_size, self.config.texture_size))

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

        se3 = self.get_composed_se3_at_frame_vectorized()
        translation = se3.t[None, None].clone()
        quaternion = se3.r.q.data[None].clone()

        translation = self.get_total_translation_at_frame_vectorized()
        quaternion = self.get_total_rotation_at_frame_vectorized()
        rotation = self.get_total_rotation_at_frame_vectorized_axis_angle()
        translation[:, :, 0] = translation[:, :, 0].detach()
        quaternion[:, 0] = quaternion[:, 0].detach()

        noopt = list(set(range(translation.shape[2])) - set(opt_frames))

        translation[:, :, noopt] = translation[:, :, noopt].detach()
        quaternion[:, noopt] = quaternion[:, noopt].detach()
        rotation[:, noopt] = rotation[:, noopt].detach()

        quaternion = quaternion[:, :opt_frames[-1] + 1]
        rotation = rotation[:, :opt_frames[-1] + 1]
        translation = translation[:, :, :opt_frames[-1] + 1]
        # translation = translation.detach()
        # quaternion = quaternion.detach()

        if self.config.features == 'deep':
            texture_map = self.texture_map
        else:
            texture_map = nn.Sigmoid()(self.texture_map)

        # Computes differences of consecutive translations and rotations
        tdiff, qdiff = self.compute_tdiff_qdiff(opt_frames, quaternion[:, -1], quaternion, translation)

        # quaternion = angle_axis_to_quaternion(rotation, order=QuaternionCoeffOrder.WXYZ)
        # quaternion = axis_angle_to_quaternion(rotation)

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
        offset_initial_quaternion = quaternion_multiply(qnorm_vectorized(self.initial_quaternion),
                                                        qnorm_vectorized(self.quaternion_offsets))
        quaternion = torch.cat([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], dim=-1)
        total_rotation_quaternion = qnorm_vectorized(quaternion_multiply(offset_initial_quaternion,
                                                                         qnorm_vectorized(quaternion)))

        return total_rotation_quaternion

    def get_total_rotation_at_frame_vectorized_axis_angle(self):
        axis_angle_rot = torch.cat([self.axis_angle_x, self.axis_angle_y, self.axis_angle_z], dim=-1)
        return self.initial_axis_angle + self.axis_angle_offsets + axis_angle_rot

    def get_total_translation_at_frame_vectorized(self):
        # The formula is initial_translation * translation_offsets * translation
        return self.initial_translation + self.translation_offsets + self.translation

    def get_composed_se3_at_frame_vectorized(self):
        return Se3.exp(self.se3_algebra_init) * Se3.exp(self.se3_algebra_offsets) * Se3.exp(self.se3_algebra)

    def compute_next_offset(self, stepi):
        self.initial_translation[:, :, stepi] = self.initial_translation[:, :, stepi - 1]
        self.initial_quaternion[:, stepi] = self.initial_quaternion[:, stepi - 1]
        self.initial_axis_angle[:, stepi] = self.initial_axis_angle[:, stepi - 1]
        self.se3_algebra_init[stepi] = self.se3_algebra_init[stepi - 1]

        self.translation_offsets[:, :, stepi] = self.translation_offsets[:, :, stepi - 1] + \
                                                self.translation[:, :, stepi - 1].detach()
        quaternion = torch.cat([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], dim=-1)
        self.quaternion_offsets[:, stepi] = qmult(qnorm(self.quaternion_offsets[:, stepi - 1]),
                                                  qnorm(quaternion[:, stepi - 1]).detach())

        axis_angle_rot = torch.cat([self.axis_angle_x, self.axis_angle_y, self.axis_angle_z], dim=-1)
        self.axis_angle_offsets[:, stepi] = self.axis_angle_offsets[:, stepi] + axis_angle_rot[:, stepi - 1].detach()

        self.axis_angle_offsets[:, stepi] = self.axis_angle_offsets[:, stepi] + axis_angle_rot[:, stepi - 1].detach()

        self.se3_algebra_offsets[stepi] = (Se3.exp(self.se3_algebra_offsets[stepi]) *
                                           Se3.exp(self.se3_algebra[stepi])).log()

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

            quaternion = torch.cat([self.quaternion_w, self.quaternion_x, self.quaternion_y, self.quaternion_z], dim=-1)
            quaternion = quaternion[:, frmi]
            quaternion = quaternion.view(quaternion.shape[:1] + torch.Size([1, 2, 4]))

            translation_all.append(translation)
            quaternion_all.append(quaternion)

        translation = torch.stack(translation_all, 2).contiguous()[:, :, :, 0]
        quaternion = torch.stack(quaternion_all, 1).contiguous()[:, :, 0]
        if self.config.optimize_shape:
            vertices = self.ivertices + self.vertices
            if self.config.mesh_normalize:
                vertices = mesh_normalize(vertices)
            else:
                vertices = vertices - vertices.mean(1)[:, None, :]  # make center of mass in origin
        else:
            vertices = self.ivertices

        return translation, quaternion, vertices, self.texture_map, exp

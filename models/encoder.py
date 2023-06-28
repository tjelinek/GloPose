from collections import namedtuple
import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_quaternion, QuaternionCoeffOrder

from utils import mesh_normalize, comp_tran_diff, qnorm, qmult, qdist, qdifference

EncoderResult = namedtuple('EncoderResult', ['translations', 'quaternions', 'vertices', 'texture_maps',
                                             'lights', 'translation_difference', 'quaternion_difference'])


class Encoder(nn.Module):
    def __init__(self, config, ivertices, faces, face_features, width, height, n_feat):
        super(Encoder, self).__init__()
        self.config = config

        # Translation initialization
        translation_init = torch.zeros(1, 1, config.input_frames, 3)
        translation_init[:, :, 0, 2] = self.config.tran_init
        self.translation = nn.Parameter(translation_init)

        # Quaternion initialization
        qinit = torch.zeros(1, config.input_frames, 4)
        qinit[:, :, 0] = 1.0
        init_angle = torch.Tensor(self.config.rot_init)
        init_quat = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ)
        self.register_buffer('init_quat', init_quat)
        qinit[:, 0, :] = init_quat.clone()
        self.quaternion = nn.Parameter(qinit)

        # Offsets initialization
        quaternion_offsets = torch.zeros(1, config.input_frames, 4)
        quaternion_offsets[:, :, 0] = 1.0
        translation_offsets = torch.zeros(1, 1, config.input_frames, 3)
        self.register_buffer('quaternion_offsets', quaternion_offsets)
        self.register_buffer('translation_offsets', translation_offsets)

        # Used translation and quaternion
        self.register_buffer('used_tran', translation_init.clone())
        self.register_buffer('used_quat', qinit.clone())

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
        if 0 in opt_frames:
            quaternion_all = [qnorm(self.quaternion[:, 0])]
            translation_all = [self.translation[:, :, 0]]
        else:
            quaternion_all = [qnorm(self.quaternion[:, 0]).detach()]
            translation_all = [self.translation[:, :, 0].detach()]
        diffs = []
        dists = [qdist(quaternion_all[-1], quaternion_all[-1]), qdist(quaternion_all[-1], quaternion_all[-1])]
        for frmi in range(1, opt_frames[-1] + 1):
            quaternion0 = qmult(qnorm(self.quaternion[:, frmi]), qnorm(self.quaternion_offsets[:, frmi]))
            translation0 = self.translation[:, :, frmi] + self.translation_offsets[:, :, frmi]
            # breakpoint()
            if frmi not in opt_frames:
                quaternion0 = quaternion0.detach()
                translation0 = translation0.detach()
            diffs.append(qnorm(qdifference(quaternion_all[-1], quaternion0)))
            if len(diffs) > 1:
                dists.append(qdist(diffs[-2], diffs[-1]))
            quaternion_all.append(quaternion0)
            translation_all.append(translation0)

        if max(opt_frames) == 0:
            quaternion0 = qmult(qnorm(self.quaternion[:, 0]), qnorm(self.quaternion_offsets[:, 0]))

        quaternion = torch.stack(quaternion_all, 1).contiguous()
        translation = torch.stack(translation_all, 2).contiguous()
        wghts = (torch.Tensor(opt_frames) - torch.Tensor(opt_frames[:1] + opt_frames[:-1])).to(translation.device)
        tdiff = wghts * comp_tran_diff(translation[0, 0, opt_frames])
        key_dists = []
        for frmi in opt_frames[1:]:
            key_dists.append(qdist(quaternion[:, frmi - 1], quaternion[:, frmi]))
        qdiff = wghts * (torch.stack([qdist(quaternion0, quaternion0)] + key_dists, 0).contiguous())
        if self.config.features == 'deep':
            texture_map = self.texture_map
        else:
            texture_map = nn.Sigmoid()(self.texture_map)

        result = EncoderResult(translations=translation,
                               quaternions=quaternion,
                               vertices=vertices,
                               texture_maps=texture_map,
                               lights=self.lights,
                               translation_difference=tdiff,
                               quaternion_difference=qdiff)
        return result

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
                translation[:, :, :, 1, :] = translation_all[-1][:, :, :, 1, :] + (1 + exp) * translation_all[-1][:, :,
                                                                                              :, 0, :]

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

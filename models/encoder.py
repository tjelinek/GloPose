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
        translation_init = torch.zeros(1, 1, config.input_frames, 3)
        translation_init[:, :, 0, 2] = self.config.tran_init
        self.translation = nn.Parameter(translation_init)
        qinit = torch.zeros(1, config.input_frames, 4)  # *0.005
        qinit[:, :, 0] = 1.0
        init_angle = torch.Tensor(self.config.rot_init)
        init_quat = angle_axis_to_quaternion(init_angle, order=QuaternionCoeffOrder.WXYZ)
        self.register_buffer('init_quat', init_quat)
        qinit[:, 0, :] = init_quat.clone()
        self.quaternion = nn.Parameter(qinit)
        offsets = torch.zeros(1, 1, config.input_frames, 7)
        offsets[:, :, :, 3] = 1.0
        self.register_buffer('offsets', offsets)
        self.register_buffer('used_tran', translation_init.clone())
        self.register_buffer('used_quat', qinit.clone())
        if self.config.use_lights:
            lights = torch.zeros(1, 3, 9)
            lights[:, :, 0] = 0.5
            self.lights = nn.Parameter(lights)
        else:
            self.lights = None
        if self.config.predict_vertices:
            self.vertices = nn.Parameter(torch.zeros(1, ivertices.shape[0], 3))
        self.register_buffer('face_features', torch.from_numpy(face_features).unsqueeze(0).type(self.translation.dtype))
        self.texture_map = nn.Parameter(torch.ones(1, n_feat, self.config.texture_size, self.config.texture_size))
        ivertices = torch.from_numpy(ivertices).unsqueeze(0).type(self.translation.dtype)
        ivertices = mesh_normalize(ivertices)
        self.register_buffer('ivertices', ivertices)
        self.aspect_ratio = height / width
        if self.config.project_coin:
            thr = 0.025
            x_coor = ivertices[:, :, 0]
            x_coor[x_coor > thr] = thr
            x_coor[x_coor < -thr] = -thr
            self.register_buffer('x_coor', x_coor)

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
            if self.config.project_coin:
                vertices[:, :, 0] = self.x_coor
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
            quaternion0 = qmult(qnorm(self.quaternion[:, frmi]), qnorm(self.offsets[:, 0, frmi, 3:]))
            translation0 = self.translation[:, :, frmi] + self.offsets[:, :, frmi, :3]
            if not frmi in opt_frames:
                quaternion0 = quaternion0.detach()
                translation0 = translation0.detach()
            diffs.append(qnorm(qdifference(quaternion_all[-1], quaternion0)))
            if len(diffs) > 1:
                dists.append(qdist(diffs[-2], diffs[-1]))
            quaternion_all.append(quaternion0)
            translation_all.append(translation0)

        if max(opt_frames) == 0:
            quaternion0 = qmult(qnorm(self.quaternion[:, 0]), qnorm(self.offsets[:, 0, 0, 3:]))

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

        result = EncoderResult(translations=translation[:, :, opt_frames],
                               quaternions=quaternion[:, opt_frames],
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

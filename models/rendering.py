from collections import namedtuple

import kaolin
import math
import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis, \
    quaternion_to_rotation_matrix, QuaternionCoeffOrder
from kornia.morphology import erosion

import cfg
from models.kaolin_wrapper import prepare_vertices


def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    return deringed_coeffs


MeshRenderResult = namedtuple('MeshRenderResult', ['face_normals', 'face_vertices_cam', 'red_index',
                                                   'ren_mask', 'ren_mesh_vertices_features',
                                                   'ren_mesh_vertices_coords',
                                                   'ren_mesh_vertices_image_coords'])


class RenderingKaolin(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57 / 2,  # field of view 45 degrees
                                                                           self.width / self.height)
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0, 0, self.config.camera_distance])[None])
        self.register_buffer('obj_center', torch.zeros((1, 3)))
        camera_up_direction = torch.Tensor((0, 1, 0))[None]
        camera_rot, _ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center,
                                                                                camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).cuda()
        self.register_buffer('kernel', kernel)

    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))
        self.register_buffer('face_indices', self.faces.clone().detach().to(dtype=torch.long, device=cfg.DEVICE))

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps=None, lights=None,
                render_depth=False):
        rendered_verts_positions = []
        all_renders = []
        ren_mesh_vertices_features_list = []

        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi], order=QuaternionCoeffOrder.WXYZ)

            renders = []

            rendering_result = self.render_mesh_with_dibr(face_features, rotation_matrix,
                                                          translation_vector, unit_vertices)

            face_vertices_z = rendering_result.face_vertices_cam[:, :, :, -1]

            if texture_maps is not None:
                ren_features = kaolin.render.mesh.texture_mapping(rendering_result.ren_mesh_vertices_features,
                                                                  texture_maps,
                                                                  mode='bilinear')
            rendered_verts_positions.append(rendering_result.ren_mesh_vertices_coords)
            ren_mesh_vertices_features_list.append(rendering_result.ren_mesh_vertices_features)

            if lights is not None:
                im_normals = rendering_result.face_normals[0, rendering_result.red_index, :]
                lighting = None
                for li in range(lights.shape[0]):
                    lighting_r = \
                        kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 0])[
                            ..., None]
                    lighting_g = \
                        kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 1])[
                            ..., None]
                    lighting_b = \
                        kaolin.render.mesh.utils.spherical_harmonic_lighting(im_normals, lights[li:(li + 1), 2])[
                            ..., None]
                    lighting_one = torch.cat((lighting_r, lighting_g, lighting_b), 3)
                    if lighting is None:
                        lighting = lighting_one
                    else:
                        lighting += lighting_one
                lighting[rendering_result.red_index[..., None][:, :, :, [0, 0, 0]] < 0] = 1
                ren_features = ren_features * lighting
            result = ren_features.permute(0, 3, 1, 2)
            if self.config.erode_renderer_mask > 0:
                ren_mask = erosion(rendering_result.ren_mask[:, None], self.kernel)[:, 0]
            else:
                ren_mask = torch.ones(1, self.height, self.width)

            if render_depth:
                depth_map = face_vertices_z[0, rendering_result.red_index, :].mean(3)[:, None]
                result_rgba = torch.cat((result, ren_mask[:, None], depth_map), 1)
            else:
                result_rgba = torch.cat((result, ren_mask[:, None]), 1)
            renders.append(result_rgba)

            renders = torch.stack(renders, 1).contiguous()
            all_renders.append(renders)

        all_renders = torch.stack(all_renders, 1).contiguous()

        return all_renders

    def compute_theoretical_flow(self, encoder_out, encoder_out_prev_frames):
        """
        Computes the theoretical flow between consecutive frames.

        Args:
            encoder_out (EncoderResult): The encoder result for the current frame.
            encoder_out_prev_frames (EncoderResult): The encoder results for the previous frames.

        Returns:
            torch.Tensor: The computed theoretical flow between consecutive frames. The output flow is respective to the
                          coordinates range [0, 1].
        """
        theoretical_flows = []
        rendering_masks = []
        for frame_i in range(0, encoder_out.quaternions.shape[1]):
            translation_vector_1 = encoder_out_prev_frames.translations[:, :, frame_i]
            rotation_matrix_1 = quaternion_to_rotation_matrix(encoder_out_prev_frames.quaternions[:, frame_i],
                                                              order=QuaternionCoeffOrder.WXYZ).to(torch.float)

            translation_vector_2 = encoder_out.translations[:, :, frame_i]
            rotation_matrix_2 = quaternion_to_rotation_matrix(encoder_out.quaternions[:, frame_i],
                                                              order=QuaternionCoeffOrder.WXYZ).to(torch.float)

            # Rotate and translate the vertices using the given rotation_matrix and translation_vector
            vertices_1 = kaolin.render.camera.rotate_translate_points(encoder_out.vertices,
                                                                      rotation_matrix_1, self.obj_center)
            vertices_2 = kaolin.render.camera.rotate_translate_points(encoder_out.vertices,
                                                                      rotation_matrix_2, self.obj_center)

            vertices_1 = vertices_1 + translation_vector_1
            vertices_2 = vertices_2 + translation_vector_2

            face_vertices_cam_1, face_vertices_image_1, face_normals_1 = prepare_vertices(vertices_1, self.faces,
                                                                                          self.camera_rot,
                                                                                          self.camera_trans,
                                                                                          self.camera_proj)
            face_vertices_cam_2, face_vertices_image_2, face_normals_2 = prepare_vertices(vertices_2, self.faces,
                                                                                          self.camera_rot,
                                                                                          self.camera_trans,
                                                                                          self.camera_proj)

            # Extract the z-coordinates of the face vertices in camera space
            face_vertices_z_1 = face_vertices_cam_1[:, :, :, -1]

            # Extract the z-components of the face normals
            face_normals_z_1 = face_normals_1[:, :, -1]

            face_vertices_3d_motion = face_vertices_cam_2 - face_vertices_cam_1
            face_vertices_image_motion = face_vertices_image_2 - face_vertices_image_1  # Vertices are in [-1, 1] range
            features_for_rendering = face_vertices_image_motion

            ren_outputs, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                     face_vertices_z_1,
                                                                                     face_vertices_image_1,
                                                                                     features_for_rendering,
                                                                                     face_normals_z_1,
                                                                                     sigmainv=self.config.sigmainv,
                                                                                     boxlen=0.02, knum=30,
                                                                                     multiplier=1000)

            theoretical_flow = ren_outputs  # torch.Size([1, H, W, 2])
            theoretical_flow_new = theoretical_flow.clone()  # Create a new tensor with the same values
            theoretical_flow_new[..., 0] = theoretical_flow[..., 0] * 0.5
            theoretical_flow_new[..., 1] = -theoretical_flow[..., 1] * 0.5  # Correction for transform into image
            theoretical_flow = theoretical_flow_new

            theoretical_flows.append(theoretical_flow)

            rendering_masks.append(ren_mask)

        theoretical_flow = torch.stack(theoretical_flows, 1)  # torch.Size([1, N, H, W, 2])
        flow_render_mask = torch.stack(rendering_masks, 1)  # torch.Size([1, N, H, W])

        return theoretical_flow, flow_render_mask

    def render_mesh_with_dibr(self, face_features, rotation_matrix, translation_vector, unit_vertices) \
            -> MeshRenderResult:
        # Rotate and translate the vertices using the given rotation_matrix and translation_vector
        vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center)

        # Apply the translation to the vertices
        vertices = vertices + translation_vector

        # Prepare the vertices for rendering by computing their camera coordinates, image coordinates, and face normals
        face_vertices_cam, face_vertices_image, face_normals = prepare_vertices(vertices, self.faces,
                                                                                self.camera_rot, self.camera_trans,
                                                                                self.camera_proj)

        # Extract the z-coordinates of the face vertices in camera space
        face_vertices_z = face_vertices_cam[:, :, :, -1]

        # Extract the z-components of the face normals
        face_normals_z = face_normals[:, :, -1]

        features_for_rendering = torch.cat((face_features, face_vertices_cam, face_vertices_image), dim=-1)

        # Perform dibr rasterization
        ren_outputs, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                 face_vertices_z,
                                                                                 face_vertices_image,
                                                                                 features_for_rendering,
                                                                                 face_normals_z,
                                                                                 sigmainv=self.config.sigmainv,
                                                                                 boxlen=0.02, knum=30, multiplier=1000)

        # Extract ren_mesh_vertices_features and ren_mesh_vertices_coords from the combined output tensor
        ren_mesh_vertices_features = ren_outputs[..., :face_features.shape[-1]]
        ren_mesh_vertices_coords = ren_outputs[..., face_features.shape[-1]:
                                                    face_features.shape[-1] + face_vertices_cam.shape[-1]]
        ren_mesh_vertices_image_coords = ren_outputs[..., face_features.shape[-1] + face_vertices_cam.shape[-1]:]

        return MeshRenderResult(face_normals, face_vertices_cam, red_index, ren_mask,
                                ren_mesh_vertices_features, ren_mesh_vertices_coords,
                                ren_mesh_vertices_image_coords)

    def get_rgb_texture(self, translation, quaternion, unit_vertices, face_features, input_batch):
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).to(
            translation.device)
        tex = torch.zeros(1, 3, self.config.texture_size, self.config.texture_size)
        cnt = torch.zeros(self.config.texture_size, self.config.texture_size)
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi], order=QuaternionCoeffOrder.WXYZ)

            rendering_result = self.render_mesh_with_dibr(face_features, rotation_matrix,
                                                          translation_vector, unit_vertices)

            coord = torch.round((1 - rendering_result.ren_mesh_vertices_features) * self.config.texture_size).to(int)
            coord[coord >= self.config.texture_size] = self.config.texture_size - 1
            coord[coord < 0] = 0
            xc = coord[0, :, :, 1].reshape([coord.shape[1] * coord.shape[2]])
            yc = (self.config.texture_size - 1 - coord[0, :, :, 0]).reshape([coord.shape[1] * coord.shape[2]])
            cr = input_batch[0, frmi, 0].reshape([coord.shape[1] * coord.shape[2]])
            cg = input_batch[0, frmi, 1].reshape([coord.shape[1] * coord.shape[2]])
            cb = input_batch[0, frmi, 2].reshape([coord.shape[1] * coord.shape[2]])
            for ki in range(xc.shape[0]):
                cnt[xc[ki], yc[ki]] = cnt[xc[ki], yc[ki]] + 1
                tex[0, 0, xc[ki], yc[ki]] = tex[0, 0, xc[ki], yc[ki]] + cr[ki]
                tex[0, 1, xc[ki], yc[ki]] = tex[0, 1, xc[ki], yc[ki]] + cg[ki]
                tex[0, 2, xc[ki], yc[ki]] = tex[0, 2, xc[ki], yc[ki]] + cb[ki]
        tex_final = tex / cnt[None, None]
        return tex_final


def generate_rotation(rotation_current, my_rot, steps=3):
    step = angle_axis_to_rotation_matrix(torch.Tensor([my_rot])).to(rotation_current.device)
    step_back = angle_axis_to_rotation_matrix(torch.Tensor([-np.array(my_rot)])).to(rotation_current.device)
    for ki in range(steps):
        rotation_current = torch.matmul(rotation_current, step_back)
    rotation_matrix_join = torch.cat((step[None], rotation_current[None]), 1)[None]
    return rotation_matrix_join


def generate_all_views(best_model, static_translation, rotation_matrix, rendering, small_step, extreme_step=None,
                       num_small_steps=1):
    rendering.config.fmo_steps = 2
    if not extreme_step is None:
        ext_renders = rendering(static_translation, generate_rotation(rotation_matrix, extreme_step, 0),
                                best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
        ext_renders_neg = rendering(static_translation, generate_rotation(rotation_matrix, -np.array(extreme_step), 0),
                                    best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    rendering.config.fmo_steps = num_small_steps + 1
    renders = rendering(static_translation, generate_rotation(rotation_matrix, small_step, 0), best_model["vertices"],
                        best_model["face_features"], best_model["texture_maps"])
    renders_neg = rendering(static_translation, generate_rotation(rotation_matrix, -np.array(small_step), 0),
                            best_model["vertices"], best_model["face_features"], best_model["texture_maps"])
    if not extreme_step is None:
        all_renders = torch.cat(
            (ext_renders_neg[:, :, -1:], torch.flip(renders_neg[:, :, 1:], [2]), renders, ext_renders[:, :, -1:]), 2)
    else:
        all_renders = torch.cat((torch.flip(renders_neg[:, :, 1:], [2]), renders), 2)
    return all_renders.detach().cpu().numpy()[0, 0].transpose(2, 3, 1, 0)


def prepare_views(best_model, config):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config.erode_renderer_mask = 7
    config.fmo_steps = best_model["renders"].shape[-4]
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    static_translation = best_model["translation"].clone()
    static_translation[:, :, :, 1] = static_translation[:, :, :, 1] + 0.5 * static_translation[:, :, :, 0]
    static_translation[:, :, :, 0] = 0
    quaternion = best_model["quaternion"][:, :1].clone()
    rotation_matrix = angle_axis_to_rotation_matrix(quaternion_to_angle_axis(quaternion[:, 0, 1]))
    rotation_matrix_step = angle_axis_to_rotation_matrix(
        quaternion_to_angle_axis(quaternion[:, 0, 0]) / config.fmo_steps / 2)
    return rendering, rotation_matrix, rotation_matrix_step, static_translation


def generate_novel_views(best_model, config):
    rendering, rotation_matrix, rotation_matrix_step, static_translation = prepare_views(best_model, config)
    for ki in range(int(config.fmo_steps / 2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)

    vertical = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi / 2 / 9, 0, 0],
                                  [math.pi / 3, 0, 0], 3)
    horizontal = generate_all_views(best_model, static_translation, rotation_matrix, rendering,
                                    [0, math.pi / 2 / 9, math.pi / 2 / 9], [0, math.pi / 3, math.pi / 3], 3)
    joint = generate_all_views(best_model, static_translation, rotation_matrix, rendering,
                               [math.pi / 2 / 9, math.pi / 2 / 9, math.pi / 2 / 9],
                               [math.pi / 3, math.pi / 3, math.pi / 3], 3)
    return horizontal, vertical, joint


def generate_video_views(best_model, config):
    rendering, rotation_matrix, rotation_matrix_step, static_translation = prepare_views(best_model, config)
    for ki in range(int(config.fmo_steps / 2)): rotation_matrix = torch.matmul(rotation_matrix, rotation_matrix_step)

    views = generate_all_views(best_model, static_translation, rotation_matrix, rendering, [math.pi / 2 / 9 / 10, 0, 0],
                               None, 45)
    return views


def generate_tsr_video(best_model, config, steps=8):
    width = best_model["renders"].shape[-1]
    height = best_model["renders"].shape[-2]
    config.erode_renderer_mask = 7
    config.fmo_steps = steps
    rendering = RenderingKaolin(config, best_model["faces"], width, height).to(best_model["translation"].device)
    renders = rendering(best_model["translation"], best_model["quaternion"], best_model["vertices"],
                        best_model["face_features"], best_model["texture_maps"])
    tsr = renders.detach().cpu().numpy()[0, 0].transpose(2, 3, 1, 0)
    return tsr

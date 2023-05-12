import math
from dataclasses import dataclass
from itertools import product

import torch.nn as nn
from kaolin.render.camera import rotate_translate_points
from kornia.geometry.conversions import angle_axis_to_rotation_matrix, quaternion_to_angle_axis
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
from kornia.morphology import erosion

import cfg
from models.kaolin_wrapper import *
from utils import quaternion_multiply, calculate_rotation_difference
from models.encoder import qdifference, qnorm


def deringing(coeffs, window):
    deringed_coeffs = torch.zeros_like(coeffs)
    deringed_coeffs[:, 0] += coeffs[:, 0]
    deringed_coeffs[:, 1:1 + 3] += \
        coeffs[:, 1:1 + 3] * math.pow(math.sin(math.pi * 1.0 / window) / (math.pi * 1.0 / window), 4.0)
    deringed_coeffs[:, 4:4 + 5] += \
        coeffs[:, 4:4 + 5] * math.pow(math.sin(math.pi * 2.0 / window) / (math.pi * 2.0 / window), 4.0)
    return deringed_coeffs


class RenderingKaolin(nn.Module):
    def __init__(self, config, faces, width, height):
        super().__init__()
        self.config = config
        self.height = height
        self.width = width
        camera_proj = kaolin.render.camera.generate_perspective_projection(1.57 / 2,
                                                                           self.width / self.height)  # 45 degrees
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor([0, 0, self.config.camera_distance])[None])
        self.register_buffer('obj_center', torch.zeros((1, 3)))
        camera_up_direction = torch.Tensor((0, 1, 0))[None]
        camera_rot, _ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center,
                                                                                camera_up_direction)
        self.register_buffer('camera_rot', camera_rot)
        self.set_faces(faces)
        # self.faces = torch.LongTensor(faces).to(cfg.DEVICE)

    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))
        self.register_buffer('face_indices', torch.tensor(self.faces, dtype=torch.long, device=cfg.DEVICE))

    @dataclass
    class RenderingResult:
        all_renders: torch.TensorType
        theoretical_flow: torch.TensorType
        texture_flow: torch.TensorType

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps=None, lights=None,
                render_depth=False):
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).to(
            translation.device)

        rendered_verts_positions = []
        all_renders = []
        ren_mesh_vertices_features_list = []

        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi], order=QuaternionCoeffOrder.WXYZ)

            renders = []

            face_normals, face_vertices_cam, red_index, ren_mask, ren_mesh_vertices_features, ren_mesh_vertices_coords \
                = self.render_mesh_with_dibr(face_features, rotation_matrix, translation_vector, unit_vertices)

            face_vertices_z = face_vertices_cam[:, :, :, -1]

            if texture_maps is not None:
                ren_features = kaolin.render.mesh.texture_mapping(ren_mesh_vertices_features,
                                                                  texture_maps,
                                                                  mode='bilinear')
            rendered_verts_positions.append(ren_mesh_vertices_coords)
            ren_mesh_vertices_features_list.append(ren_mesh_vertices_features)

            if lights is not None:
                im_normals = face_normals[0, red_index, :]
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
                lighting[red_index[..., None][:, :, :, [0, 0, 0]] < 0] = 1
                ren_features = ren_features * lighting
            result = ren_features.permute(0, 3, 1, 2)
            if self.config.erode_renderer_mask > 0:
                ren_mask = erosion(ren_mask[:, None], kernel)[:, 0]
            if render_depth:
                depth_map = face_vertices_z[0, red_index, :].mean(3)[:, None]
                result_rgba = torch.cat((result, ren_mask[:, None], depth_map), 1)
            else:
                result_rgba = torch.cat((result, ren_mask[:, None]), 1)
            renders.append(result_rgba)

            renders = torch.stack(renders, 1).contiguous()
            all_renders.append(renders)

        rendered_verts_positions = torch.stack(rendered_verts_positions, dim=0)

        q1 = quaternion[:, -2]
        q2 = quaternion[:, -1]
        # qd = calculate_rotation_difference(q1, q2)
        qd = qdifference(q1, q2)
        rd = quaternion_to_rotation_matrix(qd, order=QuaternionCoeffOrder.WXYZ)

        # rd.register_hook(lambda grad: breakpoint())

        t1 = translation[:, :, -2]
        t2 = translation[:, :, -1]
        td = t2 - t1

        rendered_3d_coords_camera_i1 = rendered_verts_positions[-2, ...].flatten(1, 2)

        rendered_3d_coords_camera_i2 = rendered_3d_coords_camera_i1.clone()
        rendered_3d_coords_camera_i2 = rotate_translate_points(rendered_3d_coords_camera_i1, rd,
                                                               torch.zeros((1, 3), device=cfg.DEVICE))

        rendered_3d_coords_camera_i2 = rendered_3d_coords_camera_i2 + td

        rendered_vertices_positions1 = rendered_vertices_positions[-2, ...].flatten(1, 2)

        vertices_positions_image_flat1 = kaolin.render.camera.perspective_camera(rendered_vertices_positions1,
                                                                                 self.camera_proj)
        vertices_positions_image_flat2 = kaolin.render.camera.perspective_camera(rendered_vertices_positions2,
                                                                                 self.camera_proj)

        vertices_positions_image2 = projected_3d_coords_to_2d_i2_flat.reshape(*rendered_verts_positions.shape[1:-1], 2)
        vertices_positions_image2 = vertices_positions_image2.nan_to_num()

        vertices_positions_image1 = projected_3d_coords_to_2d_i1_flat.reshape(*rendered_verts_positions.shape[1:-1], 2)
        vertices_positions_image1 = vertices_positions_image1.nan_to_num()

        theoretical_flow = vertices_positions_image2 - vertices_positions_image1
        theoretical_flow = (theoretical_flow * ren_mask.unsqueeze(3))

        texture_flow = ren_mesh_vertices_features_list[-1] - ren_mesh_vertices_features_list[-2]

        all_renders = torch.stack(all_renders, 1).contiguous()
        return all_renders, theoretical_flow, texture_flow

    def compute_texture_to_img_map(self, ren_mesh_vertices_features, texture_maps):
        # TODO this is an experimental inverse mapping computation, remove
        # Compute the inverse mapping
        texture_img_map = torch.zeros(texture_maps.shape[-2], texture_maps.shape[-1], 2)
        for x, y in product(range(texture_maps.shape[-2]), range(texture_maps.shape[-1])):
            texture_map_coordinates_x = torch.full((1, *ren_mesh_vertices_features.shape[-3: -1], 1), x,
                                                   device=cfg.DEVICE)
            texture_map_coordinates_y = torch.full((1, *ren_mesh_vertices_features.shape[-3: -1], 1), y,
                                                   device=cfg.DEVICE)
            texture_map_coordinates = torch.cat((texture_map_coordinates_x, texture_map_coordinates_y), dim=-1)
            rendering_difference_map = ren_mesh_vertices_features - texture_map_coordinates
            rendering_difference_map = rendering_difference_map[..., 0] ** 2 + rendering_difference_map[..., 1] ** 2
            texture_img_map[x, y] = (rendering_difference_map == torch.max(rendering_difference_map)).nonzero()[0, 1:]

        return texture_img_map
        # TODO this is an experimental inverse mapping computation, remove END

    def render_mesh_with_dibr(self, face_features, rotation_matrix, translation_vector, unit_vertices):
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

        features_for_rendering = torch.cat((face_features, face_vertices_cam), dim=-1)

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
        ren_mesh_vertices_coords = ren_outputs[..., face_features.shape[-1]:]

        return face_normals, face_vertices_cam, red_index, ren_mask, ren_mesh_vertices_features, ren_mesh_vertices_coords

    def get_rgb_texture(self, translation, quaternion, unit_vertices, face_features, input_batch):
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).to(
            translation.device)
        tex = torch.zeros(1, 3, self.config.texture_size, self.config.texture_size)
        cnt = torch.zeros(self.config.texture_size, self.config.texture_size)
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi], order=QuaternionCoeffOrder.WXYZ)

            face_normals, face_vertices_cam, red_index, ren_mask, ren_mesh_vertices_features, ren_mesh_vertices_coords \
                = self.render_mesh_with_dibr(face_features, rotation_matrix, translation_vector, unit_vertices)

            coord = torch.round((1 - ren_mesh_vertices_features) * self.config.texture_size).to(int)
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


def generate_novel_views(best_model, config):
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

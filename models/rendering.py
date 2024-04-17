from collections import namedtuple
from typing import Tuple

import kaolin
import torch
import torch.nn as nn
from kornia.geometry.conversions import quaternion_to_rotation_matrix
from kornia.geometry import Rt_to_matrix4x4
from kornia.morphology import erosion

import cfg
from models.encoder import EncoderResult
from tracker_config import TrackerConfig
from utils import normalize_rendered_flows

MeshRenderResult = namedtuple('MeshRenderResult', ['face_normals',
                                                   'face_vertices_cam',
                                                   'red_index',
                                                   'ren_mask',
                                                   'ren_mesh_vertices_features',
                                                   'ren_mesh_vertices_world_coords',
                                                   'ren_mesh_vertices_camera_coords',
                                                   'ren_mesh_vertices_image_coords',
                                                   'ren_face_normals'])

RenderedFlowResult = namedtuple('RenderedFlowResult', ['theoretical_flow',
                                                       'rendered_flow_segmentation',
                                                       'rendered_flow_occlusion'])

RenderingResult = namedtuple('RenderingResult', ['rendered_image',
                                                 'rendered_image_segmentation',
                                                 'rendered_face_world_coords',
                                                 'rendered_face_camera_coords',
                                                 'rendered_face_normals'])


class RenderingKaolin(nn.Module):
    def __init__(self, config: TrackerConfig, faces: torch.Tensor, width: int, height: int):
        super().__init__()

        self.config = config
        self.height = height
        self.width = width

        self.backview = False

        R_frontview_to_backview = torch.Tensor([[-1., 0., 0.],
                                                [0., -1., 0.],
                                                [0., 0., 1.]]).cuda().unsqueeze(0)
        t_frontview_to_backview = -torch.Tensor(self.config.camera_position).unsqueeze(-1).expand(1, 3, 1).cuda()

        self.T_frontview_to_backview = Rt_to_matrix4x4(R_frontview_to_backview, t_frontview_to_backview)

        self.fov = torch.pi / 4  # 45 degrees
        camera_proj = kaolin.render.camera.generate_perspective_projection(self.fov, self.width / self.height)
        self.register_buffer('camera_proj', camera_proj)
        self.register_buffer('camera_trans', torch.Tensor(self.config.camera_position)[None])
        self.register_buffer('camera_trans_backview', -torch.Tensor(self.config.camera_position)[None])
        self.register_buffer('obj_center', torch.zeros((1, 3)))
        camera_up_direction = torch.Tensor((0, 1, 0))[None]
        self.register_buffer('camera_up', camera_up_direction)

        camera_rot, _ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans, self.obj_center,
                                                                                camera_up_direction)
        camera_rot_backview, _ = kaolin.render.camera.generate_rotate_translate_matrices(self.camera_trans_backview,
                                                                                         self.obj_center,
                                                                                         camera_up_direction)

        self.register_buffer('camera_rot', camera_rot)
        self.register_buffer('camera_rot_backview', camera_rot_backview)

        self.intrinsics = (
            kaolin.render.camera.PinholeIntrinsics.from_fov(width, height, self.fov, x0=width / 2, y0=height / 2,
                                                            fov_direction=kaolin.render.camera.CameraFOV.VERTICAL,
                                                            ))
        camera_intrinsics = torch.Tensor([[self.intrinsics.focal_x, 0., self.intrinsics.x0],
                                          [0., self.intrinsics.focal_y, self.intrinsics.y0],
                                          [0., 0., 1.]])

        self.register_buffer('camera_intrinsics', camera_intrinsics)
        self.set_faces(faces)
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).cuda()
        self.register_buffer('kernel', kernel)

    def set_faces(self, faces):
        self.register_buffer('faces', torch.LongTensor(faces))
        self.register_buffer('face_indices', self.faces.clone().detach().to(dtype=torch.long, device=cfg.DEVICE))

    def forward(self, translation, quaternion, unit_vertices, face_features, texture_maps,
                lights=None) -> RenderingResult:
        batch_size = quaternion.shape[1]

        translation_vector = translation[0, 0]
        rotation_matrix = quaternion_to_rotation_matrix(quaternion)[0]

        unit_vertices_batched = unit_vertices.repeat(batch_size, 1, 1)
        face_features_batched = face_features.repeat(batch_size, 1, 1, 1)
        texture_maps_batched = texture_maps.repeat(batch_size, 1, 1, 1)

        dibr_result = self.render_mesh_with_dibr(face_features_batched, rotation_matrix, translation_vector,
                                                 unit_vertices_batched)

        ren_features = kaolin.render.mesh.texture_mapping(dibr_result.ren_mesh_vertices_features,
                                                          texture_maps_batched, mode='bilinear')

        if lights is not None:
            im_normals = dibr_result.face_normals[0, dibr_result.red_index, :]
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
            lighting[dibr_result.red_index[..., None][:, :, :, [0, 0, 0]] < 0] = 1
            ren_features = ren_features * lighting
        rendering_rgb = ren_features.permute(0, 3, 1, 2)
        if self.config.erode_renderer_mask > 0:
            ren_mask = erosion(dibr_result.ren_mask[:, None], self.kernel)
        else:
            ren_mask = torch.ones(batch_size, 1, self.height, self.width)

        renderings = rendering_rgb.unsqueeze(0)
        segmentations = ren_mask.unsqueeze(0)
        rendered_object_camera_coords = dibr_result.ren_mesh_vertices_camera_coords.permute(0, 3, 1, 2).unsqueeze(0)
        rendered_object_world_coords = dibr_result.ren_mesh_vertices_world_coords.permute(0, 3, 1, 2).unsqueeze(0)
        rendered_object_face_normals_camera_coords = dibr_result.ren_face_normals.permute(0, 3, 1, 2).unsqueeze(0)

        rendering_result = RenderingResult(rendered_image=renderings,
                                           rendered_image_segmentation=segmentations,
                                           rendered_face_world_coords=rendered_object_world_coords,
                                           rendered_face_camera_coords=rendered_object_camera_coords,
                                           rendered_face_normals=rendered_object_face_normals_camera_coords)
        return rendering_result

    def compute_theoretical_flow(self, encoder_out_pose_2, encoder_out_pose_1, flow_arcs_indices) -> RenderedFlowResult:
        """
        Computes the theoretical flow between consecutive frames.

        Args:
            encoder_out_pose_2 (EncoderResult): The encoder result for the current frame.
            encoder_out_pose_1 (EncoderResult): The encoder results for the previous frames.
            flow_arcs_indices (Sorted[Tuple[int, int]]): Indexes in encoder_out_prev_frames and encoder_out given as a
                                                         sorted collection of tuples.

        Returns:
            torch.Tensor: The computed theoretical flow between consecutive frames. The output flow is respective to the
                          coordinates range [0, 1].
        """

        batches = self.rotations_translations_batched(encoder_out_pose_1, encoder_out_pose_2, flow_arcs_indices)
        (rotation_matrix_1_batch, rotation_matrix_2_batch,
         translation_vector_1_batch, translation_vector_2_batch) = batches

        batch_size = translation_vector_1_batch.shape[0]
        vertices_1 = encoder_out_pose_1.vertices

        batched_tensors = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1)
        camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch = batched_tensors

        batched_tensors_backview = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1,
                                                                               backview=self.backview)
        camera_rot_batch_backview, camera_trans_batch_backview, _, _, _ = batched_tensors_backview

        # Rotate and translate the vertices using the given rotation_matrix and translation_vector
        vertices_1_batch = kaolin.render.camera.rotate_translate_points(vertices_1_batch,
                                                                        rotation_matrix_1_batch, obj_center_batch)
        vertices_2_batch = kaolin.render.camera.rotate_translate_points(vertices_2_batch,
                                                                        rotation_matrix_2_batch, obj_center_batch)

        vertices_1_batch = vertices_1_batch + translation_vector_1_batch.unsqueeze(1)
        vertices_2_batch = vertices_2_batch + translation_vector_2_batch.unsqueeze(1)

        prepared_vertices_1 = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices_1_batch, faces=self.faces,
                                                                        camera_proj=self.camera_proj,
                                                                        camera_rot=camera_rot_batch_backview,
                                                                        camera_trans=camera_trans_batch_backview)
        face_vertices_cam_1, face_vertices_image_1, face_normals_1 = prepared_vertices_1

        prepared_vertices_2 = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices_2_batch, faces=self.faces,
                                                                        camera_proj=self.camera_proj,
                                                                        camera_rot=camera_rot_batch,
                                                                        camera_trans=camera_trans_batch)
        face_vertices_cam_2, face_vertices_image_2, face_normals_2 = prepared_vertices_2

        # Extract the z-coordinates of the face vertices in camera space
        face_vertices_z_1 = face_vertices_cam_1[:, :, :, -1]

        # Extract the z-components of the face normals
        face_normals_z_1 = face_normals_1[:, :, -1]
        # This implementation is correct, but due to low mesh resolution, it does not work
        face_occlusion_indication = 1. * (face_normals_2[:, :, -1] < 0)
        face_occlusion_indication_features = face_occlusion_indication[..., None, None].repeat(1, 1, 3, 1)

        face_vertices_image_motion = face_vertices_image_2 - face_vertices_image_1  # Vertices are in [-1, 1] range

        features_for_rendering = torch.cat([face_vertices_image_motion,
                                            face_occlusion_indication_features,
                                            face_vertices_cam_2], dim=-1).float()

        ren_outputs_1, ren_mask_1, red_index_1 = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                       face_vertices_z_1,
                                                                                       face_vertices_image_1,
                                                                                       features_for_rendering,
                                                                                       face_normals_z_1,
                                                                                       sigmainv=self.config.sigmainv,
                                                                                       boxlen=0.02, knum=30,
                                                                                       multiplier=1000)

        theoretical_flow = ren_outputs_1[..., :2]

        theoretical_flow[..., 0] = theoretical_flow[..., 0] * 0.5
        theoretical_flow[..., 1] = -theoretical_flow[..., 1] * 0.5  # Correction for transform into image

        theoretical_flow = theoretical_flow.permute(0, 3, 1, 2).unsqueeze(0)  # torch.Size([1, N, 2, H, W])
        flow_render_mask = ren_mask_1.unsqueeze(1).unsqueeze(0)  # torch.Size([1, N, 1, H, W])
        occlusion_mask = ren_outputs_1[..., 2].detach().unsqueeze(1).unsqueeze(0)  # torch.Size([1, N, 1, H, W])

        return RenderedFlowResult(theoretical_flow, flow_render_mask, occlusion_mask)

    def get_batched_tensors_for_flow_rendering(self, batch_size, vertices_1, backview=False):
        vertices_1_batch = vertices_1.repeat(batch_size, 1, 1)
        vertices_2_batch = vertices_1_batch
        obj_center_batch = self.obj_center.repeat(batch_size, 1)
        camera_rot_batch = self.camera_rot.repeat(batch_size, 1, 1)
        camera_trans_batch = self.camera_trans.repeat(batch_size, 1)

        if backview:
            camera_rot_batch[:] = self.camera_rot_backview
            camera_trans_batch[:] = self.camera_trans_backview

        return camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch

    def get_occlusion_mask_using_rendered_coordinates(self, rendered_pose1_with_pose2_coordinates,
                                                      rendered_pose2_with_pose2_coordinates, theoretical_flow):
        theoretical_flow_discrete = self.theoretical_flow_kaolin_to_image_warp(theoretical_flow)

        x_coords, x_coords_new, y_coords, y_coords_new = (
            self.get_original_and_warped_coordinates_from_flow(theoretical_flow_discrete))

        position_difference = (rendered_pose2_with_pose2_coordinates[0, y_coords_new, x_coords_new] -
                               rendered_pose1_with_pose2_coordinates[0, y_coords, x_coords])
        position_difference_norm = torch.linalg.vector_norm(position_difference, dim=-1)

        occlusion_mask = torch.zeros(1, 1, self.height, self.width).cuda()
        occlusion_mask[0, 0, y_coords, x_coords] = (position_difference_norm > 1e-1).float()

        return occlusion_mask

    def get_occlusion_mask_using_rendered_indices(self, rendered_pose1_with_pose2_indices,
                                                  rendered_pose2_with_pose2_indices, theoretical_flow):
        theoretical_flow_discrete = self.theoretical_flow_kaolin_to_image_warp(theoretical_flow)

        x_coords, x_coords_new, y_coords, y_coords_new = (
            self.get_original_and_warped_coordinates_from_flow(theoretical_flow_discrete))

        indices_eq_tensor = 1. * (rendered_pose2_with_pose2_indices == rendered_pose1_with_pose2_indices)

        occlusion_mask = torch.zeros(1, 1, self.height, self.width).cuda()
        occlusion_mask[0, 0, y_coords, x_coords] = indices_eq_tensor[0]

        return occlusion_mask

    def get_original_and_warped_coordinates_from_flow(self, theoretical_flow_discrete):
        x_coord_delta = theoretical_flow_discrete[..., 0]
        y_coord_delta = theoretical_flow_discrete[..., 1]
        x_coords, y_coords = torch.meshgrid(torch.arange(self.height), torch.arange(self.width))
        x_coords = x_coords.long().cuda()
        y_coords = y_coords.long().cuda()
        x_coords_new = torch.clamp(x_coords + x_coord_delta, 0, self.width).long()
        y_coord_new = torch.clamp(y_coords + y_coord_delta, 0, self.height).long()
        return x_coords, x_coords_new, y_coords, y_coord_new

    def theoretical_flow_kaolin_to_image_warp(self, theoretical_flow):
        theoretical_flow_discrete = theoretical_flow.clone()
        theoretical_flow_discrete[..., 0] *= self.height * 0.5
        theoretical_flow_discrete[..., 1] *= -self.width * 0.5
        theoretical_flow_discrete = theoretical_flow_discrete[0].long()
        return theoretical_flow_discrete

    def compute_theoretical_flow_using_rendered_vertices(self, rendering_result_frame_1: RenderingResult,
                                                         encoder_out_frame_2: EncoderResult,
                                                         encoder_out_frame_1: EncoderResult,
                                                         flow_arcs_indices,
                                                         ctx=None) -> RenderedFlowResult:

        rendered_vertices_frame_1 = rendering_result_frame_1.rendered_face_world_coords
        rendered_mask_frame_1 = rendering_result_frame_1.rendered_image_segmentation

        indices_pose_1_list = [frame_i_prev for frame_i_prev, _ in flow_arcs_indices]
        indices_pose_1 = torch.tensor(indices_pose_1_list, dtype=torch.long).cuda()
        rendered_vertices_frame_1_batched = torch.index_select(rendered_vertices_frame_1, 1, indices_pose_1)
        rendered_mask_frame_1_batched = torch.index_select(rendered_mask_frame_1, 1, indices_pose_1)

        batches = self.rotations_translations_batched(encoder_out_frame_1, encoder_out_frame_2, flow_arcs_indices)
        (rotation_matrix_1_batch, rotation_matrix_2_batch,
         translation_vector_1_batch, translation_vector_2_batch) = batches

        batch_size = translation_vector_1_batch.shape[0]
        vertices_1 = encoder_out_frame_1.vertices

        batched_tensors = self.get_batched_tensors_for_flow_rendering(batch_size, vertices_1)
        camera_rot_batch, camera_trans_batch, obj_center_batch, vertices_1_batch, vertices_2_batch = batched_tensors

        rendered_vertices_frame = rendered_vertices_frame_1_batched.permute(0, 1, 3, 4, 2)
        rendered_vertices_frame_norm = rendered_vertices_frame.norm(dim=-1)

        zero_vertices_positions = tuple(rendered_vertices_frame_norm.eq(0).nonzero().T)

        vertices_flattened = rendered_vertices_frame.flatten(start_dim=2, end_dim=-2)[0]

        vertices_1_nonzero = kaolin.render.camera.rotate_translate_points(vertices_flattened, rotation_matrix_1_batch,
                                                                          obj_center_batch)
        vertices_2_nonzero = kaolin.render.camera.rotate_translate_points(vertices_flattened, rotation_matrix_2_batch,
                                                                          obj_center_batch)

        vertices_1_nonzero = vertices_1_nonzero + translation_vector_1_batch.unsqueeze(1)
        vertices_2_nonzero = vertices_2_nonzero + translation_vector_2_batch.unsqueeze(1)

        vertices_1_camera = kaolin.render.camera.rotate_translate_points(vertices_1_nonzero, camera_rot_batch,
                                                                         camera_trans_batch)
        vertices_1_image = kaolin.render.camera.perspective_camera(vertices_1_camera, self.camera_proj)

        vertices_2_camera = kaolin.render.camera.rotate_translate_points(vertices_2_nonzero, camera_rot_batch,
                                                                         camera_trans_batch)
        vertices_2_image = kaolin.render.camera.perspective_camera(vertices_2_camera, self.camera_proj)

        vertices_flow = vertices_2_image - vertices_1_image

        if ctx is not None:
            ctx.x_world = rendered_vertices_frame_1_batched.detach().clone()

            vertices_1_camera_ren = vertices_1_camera.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_camera = vertices_1_camera_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

            vertices_1_image_ren = vertices_1_image.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_image = vertices_1_image_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

            vertices_2_image_ren = vertices_2_image.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1]))
            ctx.x_prime_image = vertices_2_image_ren.detach().clone().permute(0, 3, 1, 2).unsqueeze(0)

        theoretical_flow = vertices_flow.unflatten(dim=1, sizes=tuple(rendered_vertices_frame.shape[2:-1])).unsqueeze(0)

        theoretical_flow_new = theoretical_flow.clone()  # Create a new tensor with the same values
        theoretical_flow_new[..., 0] = theoretical_flow[..., 0] * 0.5
        theoretical_flow_new[..., 1] = -theoretical_flow[..., 1] * 0.5  # Correction for transform into image
        theoretical_flow_new[zero_vertices_positions].zero_()
        theoretical_flow = theoretical_flow_new.permute(0, 1, 4, 2, 3)  # torch.Size([1, N, H, W, 2])

        flow_segmentation = rendered_mask_frame_1_batched

        # TODO implement mock occlusion as real occlusion
        mock_occlusion = torch.zeros(flow_segmentation.shape).cuda()

        flow_result = RenderedFlowResult(theoretical_flow, flow_segmentation, mock_occlusion)

        return flow_result

    @staticmethod
    def render_flow_for_frame(renderer, encoder, flow_arc_source, flow_arc_target):
        keyframes = [flow_arc_source, flow_arc_target]
        flow_frames = [flow_arc_source, flow_arc_target]
        encoder_result, encoder_result_flow_frames = encoder.frames_and_flow_frames_inference(keyframes,
                                                                                              flow_frames)
        rendered_flow_res = renderer.compute_theoretical_flow(encoder_result, encoder_result_flow_frames,
                                                              flow_arcs_indices=[(0, 1)])
        return rendered_flow_res

    @staticmethod
    def rotations_translations_batched(encoder_out_frame_1, encoder_out_frame_2, flow_arcs_indices):

        indices_pose_1_list = [frame_i_prev for frame_i_prev, _ in flow_arcs_indices]
        indices_pose_2_list = [frame_i for _, frame_i in flow_arcs_indices]
        # Convert lists to tensors
        indices_pose_1 = torch.tensor(indices_pose_1_list, dtype=torch.long).cuda()
        indices_pose_2 = torch.tensor(indices_pose_2_list, dtype=torch.long).cuda()
        # Batch gather translations
        translation_vector_1_batch = torch.index_select(encoder_out_frame_1.translations, 2, indices_pose_1)[0, 0]
        translation_vector_2_batch = torch.index_select(encoder_out_frame_2.translations, 2, indices_pose_2)[0, 0]
        # Batch convert quaternions to rotation matrices
        quaternion_batch_1 = torch.index_select(encoder_out_frame_1.quaternions, 1, indices_pose_1)
        quaternion_batch_2 = torch.index_select(encoder_out_frame_2.quaternions, 1, indices_pose_2)

        rotation_matrix_1_batch = quaternion_to_rotation_matrix(quaternion_batch_1).to(torch.float)[0]
        rotation_matrix_2_batch = quaternion_to_rotation_matrix(quaternion_batch_2).to(torch.float)[0]

        return rotation_matrix_1_batch, rotation_matrix_2_batch, translation_vector_1_batch, translation_vector_2_batch

    def render_mesh_with_dibr(self, face_features, rotation_matrix, translation_vector, unit_vertices) \
            -> MeshRenderResult:
        # Rotate and translate the vertices using the given rotation_matrix and translation_vector
        vertices = kaolin.render.camera.rotate_translate_points(unit_vertices, rotation_matrix, self.obj_center)

        # Apply the translation to the vertices
        vertices = vertices + translation_vector.unsqueeze(1)

        # Prepare the vertices for rendering by computing their camera coordinates, image coordinates, and face normals
        camera_rot_batched = self.camera_rot.repeat(translation_vector.shape[0], 1, 1)
        camera_trans_batched = self.camera_trans.repeat(translation_vector.shape[0], 1)
        if self.backview:
            camera_rot_batched[0] = self.camera_rot_backview[0]
            camera_trans_batched[0] = self.camera_trans_backview[0]

        prepared_vertices = kaolin.render.mesh.utils.prepare_vertices(vertices=vertices,
                                                                      faces=self.faces,
                                                                      camera_rot=camera_rot_batched,
                                                                      camera_trans=camera_trans_batched,
                                                                      camera_proj=self.camera_proj)

        vertices_world_coordinates = kaolin.ops.mesh.index_vertices_by_faces(unit_vertices, self.faces)

        face_vertices_cam, face_vertices_image, face_normals = prepared_vertices
        face_normals_feature = face_normals.unsqueeze(-1)

        # Extract the z-coordinates of the face vertices in camera space
        face_vertices_z = face_vertices_cam[:, :, :, -1]

        # Extract the z-components of the face normals
        face_normals_z = face_normals[:, :, -1]

        features_for_rendering = torch.cat((face_features,
                                            face_vertices_cam,
                                            face_vertices_image,
                                            vertices_world_coordinates,
                                            face_normals_feature), dim=-1)

        # Perform dibr rasterization
        ren_outputs, ren_mask, red_index = kaolin.render.mesh.dibr_rasterization(self.height, self.width,
                                                                                 face_vertices_z,
                                                                                 face_vertices_image,
                                                                                 features_for_rendering,
                                                                                 face_normals_z,
                                                                                 sigmainv=self.config.sigmainv,
                                                                                 boxlen=0.02, knum=30, multiplier=1000)

        # Extract ren_mesh_vertices_features and ren_mesh_vertices_coords from the combined output tensor
        split_tuple = torch.split(ren_outputs, [face_features.shape[-1],
                                                face_vertices_cam.shape[-1],
                                                face_vertices_image.shape[-1],
                                                vertices_world_coordinates.shape[-1],
                                                face_normals_feature.shape[-1]], dim=-1)

        (ren_mesh_vertices_features, ren_mesh_vertices_camera_coords, ren_mesh_vertices_image_coords,
         ren_mesh_vertices_world_coords, ren_face_normals_features) = split_tuple

        return MeshRenderResult(face_normals, face_vertices_cam, red_index, ren_mask,
                                ren_mesh_vertices_features,
                                ren_mesh_vertices_world_coords,
                                ren_mesh_vertices_camera_coords,
                                ren_mesh_vertices_image_coords,
                                ren_face_normals_features)

    def get_rgb_texture(self, translation, quaternion, unit_vertices, face_features, input_batch):
        kernel = torch.ones(self.config.erode_renderer_mask, self.config.erode_renderer_mask).to(
            translation.device)
        tex = torch.zeros(1, 3, self.config.texture_size, self.config.texture_size)
        cnt = torch.zeros(self.config.texture_size, self.config.texture_size)
        for frmi in range(quaternion.shape[1]):
            translation_vector = translation[:, :, frmi]
            rotation_matrix = quaternion_to_rotation_matrix(quaternion[:, frmi])

            rendering_result = self.render_mesh_with_dibr(face_features, rotation_matrix, translation_vector,
                                                          unit_vertices)

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


def infer_normalized_renderings(renderer: RenderingKaolin, encoder_face_features, encoder_result,
                                encoder_result_flow_frames, flow_arcs_indices, input_image_width, input_image_height) \
        -> Tuple[torch.Tensor, torch.Tensor, RenderedFlowResult]:
    rendering_result = renderer.forward(encoder_result.translations, encoder_result.quaternions,
                                        encoder_result.vertices, encoder_face_features, encoder_result.texture_maps,
                                        encoder_result.lights)

    rendering = rendering_result.rendered_image
    rendering_mask = rendering_result.rendered_image_segmentation

    flow_result = renderer.compute_theoretical_flow(encoder_result, encoder_result_flow_frames, flow_arcs_indices)

    print("Flow time: ", time() - start_time)

    theoretical_flow, rendered_flow_segmentation, occlusion_masks = flow_result

    # Renormalization compensating for the fact that we render into bounding box that is smaller than the actual image
    normalized_theoretical_flow = normalize_rendered_flows(theoretical_flow, renderer.width, renderer.height,
                                                           input_image_width, input_image_height)
    flow_result = flow_result._replace(theoretical_flow=normalized_theoretical_flow)

    return rendering, rendering_mask, flow_result


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

from collections import defaultdict

from typing import Tuple, List, Any, Optional

import networkx as nx
import pycolmap
import torch
import imageio
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import seaborn as sns
import torchvision
from PIL import Image
from kornia.geometry import Se3, Quaternion
from kornia.image import ImageSize
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Patch
from pathlib import Path
from kornia.geometry.conversions import quaternion_to_axis_angle

from auxiliary_scripts.data_utils import load_texture, load_mesh_using_trimesh
from auxiliary_scripts.image_utils import overlay_occlusion
from data_structures.datagraph_utils import get_relative_gt_obj_rotation
from data_structures.pose_icosphere import PoseIcosphere
from data_structures.rerun_annotations import RerunAnnotations
from tracker_config import TrackerConfig
from data_structures.data_graph import DataGraph
from utils import normalize_vertices, extract_intrinsics_from_tensor
from auxiliary_scripts.math_utils import Se3_last_cam_to_world_from_Se3_obj, Se3_epipolar_cam_from_Se3_obj
from flow import (visualize_flow_with_images, flow_unit_coords_to_image_coords, source_coords_to_target_coords_image,
                  source_coords_to_target_coords, source_coords_to_target_coords_np)


class WriteResults:

    def __init__(self, write_folder, shape: ImageSize, tracking_config: TrackerConfig, data_graph: DataGraph,
                 pose_icosphere: PoseIcosphere, images_paths, segmentation_paths, Se3_world_to_cam: Se3):

        self.image_height = shape.height
        self.image_width = shape.width

        self.images_paths: Optional[List] = images_paths
        self.segmentation_paths: Optional[List] = segmentation_paths

        self.data_graph: DataGraph = data_graph

        self.pose_icosphere: PoseIcosphere = pose_icosphere

        self.logged_flow_tracks_inits: List = list()

        self.tracking_config: TrackerConfig = tracking_config

        self.write_folder = Path(write_folder)

        self.observations_path = self.write_folder / Path('images')
        self.segmentation_path = self.write_folder / Path('segments')
        self.ransac_path = self.write_folder / Path('ransac')
        self.exported_mesh_path = self.write_folder / Path('3d_model')

        self.Se3_world_to_cam: Se3 = Se3_world_to_cam

        self.init_directories()

        self.template_fields: List[str] = []

        self.rerun_init()

    def init_directories(self):
        if not self.tracking_config.write_to_rerun_rather_than_disk:
            self.observations_path.mkdir(exist_ok=True, parents=True)
            self.segmentation_path.mkdir(exist_ok=True, parents=True)
            self.ransac_path.mkdir(exist_ok=True, parents=True)
            self.exported_mesh_path.mkdir(exist_ok=True, parents=True)

    def rerun_init(self):
        rr.init(f'{self.tracking_config.sequence}-{self.tracking_config.experiment_name}')
        rerun_file = (self.write_folder /
                      f'rerun_{self.tracking_config.experiment_name}_{self.tracking_config.sequence}.rrd')
        rr.save(rerun_file)

        self.template_fields = {
            RerunAnnotations.chained_pose_polar_template,
            RerunAnnotations.chained_pose_long_flow_template,
            RerunAnnotations.chained_pose_short_flow_template,

            RerunAnnotations.cam_delta_r_short_flow_template,
            RerunAnnotations.cam_delta_t_short_flow_template,
            RerunAnnotations.cam_delta_r_long_flow_template,
            RerunAnnotations.cam_delta_t_long_flow_template,

            RerunAnnotations.long_short_chain_diff_template,

            RerunAnnotations.cam_rot_ref_to_last_template,
            RerunAnnotations.cam_tran_ref_to_last_template,
            RerunAnnotations.obj_rot_ref_to_last_template,
            RerunAnnotations.obj_tran_ref_to_last_template,

            RerunAnnotations.translation_scale
        }

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                contents=[
                    rrb.Vertical(
                        contents=[
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(name="Observed Flow Occlusion",
                                                      origin=RerunAnnotations.observed_flow_frontview),
                                    # rrb.Spatial2DView(name="Observed Flow Uncertainty",
                                    #                   origin=RerunAnnotations.observed_flow_with_uncertainty_frontview),
                                    # rrb.Spatial2DView(name="Observed Flow GT Disparity",
                                    #                   origin=RerunAnnotations.observed_flow_errors_frontview),
                                ],
                                name='Flows'
                            ),
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(name="Template Image Current",
                                                      origin=RerunAnnotations.template_image_frontview),
                                    rrb.Spatial2DView(name="Observed Image",
                                                      origin=RerunAnnotations.observed_image_frontview),
                                ],
                                name='Observed Images'
                            )
                        ],
                        name='Observed Input'
                    ),
                    rrb.Vertical(
                        contents=[
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(name="Template Image Current",
                                                      origin=RerunAnnotations.template_image_frontview),
                                    rrb.Spatial2DView(name="Observed Image",
                                                      origin=RerunAnnotations.observed_image_frontview),
                                ],
                                name='Observed Images'
                            ),
                            rrb.Grid(
                                contents=[
                                    rrb.Spatial2DView(name=f"Template {i}",
                                                      origin=f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}')
                                    for i in range(27)
                                ],
                                grid_columns=9,
                                name='Templates'
                            ),
                        ],
                        name='Templates'
                    ),
                    rrb.Spatial3DView(
                        origin=RerunAnnotations.space_visualization,
                        name='3D Space',
                        background=[255, 255, 255]
                    ),
                    rrb.Spatial3DView(
                        origin=RerunAnnotations.colmap_visualization,
                        name='COLMAP',
                        background=[255, 255, 255]
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.TimeSeriesView(name="Pose Estimation (w.o. flow computation)",
                                               origin=RerunAnnotations.pose_estimation_timing),
                        ],
                        grid_columns=2,
                        name='Timings'
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.Spatial3DView(name="Triangulated Point Cloud MFT Flow, GT, Rt",
                                              origin=RerunAnnotations.triangulated_points_gt_Rt_mft_flow),
                            rrb.Spatial3DView(name="Triangulated Point Cloud GT Flow, GT, Rt",
                                              origin=RerunAnnotations.triangulated_points_gt_Rt_gt_flow),
                            rrb.Spatial3DView(name="Point Cloud Dust3r, Image 1",
                                              origin=RerunAnnotations.point_cloud_dust3r_im1),
                            rrb.Spatial3DView(name="Point Cloud Dust3r, Image 2",
                                              origin=RerunAnnotations.point_cloud_dust3r_im2)
                        ],
                        grid_columns=2,
                        name='Point Clouds'
                    ),
                    rrb.Tabs(
                        rrb.Horizontal(
                            contents=[
                                rrb.Spatial2DView(name="RANSAC Inliers Visualization",
                                                  origin=RerunAnnotations.matching_correspondences_inliers),
                                rrb.Spatial2DView(name="RANSAC Outliers Visualizations",
                                                  origin=RerunAnnotations.matching_correspondences_outliers),
                            ],
                            name='Matching - Long Jumps'
                        ),
                        name='Matching'
                    ),
                    rrb.Tabs(
                        contents=[
                            rrb.Grid(
                                contents=[
                                    rrb.TimeSeriesView(name="RANSAC - Frontview",
                                                       origin=RerunAnnotations.ransac_stats
                                                       ),
                                    rrb.TimeSeriesView(name="Pose - Rotation",
                                                       origin=RerunAnnotations.obj_rot_1st_to_last
                                                       ),
                                    rrb.TimeSeriesView(name="Pose - Translation",
                                                       origin=RerunAnnotations.obj_tran_1st_to_last
                                                       ),
                                ],
                                grid_columns=2,
                                name='Epipolar'
                            ),
                            rrb.Grid(
                                contents=[
                                    rrb.TimeSeriesView(name="Chained Pose Long Flow",
                                                       origin=RerunAnnotations.chained_pose_long_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Chained Pose Short Flow",
                                                       origin=RerunAnnotations.chained_pose_short_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Chained Pose Polar Angles",
                                                       origin=RerunAnnotations.chained_pose_polar
                                                       ),
                                    rrb.TimeSeriesView(name="Short Flow Chaining Filtering - % Remaining in Long Jump",
                                                       origin=RerunAnnotations.long_short_chain_remaining_pts
                                                       ),
                                ],
                                grid_columns=2,
                                name='Flow Chaining'
                            ),
                            rrb.Grid(
                                contents=[
                                    rrb.TimeSeriesView(name="Cam Rot Delta Short Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_r_short_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Cam Rot Delta Long Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_r_long_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Cam Tran Delta Short Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_t_short_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Cam Tran Delta Long Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_t_long_flow
                                                       ),
                                ],
                                grid_columns=2,
                                name='RANSAC pose vs Zaragoza pose'
                            ),
                            rrb.Grid(
                                contents=[
                                    rrb.TimeSeriesView(name="Camera Rotation Ref -> Last",
                                                       origin=RerunAnnotations.cam_rot_ref_to_last
                                                       ),
                                    rrb.TimeSeriesView(name="Camera Translation Ref -> Last",
                                                       origin=RerunAnnotations.cam_tran_ref_to_last
                                                       ),
                                    rrb.TimeSeriesView(name="Object Rotation Ref -> Last",
                                                       origin=RerunAnnotations.obj_rot_ref_to_last
                                                       ),
                                    rrb.TimeSeriesView(name="Object Translation Ref -> Last",
                                                       origin=RerunAnnotations.obj_tran_ref_to_last
                                                       ),
                                ],
                                grid_columns=2,
                                name='Pose'
                            ),
                            rrb.Grid(
                                contents=[
                                    rrb.TimeSeriesView(name="Translation Per Axis GT Scale",
                                                       origin=RerunAnnotations.translation_scale
                                                       ),
                                ],
                                grid_columns=1,
                                name='Translation Scaling'
                            ),
                        ],
                        name='Pose'
                    ),
                ],
                name=f'Results - {self.tracking_config.sequence}'
            )
        )

        axes_colors = {
            'x': (0, 127, 0),
            'y': (0, 51, 102),
            'z': (102, 0, 102),
        }

        gt_axes_colors = {
            'x': (0, 255, 0),
            'y': (102, 178, 255),
            'z': (255, 155, 255),
        }

        annotations = set()
        for axis, c in axes_colors.items():
            annotations |= set(map(
                lambda annotation: (annotation, c),
                [
                    RerunAnnotations.obj_tran_1st_to_last_axes[axis],
                    RerunAnnotations.obj_tran_ref_to_last_axes[axis],
                    RerunAnnotations.cam_tran_ref_to_last_axes[axis],
                    RerunAnnotations.obj_rot_1st_to_last_axes[axis],
                    RerunAnnotations.obj_rot_ref_to_last_axes[axis],
                    RerunAnnotations.cam_rot_ref_to_last_axes[axis],
                    RerunAnnotations.chained_pose_long_flow_axes[axis],
                    RerunAnnotations.chained_pose_short_flow_axes[axis],
                    RerunAnnotations.translation_scale_gt_axes[axis],
                ]
                ))

        for axis, c in gt_axes_colors.items():
            annotations |= set(map(
                lambda annotation: (annotation, c),
                [
                    RerunAnnotations.obj_rot_1st_to_last_gt_axes[axis],
                    RerunAnnotations.obj_rot_ref_to_last_gt_axes[axis],
                    RerunAnnotations.cam_rot_ref_to_last_gt_axes[axis],
                    RerunAnnotations.obj_tran_1st_to_last_gt_axes[axis],
                    RerunAnnotations.obj_tran_ref_to_last_gt_axes[axis],
                    RerunAnnotations.cam_tran_ref_to_last_gt_axes[axis],
                ]
                ))

            for rerun_annotation, color in annotations:
                rr.log(rerun_annotation, rr.SeriesLine(color=color,
                                                       name=rerun_annotation.split('/')[-1]), static=True)

        for template_annotation in self.template_fields:
            rr.log(template_annotation,
                   rr.SeriesPoint(
                       color=[255, 0, 0],
                       name="new template",
                       marker="circle",
                       marker_size=4,
                   ),
                   static=True)

        rr.send_blueprint(blueprint)

    @torch.no_grad()
    def write_results(self, frame_i):

        self.visualize_observed_data(frame_i)

        if not self.tracking_config.write_to_rerun_rather_than_disk:
            self.visualize_flow_with_matching(frame_i)

        if self.tracking_config.visualize_outliers_distribution:
            datagraph_camera_data = self.data_graph.get_frame_data(frame_i)
            new_flow_arc = (datagraph_camera_data.long_jump_source, frame_i)
            self.visualize_outliers_distribution(new_flow_arc)

        self.visualize_3d_camera_space(frame_i)

        if self.tracking_config.write_to_rerun_rather_than_disk:
            # self.log_poses_into_rerun(frame_i)
            # self.visualize_flow_with_matching_rerun(frame_i)
            pass

        if self.tracking_config.preinitialization_method == 'essential_matrix_decomposition':
            if self.tracking_config.analyze_ransac_matching_errors:
                self.analyze_ransac_matchings_errors(frame_i)

            if (self.tracking_config.analyze_ransac_matchings and
                    frame_i % self.tracking_config.analyze_ransac_matchings_frequency == 0):
                # self.analyze_ransac_matchings(frame_i)
                pass

    def visualize_colmap_track(self, frame_i: int, colmap_reconstruction: pycolmap.Reconstruction):
        rr.set_time_sequence(RerunAnnotations.space_visualization, 0)

        points_3d_coords = np.stack([p.xyz for p in colmap_reconstruction.points3D.values()], axis=0)
        points_3d_colors = np.stack([p.color for p in colmap_reconstruction.points3D.values()], axis=0)
        rr.log(RerunAnnotations.colmap_pointcloud, rr.Points3D(points_3d_coords, colors=points_3d_colors))

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        T_world_to_cam_se3_batched = Se3.from_matrix(self.Se3_world_to_cam.matrix().repeat(n_poses, 1, 1))

        gt_rotations, gt_translations, rotations, translations = self.read_poses_from_datagraph(all_frames_from_0)
        gt_rotations_rad = torch.deg2rad(gt_rotations)

        gt_obj_se3 = Se3(Quaternion.from_axis_angle(gt_rotations_rad), gt_translations)
        gt_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(gt_obj_se3, T_world_to_cam_se3_batched)
        gt_t_cam = gt_cam_se3.translation.numpy(force=True)

        cmap_gt = plt.get_cmap('Reds')
        gradient = np.linspace(1., 0.5, self.tracking_config.input_frames)
        colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)

        strips_gt = np.stack([gt_t_cam[:-1], gt_t_cam[1:]], axis=1)

        strips_radii_factor = (max(torch.max(torch.cat([translations, gt_translations]).norm(dim=1)).item(), 5.) / 5.)
        strips_radii = [0.01 * strips_radii_factor] * n_poses

        rr.log(RerunAnnotations.colmap_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt,  # gt_t_cam
                               colors=colors_gt,
                               radii=strips_radii))

        datagraph_camera_node = self.data_graph.get_frame_data(frame_i)
        template_frame_idx = datagraph_camera_node.long_jump_source

        datagraph_template_node = self.data_graph.get_frame_data(template_frame_idx)

        template_node_Se3 = datagraph_template_node.predicted_object_se3_total
        template_node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(template_node_Se3, self.Se3_world_to_cam)

        image_id_to_poses = {}
        image_name_to_image_id = {image.name: image_id for image_id, image in colmap_reconstruction.images.items()}

        G_reliable = nx.Graph()

        for image_id, image in colmap_reconstruction.images.items():
            frame_index = self.images_paths.index(Path(image.name))

            image_t_cam = torch.tensor(image.cam_from_world.translation)
            image_q_cam_xyzw = torch.tensor(image.cam_from_world.rotation.quat)
            image_q_cam_wxyz = image_q_cam_xyzw[[3, 0, 1, 2]]
            image_Se3_cam = Se3(Quaternion(image_q_cam_wxyz), image_t_cam)

            image_id_to_poses[image_id] = image_t_cam

            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Transform3D(translation=image_t_cam,
                               rotation=rr.Quaternion(xyzw=image_q_cam_xyzw),
                               from_parent=True)
            )

            camera_params = colmap_reconstruction.cameras[1]
            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Pinhole(resolution=[camera_params.width, camera_params.height],
                           focal_length=[camera_params.params[0], camera_params.params[0]],
                           camera_xyz=None  # rr.ViewCoordinates.RUB
                           ),
            )

            frame_node = self.data_graph.get_frame_data(frame_index)

            for reliable_node_idx in frame_node.reliable_sources:
                reliable_node_data = self.data_graph.get_frame_data(reliable_node_idx)
                reliable_node_name = Path(reliable_node_data.image_filename).name

                reliable_node_image_id = image_name_to_image_id[reliable_node_name]

                G_reliable.add_edge(image_id, reliable_node_image_id)

        strips = []
        for im_id1, im_id2 in G_reliable.edges:
            im1_t = image_id_to_poses[im_id1]
            im2_t = image_id_to_poses[im_id2]

            strips.append([im1_t, im2_t])

        rr.log(
            RerunAnnotations.colmap_predicted_line_strips_reliable,
            rr.LineStrips3D(
                strips,
                colors=[[255, 0, 0] ] * len(strips),
                radii=[0.005] * len(strips),
            )
        )



    def visualize_3d_camera_space(self, frame_i: int):

        rr.set_time_sequence(RerunAnnotations.space_visualization, frame_i)

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        T_world_to_cam_se3 = self.Se3_world_to_cam
        T_world_to_cam_se3_batched = Se3.from_matrix(T_world_to_cam_se3.matrix().repeat(n_poses, 1, 1))

        gt_rotations, gt_translations, rotations, translations = self.read_poses_from_datagraph(all_frames_from_0)
        gt_rotations_rad = torch.deg2rad(gt_rotations)
        rotations_rad = torch.deg2rad(rotations)

        gt_obj_se3 = Se3(Quaternion.from_axis_angle(gt_rotations_rad), gt_translations)
        pred_obj_se3 = Se3(Quaternion.from_axis_angle(rotations_rad), translations)

        if (frame_i == 1 and self.tracking_config.gt_mesh_path is not None
                and self.tracking_config.gt_texture_path is not None):
            gt_texture = load_texture(Path(self.tracking_config.gt_texture_path),
                                      self.tracking_config.texture_size)
            gt_texture_int = (gt_texture[0].permute(1, 2, 0) * 255).to(torch.uint8)

            gt_mesh = load_mesh_using_trimesh(Path(self.tracking_config.gt_mesh_path))

            normalized_vertices = normalize_vertices(torch.Tensor(gt_mesh.vertices))

            vertex_texcoords = gt_mesh.visual.uv
            vertex_texcoords[:, 1] = 1.0 - vertex_texcoords[:, 1]

            rr.log(
                RerunAnnotations.space_gt_mesh,
                rr.Mesh3D(
                    triangle_indices=gt_mesh.faces,
                    albedo_texture=gt_texture_int,
                    vertex_texcoords=vertex_texcoords,
                    vertex_positions=normalized_vertices
                )
            )

        gt_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(gt_obj_se3, T_world_to_cam_se3_batched)
        pred_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(pred_obj_se3, T_world_to_cam_se3_batched)

        q_cam_gt_xyzw = gt_cam_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        q_cam_xyzw = pred_cam_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        gt_t_cam = gt_cam_se3.translation.numpy(force=True)
        pred_t_cam = pred_cam_se3.translation.numpy(force=True)

        rr.set_time_sequence('frame', frame_i)

        rr.log(
            RerunAnnotations.space_predicted_camera_pose,
            rr.Transform3D(translation=pred_t_cam[-1],
                           rotation=rr.Quaternion(xyzw=q_cam_xyzw[-1]),
                           )
        )
        rr.log(
            RerunAnnotations.space_gt_camera_pose,
            rr.Transform3D(translation=gt_t_cam[-1],
                           rotation=rr.Quaternion(xyzw=q_cam_gt_xyzw[-1]),
                           )
        )

        cmap_gt = plt.get_cmap('Greens')
        cmap_pred = plt.get_cmap('Blues')
        gradient = np.linspace(1., 0., self.tracking_config.input_frames)
        colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)
        colors_pred = (np.asarray([cmap_pred(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)

        strips_gt = np.stack([gt_t_cam[:-1], gt_t_cam[1:]], axis=1)
        strips_pred = np.stack([pred_t_cam[:-1], pred_t_cam[1:]], axis=1)

        strips_radii_factor = (max(torch.max(torch.cat([translations, gt_translations]).norm(dim=1)).item(), 5.) / 5.)
        strips_radii = [0.01 * strips_radii_factor] * n_poses

        rr.log(RerunAnnotations.space_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt,  # gt_t_cam
                               colors=colors_gt,
                               radii=strips_radii))

        rr.log(RerunAnnotations.space_predicted_camera_track,
               rr.LineStrips3D(strips=strips_pred,  # pred_t_cam
                               colors=colors_pred,
                               radii=strips_radii))

        datagraph_camera_node = self.data_graph.get_frame_data(frame_i)
        template_frame_idx = datagraph_camera_node.long_jump_source

        datagraph_template_node = self.data_graph.get_frame_data(template_frame_idx)

        template_node_Se3 = datagraph_template_node.predicted_object_se3_total
        template_node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(template_node_Se3, T_world_to_cam_se3)

        rr.log(RerunAnnotations.space_predicted_closest_keypoint,
               rr.LineStrips3D(strips=[[pred_t_cam[-1],
                                        template_node_cam_se3.translation.squeeze().numpy(force=True)]],
                               colors=[[255, 0, 0]],
                               radii=[0.025 * strips_radii_factor]))

        if len(datagraph_camera_node.reliable_sources) > 1:
            for reliable_template_idx in datagraph_camera_node.reliable_sources:
                datagraph_template_node = self.data_graph.get_frame_data(reliable_template_idx)

                template_node_Se3 = datagraph_template_node.predicted_object_se3_total
                template_node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(template_node_Se3, T_world_to_cam_se3)

                rr.log(f'{RerunAnnotations.space_predicted_reliable_templates}/{reliable_template_idx}',
                       rr.LineStrips3D(strips=[[pred_t_cam[-1],
                                                template_node_cam_se3.translation.squeeze().numpy(force=True)]],
                                       colors=[[255, 255, 0]],
                                       radii=[0.025 * strips_radii_factor]))

        for i, icosphere_node in enumerate(self.pose_icosphere.reference_poses):

            if icosphere_node.keyframe_idx_observed not in self.logged_flow_tracks_inits:
                template_idx = len(self.logged_flow_tracks_inits)

                template = icosphere_node.observation.observed_image[0, 0].permute(1, 2, 0).numpy(force=True)

                self.logged_flow_tracks_inits.append(icosphere_node.keyframe_idx_observed)
                template_image_grid_annotation = (f'{RerunAnnotations.space_predicted_camera_keypoints}/'
                                                  f'{template_idx}')
                rr.log(template_image_grid_annotation, rr.Image(template))

                for template_annotation in self.template_fields:
                    rr.log(template_annotation, rr.Scalar(0.0))

                node_Se3 = Se3(icosphere_node.quaternion, icosphere_node.translation)
                node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(node_Se3, T_world_to_cam_se3)
                node_cam_q_xyzw = node_cam_se3.quaternion.q[:, [1, 2, 3, 0]]

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Transform3D(translation=node_cam_se3.translation.squeeze().numpy(force=True),
                                   rotation=rr.Quaternion(xyzw=node_cam_q_xyzw.squeeze().numpy(force=True)))
                )
                frame_data = self.data_graph.get_frame_data(icosphere_node.keyframe_idx_observed)
                fx, fy, cx, cy = extract_intrinsics_from_tensor(frame_data.gt_pinhole_K)

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Pinhole(
                        resolution=[self.image_width, self.image_height],
                        focal_length=[float(fx.item()),
                                      float(fy.item())],
                        camera_xyz=rr.ViewCoordinates.RUB,
                    ),
                )

    @staticmethod
    def write_obj_mesh(vertices, faces, face_features, name, materials_model_name=None):
        if materials_model_name is None:
            materials_model_name = "model.mtl"

        file = open(name, "w")
        file.write("mtllib " + materials_model_name + "\n")
        file.write("o FMO\n")
        for ver in vertices:
            file.write("v {:.6f} {:.6f} {:.6f} \n".format(ver[0], ver[1], ver[2]))
        for ffeat in face_features:
            for feat in ffeat:
                if len(feat) == 3:
                    file.write("vt {:.6f} {:.6f} {:.6f} \n".format(feat[0], feat[1], feat[2]))
                else:
                    file.write("vt {:.6f} {:.6f} \n".format(feat[0], feat[1]))
        file.write("usemtl Material.002\n")
        file.write("s 1\n")
        for fi in range(faces.shape[0]):
            fc = faces[fi] + 1
            ti = 3 * fi + 1
            file.write("f {}/{} {}/{} {}/{}\n".format(fc[0], ti, fc[1], ti + 1, fc[2], ti + 2))
        file.close()

    def measure_ransac_stats(self, frame_i):
        correct_threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold
        results = defaultdict(list)

        for i in range(1, frame_i + 1):
            flow_arc_source = self.data_graph.get_frame_data(i).long_jump_source
            flow_arc = (flow_arc_source, i)

            arc_data = self.data_graph.get_edge_observations(*flow_arc)

            pred_inlier_ratio = arc_data.ransac_inlier_ratio
            inlier_mask = arc_data.ransac_inliers_mask

            observed_flow_image = flow_unit_coords_to_image_coords(arc_data.observed_flow.observed_flow)
            gt_flow_image = flow_unit_coords_to_image_coords(arc_data.synthetic_flow_result.observed_flow)

            src_pts_pred_visible_yx = arc_data.observed_visible_fg_points_mask.nonzero()
            dst_pts_pred_visible_yx = source_coords_to_target_coords(src_pts_pred_visible_yx, observed_flow_image)
            dst_pts_pred_visible_yx_gt = source_coords_to_target_coords(src_pts_pred_visible_yx, gt_flow_image)

            correct_flows_epe = torch.linalg.norm(dst_pts_pred_visible_yx - dst_pts_pred_visible_yx_gt, dim=1)

            correct_flows = (correct_flows_epe < correct_threshold)

            dst_pts_pred_visible_yx_small_errors = arc_data.dst_pts_yx
            dst_pts_pred_visible_yx_gt_small_errors = arc_data.dst_pts_yx_gt

            inliers_errors = torch.linalg.norm(arc_data.dst_pts_yx[inlier_mask] -
                                               arc_data.dst_pts_yx_gt[inlier_mask], dim=-1)
            correct_inliers = inliers_errors < correct_threshold

            fg_points_num = float(arc_data.adjusted_segmentation.sum()) + 1e-5
            pred_visible_num = float(arc_data.observed_visible_fg_points_mask.sum())
            correct_flows_num = float(correct_flows.sum())
            predicted_inliers_num = float(inlier_mask.sum())
            correct_inliers_num = float(correct_inliers.sum())
            actually_visible_num = float(arc_data.gt_visible_fg_points_mask.sum())

            # results['foreground_points'].append(fg_points_num / fg_points_num)
            results['visible'].append(actually_visible_num / fg_points_num)
            results['predicted_as_visible'].append(pred_visible_num / fg_points_num)
            results['correctly_predicted_flows'].append(correct_flows_num / fg_points_num)
            results['ransac_predicted_inliers'].append(predicted_inliers_num / fg_points_num)
            results['correctly_predicted_inliers'].append(correct_inliers_num / fg_points_num)
            results['ransac_inlier_ratio'].append(pred_inlier_ratio)
            results['mft_flow_gt_flow_difference'].append(dst_pts_pred_visible_yx_small_errors -
                                                          dst_pts_pred_visible_yx_gt_small_errors)

        return results

    def analyze_ransac_matchings_errors(self, frame_i):

        if (frame_i >= 5 and frame_i % 5 == 0) or frame_i >= self.tracking_config.input_frames:

            front_results = self.measure_ransac_stats(frame_i)

            mft_flow_gt_flow_difference_front = front_results.pop('mft_flow_gt_flow_difference')

            if self.tracking_config.plot_mft_flow_kde_error_plot:
                self.plot_distribution_of_inliers_errors(mft_flow_gt_flow_difference_front)

    def analyze_ransac_matchings(self, frame_i):

        ransac_stats = self.measure_ransac_stats(frame_i)

        ransac_stats.pop('mft_flow_gt_flow_difference')

        # We want each line to have its assigned color
        for i, metric in enumerate(ransac_stats.keys()):

            rerun_time_series_entity = getattr(RerunAnnotations, f'ransac_stats_{metric}')

            rr.set_time_sequence("frame", frame_i)
            metric_val: float = ransac_stats[metric][-1]
            rr.log(rerun_time_series_entity, rr.Scalar(metric_val))

    def plot_distribution_of_inliers_errors(self, mft_flow_gt_flow_differences):
        sns.set(style="whitegrid")

        frame = len(mft_flow_gt_flow_differences) - 1

        if frame + 1 % self.tracking_config.mft_flow_kde_error_plot_frequency != 0:
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        gt_rotation: np.ndarray = (torch.rad2deg(self.data_graph.get_frame_data(frame).gt_rot_axis_angle)[0].
                                   numpy(force=True))
        gt_rotation = np.round(gt_rotation, 2)

        data_tensor: torch.Tensor = mft_flow_gt_flow_differences[frame].numpy(force=True)

        sns.kdeplot(
            x=data_tensor[:, 1], y=data_tensor[:, 0],
            thresh=0, levels=20, fill=False, cmap="mako", ax=ax
        )
        ax.set_xlabel("X Axis Error [px]")
        ax.set_ylabel("Y Axis Error [px]")
        ax.set_title(f'Distribution of MFT Errors,\nFrame {frame} (gt rotation [X, Y, Z] {gt_rotation})')

        # Adjust layout for better fit and display the plot
        plt.savefig(self.ransac_path / f'inliers_errors_frame_{frame}.svg')
        plt.close()

    def visualize_flow_with_matching(self, frame_i):
        if self.tracking_config.preinitialization_method != 'essential_matrix_decomposition':
            return

        dpi = 600

        datagraph_frontview_data = self.data_graph.get_frame_data(frame_i)
        new_flow_arcs = [(datagraph_frontview_data.long_jump_source, frame_i)]

        for new_flow_arc in new_flow_arcs:

            flow_arc_source, flow_arc_target = new_flow_arc

            fig, axs = plt.subplots(3, 1, figsize=(8, 12), dpi=600)
            axs: Any = axs

            flow_source_label = self.tracking_config.gt_flow_source
            if flow_source_label == 'FlowNetwork':
                flow_source_label = self.tracking_config.long_flow_model

            heading_text = (f"Frames {new_flow_arc}\n"
                            f"Flow: {flow_source_label}")

            axs = np.atleast_2d(axs).T

            axs[0, 0].text(1.05, 1, heading_text, transform=axs[0, 0].transAxes, verticalalignment='top',
                           fontsize='medium')

            arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

            rendered_flow_res = arc_observation.synthetic_flow_result

            rend_flow = flow_unit_coords_to_image_coords(rendered_flow_res.observed_flow)
            rend_flow_np = rend_flow.numpy(force=True)

            flow_observation = arc_observation.observed_flow
            opt_flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)
            occlusion_mask = self.convert_observation_to_numpy(flow_observation.observed_flow_occlusion)
            occlusion_mask_thresh = np.greater_equal(occlusion_mask, self.tracking_config.occlusion_coef_threshold)
            segmentation_mask = flow_observation.observed_flow_segmentation.numpy(force=True)

            template_data = self.data_graph.get_frame_data(flow_arc_source)
            target_data = self.data_graph.get_frame_data(flow_arc_target)
            template_observation_frontview = template_data.frame_observation
            target_observation_frontview = target_data.frame_observation
            template_image = self.convert_observation_to_numpy(template_observation_frontview.observed_image)
            target_image = self.convert_observation_to_numpy(target_observation_frontview.observed_image)

            template_overlay = overlay_occlusion(template_image, occlusion_mask_thresh.astype(np.float32))

            display_bounds = (0, self.image_width, 0, self.image_height)

            for ax in axs.flat:
                ax.axis('off')

            darkening_factor = 0.5
            axs[0, 0].imshow(template_overlay * darkening_factor, extent=display_bounds)
            axs[0, 0].set_title(f'Template occlusion')

            axs[1, 0].imshow(template_image * darkening_factor, extent=display_bounds)
            axs[1, 0].set_title(f'Template')

            axs[2, 0].imshow(target_image * darkening_factor, extent=display_bounds)
            axs[2, 0].set_title(f'Target')

            step = self.image_width // 20
            x, y = np.meshgrid(np.arange(self.image_width, step=step), np.arange(self.image_height, step=step))
            template_coords = np.stack((y, x), axis=0).reshape(2, -1).T

            # Plot lines on the target front and back view subplots
            occlusion_threshold = self.tracking_config.occlusion_coef_threshold

            flow_frontview_np = opt_flow.numpy(force=True)

            self.visualize_inliers_outliers_matching(axs[1, 0], axs[2, 0], flow_frontview_np,
                                                     rend_flow_np, segmentation_mask, occlusion_mask,
                                                     arc_observation.ransac_inliers,
                                                     arc_observation.ransac_outliers)

            self.plot_matched_lines(axs[1, 0], axs[2, 0], template_coords, occlusion_mask, occlusion_threshold,
                                    flow_frontview_np, cmap='spring', marker='o', segment_mask=segmentation_mask)

            legend_elements = [Patch(facecolor='green', edgecolor='green', label='TP inliers'),
                               Patch(facecolor='red', edgecolor='red', label='FP inliers'),
                               Patch(facecolor='blue', edgecolor='blue', label='Predicted outliers'), ]

            axs[2, 0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

            destination_path = self.ransac_path / f'matching_gt_flow_{flow_arc_source}_{flow_arc_target}.png'

            self.log_pyplot(flow_arc_target, fig, destination_path, RerunAnnotations.matching_correspondences_inliers,
                            dpi=dpi, bbox_inches='tight')

    def visualize_flow_with_matching_rerun(self, frame_i):
        if self.tracking_config.preinitialization_method != 'essential_matrix_decomposition':
            return

        datagraph_camera_data = self.data_graph.get_frame_data(frame_i)
        new_flow_arc = (datagraph_camera_data.long_jump_source, frame_i)
        flow_arc_source, flow_arc_target = new_flow_arc

        rr.set_time_sequence('frame', flow_arc_target)

        arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

        flow_observation = arc_observation.observed_flow
        opt_flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)
        opt_flow_np = opt_flow.numpy(force=True)

        template_data = self.data_graph.get_frame_data(flow_arc_source)
        target_data = self.data_graph.get_frame_data(flow_arc_target)
        template_image = self.convert_observation_to_numpy(template_data.frame_observation.observed_image)
        target_image = self.convert_observation_to_numpy(target_data.frame_observation.observed_image)

        template_target_image = np.concatenate([template_image, target_image], axis=0)
        rerun_image = rr.Image(template_target_image)
        rr.log(RerunAnnotations.matching_correspondences_inliers, rerun_image)
        rr.log(RerunAnnotations.matching_correspondences_outliers, rerun_image)

        inliers_src_yx = arc_observation.ransac_inliers.numpy(force=True)
        outliers_src_yx = arc_observation.ransac_outliers.numpy(force=True)

        inliers_target_yx = source_coords_to_target_coords_np(inliers_src_yx, opt_flow_np)
        outliers_target_yx = source_coords_to_target_coords_np(outliers_src_yx, opt_flow_np)

        def log_correspondences_rerun(cmap, src_yx, target_yx, rerun_annotation):
            target_yx_2nd_image = target_yx
            target_yx_2nd_image[:, 0] = self.image_height + target_yx_2nd_image[:, 0]

            line_strips_xy = np.stack([src_yx[:, [1, 0]], target_yx_2nd_image[:, [1, 0]]], axis=1)

            num_points = line_strips_xy.shape[0]
            colors = [cmap(i / num_points)[:3] for i in range(num_points)]
            colors = (np.array(colors) * 255).astype(int).tolist()

            rr.log(
                rerun_annotation,
                rr.LineStrips2D(
                    strips=line_strips_xy,
                    colors=colors,
                ),
            )

        cmap_inliers = plt.get_cmap('Greens')
        log_correspondences_rerun(cmap_inliers, inliers_src_yx, inliers_target_yx,
                                  RerunAnnotations.matching_correspondences_inliers)
        cmap_outliers = plt.get_cmap('Reds')
        log_correspondences_rerun(cmap_outliers, outliers_src_yx, outliers_target_yx,
                                  RerunAnnotations.matching_correspondences_outliers)

    def visualize_outliers_distribution(self, new_flow_arc):

        new_flow_arc_data = self.data_graph.get_edge_observations(*new_flow_arc)
        gt_flow = new_flow_arc_data.synthetic_flow_result.observed_flow

        inlier_list = torch.nonzero(new_flow_arc_data.ransac_inliers_mask)[:, 0]
        outlier_list = torch.nonzero(~new_flow_arc_data.ransac_inliers_mask)[:, 0]

        src_pts_front = new_flow_arc_data.src_pts_yx
        dst_pts_front = new_flow_arc_data.dst_pts_yx
        dst_pts_gt_flow_front = source_coords_to_target_coords(src_pts_front, gt_flow)

        src_pts_front_inliers = src_pts_front[inlier_list]
        src_pts_front_outliers = src_pts_front[outlier_list]

        dst_pts_front_inliers = src_pts_front[inlier_list]
        dst_pts_front_outliers = dst_pts_front[outlier_list]
        dst_pts_front_inliers_gt = dst_pts_gt_flow_front[inlier_list]
        dst_pts_front_outliers_gt = dst_pts_gt_flow_front[outlier_list]

        errors_inliers = dst_pts_front_inliers - dst_pts_front_inliers_gt
        errors_outliers = dst_pts_front_outliers - dst_pts_front_outliers_gt

        dst_pts_front_outliers_random = dst_pts_front_outliers[np.random.permutation(dst_pts_front_outliers.shape[0])]
        errors_outliers_random = dst_pts_front_outliers_random - dst_pts_front_outliers_gt

        if len(outlier_list) == 0 or len(inlier_list) == 0:
            return

        # Randomly select one inlier and its error
        random_idx = np.random.choice(len(outlier_list))
        matching_error = errors_outliers[random_idx]
        matching_error_random = errors_outliers_random[random_idx]
        # Compute cosine similarity and Euclidean distances
        cosine_similarities = [
            torch.nn.functional.cosine_similarity(matching_error.unsqueeze(0), err.unsqueeze(0), dim=1).item()
            for err in errors_outliers]
        error_magnitudes = torch.linalg.norm(errors_outliers, dim=1).numpy(force=True)
        euclidean_distances = [torch.norm(src_pts_front[outlier_list[random_idx]] - src_pts_front[outlier_list[i]],
                                          dim=0).item()
                               for i in range(len(outlier_list))]

        cosine_similarities_random = [
            torch.nn.functional.cosine_similarity(matching_error_random.unsqueeze(0), err.unsqueeze(0), dim=1).item()
            for err in errors_outliers_random]
        error_magnitudes_random = torch.linalg.norm(errors_outliers_random, dim=1).numpy(force=True)

        fig, ax = plt.subplots()
        scatter = ax.scatter(euclidean_distances, cosine_similarities_random, c=error_magnitudes_random, cmap='Blues',
                             alpha=0.2)
        scatter = ax.scatter(euclidean_distances, cosine_similarities, c=error_magnitudes, cmap='Greys',
                             alpha=0.4)

        flow_source_text = self.tracking_config.gt_flow_source if self.tracking_config.gt_flow_source != 'FlowNetwork' \
            else self.tracking_config.long_flow_model

        legend_elements = [Patch(facecolor='grey', edgecolor='grey', label=f'{flow_source_text} Dense Matching'),
                           Patch(facecolor='blue', edgecolor='blue', label='Random Matching Baseline'), ]

        ax.legend(handles=legend_elements, loc='lower right', fontsize='small')

        colorbar = plt.colorbar(scatter, ax=ax)
        colorbar.set_label('Reprojection Error Magnitude [Pixels]')
        ax.set_xlabel('Euclidean Distance from Randomly Selected Outlier')
        ax.set_ylabel('Cosine Similarity to Random Outlier Reprojection Error')
        ax.set_title(f'{flow_source_text} - RANSAC Outliers Error Correlations')
        plt.savefig(self.ransac_path / f'outliers_spatial_correlation_frame_{new_flow_arc[1]}')
        plt.close()

    def visualize_inliers_outliers_matching(self, ax_source, axs_target, flow_np, rendered_flow, seg_mask,
                                            occlusion, inliers, outliers):
        matching_text = f'ransac method: {self.tracking_config.ransac_inlier_filter}\n'
        if inliers is not None:
            inliers = inliers.numpy(force=True)  # Ensure shape is (2, N)
            self.draw_cross_axes_flow_matches(inliers, occlusion, flow_np, rendered_flow,
                                              ax_source, axs_target, 'Greens', 'Reds', 'inliers',
                                              max_points=20)
            matching_text += f'inliers: {inliers.shape[1]}\n'
        if outliers is not None:
            outliers = outliers.numpy(force=True)  # Ensure shape is (2, N)
            self.draw_cross_axes_flow_matches(outliers, occlusion, flow_np, rendered_flow, ax_source,
                                              axs_target, 'Blues', 'Oranges', 'outliers',
                                              max_points=10)
            matching_text += f'outliers: {outliers.shape[1]}'
        ax_source.text(0.95, 0.95, matching_text, transform=ax_source.transAxes, fontsize=4,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    def draw_cross_axes_flow_matches(self, source_coords, occlusion_mask, flow_np, flow_np_from_movement,
                                     axs1, axs2, cmap_correct, cmap_incorrect, point_type, max_points=30):

        outlier_pixel_threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold

        total_points = source_coords.shape[1]

        assert self.tracking_config.matching_visualization_type in {'matching', 'dots'}

        if total_points > max_points and self.tracking_config.matching_visualization_type == 'matching':
            random_sample = np.random.default_rng(seed=42).permutation(total_points)[:max_points]
            source_coords = source_coords[random_sample, :]

        source_coords[:, 0] = self.image_height - source_coords[:, 0]
        target_coords = source_coords_to_target_coords_image(source_coords, flow_np)
        target_coords_from_pred_movement = source_coords_to_target_coords_image(source_coords, flow_np_from_movement)

        norm = Normalize(vmin=0, vmax=source_coords.shape[1] - 1)
        cmap_correct = plt.get_cmap(cmap_correct)
        cmap_incorrect = plt.get_cmap(cmap_incorrect)
        mappable_correct = ScalarMappable(norm=norm, cmap=cmap_correct)
        mappable_incorrect = ScalarMappable(norm=norm, cmap=cmap_incorrect)

        dots_axs1 = []
        dots_axs2 = []
        colors_axs1 = []
        colors_axs2 = []

        for i in range(0, source_coords.shape[1]):

            yxA = source_coords[:, i].astype(np.int32)
            yxB = target_coords[:, i]
            yxB_movement = target_coords_from_pred_movement[:, i]

            yA, xA = yxA
            yB, xB = yxB
            yB_movement, xB_movement = yxB_movement

            color = mappable_correct.to_rgba(source_coords.shape[1] / 2 + i / 2)
            alpha = 1.0
            if point_type == 'inliers':
                if (np.linalg.norm(yxB - yxB_movement) > outlier_pixel_threshold or
                        occlusion_mask[-yA, xA] >= self.tracking_config.occlusion_coef_threshold):
                    color = mappable_incorrect.to_rgba(source_coords.shape[1] / 2 + i / 2)

                    axs2.plot(xB_movement, yB_movement, color=color, marker='+', markersize=0.5)

                    if self.tracking_config.matching_visualization_type == 'matching':
                        axs2.text(xB, yB, str(i), fontsize=1, ha='left', va='center', color='white')

            elif point_type == 'outliers':
                if np.linalg.norm(yxB - yxB_movement) <= outlier_pixel_threshold:
                    alpha = 0.0
                    color = mappable_incorrect.to_rgba(source_coords.shape[1] / 2 + i / 2)

            dots_axs1.append((xA, yA))
            dots_axs2.append((xB, yB))
            colors_axs1.append(color)
            colors_axs2.append(color)

            if self.tracking_config.matching_visualization_type == 'matching':
                # Create a ConnectionPatch for each pair of sampled points
                axs1.plot(xA, yA, color=color, marker='X', markersize=0.5, alpha=alpha)
                axs2.plot(xB, yB, color=color, marker='X', markersize=0.5, alpha=alpha)

                con = ConnectionPatch(xyA=(xA, yA), xyB=(xB, yB),
                                      coordsA='data', coordsB='data',
                                      axesA=axs1, axesB=axs2, color=color, lw=0.5, alpha=alpha)
                axs2.add_artist(con)

                axs1.text(xA, yA, str(i), fontsize=1, ha='left', va='center', color='white', alpha=alpha)
                axs2.text(xB, yB, str(i), fontsize=1, ha='left', va='center', color='white', alpha=alpha)

        if self.tracking_config.matching_visualization_type == 'dots' and len(dots_axs1) > 0:
            axs1.scatter(*zip(*dots_axs1), c=colors_axs1, marker='X', s=0.5, alpha=0.33)
            axs2.scatter(*zip(*dots_axs2), c=colors_axs2, marker='X', s=0.5, alpha=0.33)

    def log_poses_into_rerun(self, frame_i: int):

        data_graph_node = self.data_graph.get_frame_data(frame_i)
        camera_specific_graph_node = self.data_graph.get_frame_data(frame_i)

        short_jump_source = camera_specific_graph_node.short_jump_source
        long_jump_source = camera_specific_graph_node.long_jump_source

        rr.set_time_sequence("frame", frame_i)
        datagraph_short_edge = self.data_graph.get_edge_observations(short_jump_source, frame_i)
        datagraph_long_edge = self.data_graph.get_edge_observations(long_jump_source, frame_i)

        rr.log(RerunAnnotations.long_short_chain_remaining_pts,
               rr.Scalar(datagraph_long_edge.remaining_pts_after_filtering))
        rr.log(RerunAnnotations.pose_estimation_time, rr.Scalar(data_graph_node.pose_estimation_time))

        pred_obj_quaternion = data_graph_node.predicted_object_se3_long_jump.quaternion.q
        obj_rot_1st_to_last = torch.rad2deg(quaternion_to_axis_angle(pred_obj_quaternion)).cpu().squeeze()
        obj_rot_1st_to_last_gt = torch.rad2deg(data_graph_node.gt_rot_axis_angle).cpu().squeeze()

        obj_tran_1st_to_last = data_graph_node.predicted_object_se3_long_jump.translation.cpu().squeeze()
        obj_tran_1st_to_last_gt = data_graph_node.gt_translation.cpu().squeeze()

        pred_obj_ref_to_last = datagraph_long_edge.predicted_obj_delta_se3
        pred_cam_ref_to_last = datagraph_long_edge.predicted_cam_delta_se3

        gt_obj_ref_to_last = get_relative_gt_obj_rotation(long_jump_source, frame_i, self.data_graph)

        Se3_world_to_cam = self.Se3_world_to_cam
        gt_cam_ref_to_last = Se3_epipolar_cam_from_Se3_obj(gt_obj_ref_to_last, Se3_world_to_cam)

        pred_cam_RANSAC_ref_to_last = datagraph_long_edge.predicted_cam_delta_se3_ransac
        pred_cam_prev_to_last = datagraph_short_edge.predicted_cam_delta_se3
        pred_cam_RANSAC_prev_to_last = datagraph_short_edge.predicted_cam_delta_se3_ransac

        rr.log(RerunAnnotations.cam_delta_r_long_flow_zaragoza,
               rr.Scalar(torch.rad2deg(2 * pred_cam_ref_to_last.quaternion.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_r_long_flow_RANSAC,
               rr.Scalar(torch.rad2deg(2 * pred_cam_RANSAC_ref_to_last.quaternion.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_r_short_flow_zaragoza,
               rr.Scalar(torch.rad2deg(2 * pred_cam_prev_to_last.quaternion.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_r_short_flow_RANSAC,
               rr.Scalar(torch.rad2deg(2 * pred_cam_RANSAC_prev_to_last.quaternion.polar_angle).cpu()))

        rr.log(RerunAnnotations.cam_delta_t_long_flow_zaragoza,
               rr.Scalar(pred_cam_ref_to_last.translation.squeeze().numpy(force=True)))
        rr.log(RerunAnnotations.cam_delta_t_long_flow_RANSAC,
               rr.Scalar(pred_cam_RANSAC_ref_to_last.translation.squeeze().numpy(force=True)))
        rr.log(RerunAnnotations.cam_delta_t_short_flow_zaragoza,
               rr.Scalar(pred_cam_prev_to_last.translation.squeeze().numpy(force=True)))
        rr.log(RerunAnnotations.cam_delta_t_short_flow_RANSAC,
               rr.Scalar(pred_cam_RANSAC_prev_to_last.translation.squeeze().numpy(force=True)))

        pred_obj_rot_ref_to_last = quaternion_to_axis_angle(pred_obj_ref_to_last.quaternion.q).cpu().squeeze().rad2deg()
        pred_cam_rot_ref_to_last = quaternion_to_axis_angle(pred_cam_ref_to_last.quaternion.q).cpu().squeeze().rad2deg()
        gt_obj_rot_ref_to_last = quaternion_to_axis_angle(gt_obj_ref_to_last.quaternion.q).cpu().squeeze().rad2deg()
        gt_cam_rot_ref_to_last = quaternion_to_axis_angle(gt_cam_ref_to_last.quaternion.q).cpu().squeeze().rad2deg()

        rr.log(RerunAnnotations.long_short_chain_diff, rr.Scalar(data_graph_node.predicted_obj_long_short_chain_diff))

        long_jump_chain_pose_q = data_graph_node.predicted_object_se3_long_jump.quaternion
        short_jumps_chain_pose_q = data_graph_node.predicted_object_se3_short_jump.quaternion
        long_jumps_pose_axis_angle = torch.rad2deg(quaternion_to_axis_angle(long_jump_chain_pose_q.q)).cpu().squeeze()
        short_jumps_pose_axis_angle = torch.rad2deg(quaternion_to_axis_angle
                                                    (short_jumps_chain_pose_q.q)).cpu().squeeze()
        rr.log(RerunAnnotations.chained_pose_long_flow_polar,
               rr.Scalar(torch.rad2deg(long_jump_chain_pose_q.polar_angle * 2).item()))
        rr.log(RerunAnnotations.chained_pose_short_flow_polar,
               rr.Scalar(torch.rad2deg(short_jumps_chain_pose_q.polar_angle * 2).item()))

        scale_factor_per_axis_long_edge = datagraph_long_edge.camera_scale_per_axis_gt
        scale_factor_estimated_long_edge = datagraph_long_edge.camera_scale_estimated

        rr.log(RerunAnnotations.translation_scale_estimated, rr.Scalar(scale_factor_estimated_long_edge))

        for axis, axis_label in enumerate(['x', 'y', 'z']):
            rr.log(RerunAnnotations.obj_rot_1st_to_last_axes[axis_label], rr.Scalar(obj_rot_1st_to_last[axis]))
            rr.log(RerunAnnotations.obj_rot_1st_to_last_gt_axes[axis_label], rr.Scalar(obj_rot_1st_to_last_gt[axis]))
            rr.log(RerunAnnotations.obj_tran_1st_to_last_axes[axis_label], rr.Scalar(obj_tran_1st_to_last[axis]))

            rr.log(RerunAnnotations.obj_tran_1st_to_last_gt_axes[axis_label], rr.Scalar(obj_tran_1st_to_last_gt[axis]))
            rr.log(RerunAnnotations.obj_rot_ref_to_last_axes[axis_label], rr.Scalar(pred_obj_rot_ref_to_last[axis]))
            rr.log(RerunAnnotations.cam_rot_ref_to_last_axes[axis_label], rr.Scalar(pred_cam_rot_ref_to_last[axis]))

            rr.log(RerunAnnotations.obj_rot_ref_to_last_axes[axis_label], rr.Scalar(gt_obj_rot_ref_to_last[axis]))
            rr.log(RerunAnnotations.cam_rot_ref_to_last_gt_axes[axis_label], rr.Scalar(gt_cam_rot_ref_to_last[axis]))

            rr.log(RerunAnnotations.obj_tran_ref_to_last_axes[axis_label],
                   rr.Scalar(pred_obj_ref_to_last.translation[0, axis].item()))
            rr.log(RerunAnnotations.cam_tran_ref_to_last_axes[axis_label],
                   rr.Scalar(pred_cam_ref_to_last.translation[0, axis].item()))

            rr.log(RerunAnnotations.obj_tran_ref_to_last_gt_axes[axis_label],
                   rr.Scalar(gt_obj_ref_to_last.translation[0, axis].item()))
            rr.log(RerunAnnotations.cam_tran_ref_to_last_gt_axes[axis_label],
                   rr.Scalar(gt_cam_ref_to_last.translation[0, axis].item()))

            rr.log(RerunAnnotations.chained_pose_long_flow_axes[axis_label],
                   rr.Scalar(long_jumps_pose_axis_angle[axis].item()))
            rr.log(RerunAnnotations.chained_pose_short_flow_axes[axis_label],
                   rr.Scalar(short_jumps_pose_axis_angle[axis].item()))

            rr.log(RerunAnnotations.translation_scale_gt_axes[axis_label],
                   rr.Scalar(scale_factor_per_axis_long_edge[axis].item()))

    def read_poses_from_datagraph(self, frame_indices) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        rotations = []
        translations = []
        gt_rotations = []
        gt_translations = []
        for frame in frame_indices:
            frame_data = self.data_graph.get_frame_data(frame)

            last_quaternion = frame_data.predicted_object_se3_long_jump.quaternion.q
            last_rotation = torch.rad2deg(quaternion_to_axis_angle(last_quaternion).squeeze())
            last_translation = frame_data.predicted_object_se3_long_jump.translation.squeeze()

            rotations.append(last_rotation)
            translations.append(last_translation)

            gt_rotation = torch.rad2deg(frame_data.gt_rot_axis_angle.squeeze())
            gt_translation = frame_data.gt_translation.squeeze()
            gt_rotations.append(gt_rotation)
            gt_translations.append(gt_translation)

        rotations = torch.stack(rotations)
        translations = torch.stack(translations)
        gt_rotations = torch.stack(gt_rotations)
        gt_translations = torch.stack(gt_translations)

        return gt_rotations, gt_translations, rotations, translations

    @staticmethod
    def convert_observation_to_numpy(observation):
        return observation[0, 0].permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def plot_matched_lines(ax1, ax2, source_coords, occlusion_mask, occl_threshold, flow, cmap='jet', marker='o',
                           segment_mask=None, segment_threshold=0.99):
        """
        Draws lines from source coordinates in ax1 to target coordinates in ax2.
        Args:
        - ax1: The matplotlib axis to draw source points on.
        - ax2: The matplotlib axis to draw target points on.
        - source_coords: Source coordinates as a 2xN numpy array (N is the number of points).
        - target_coords: Target coordinates as a 2xN numpy array.
        - cmap: Colormap used for plotting.
        - marker: Marker style for the points.
        """
        # Assuming flow is [2, H, W] and source_coords is [2, N]
        y1, x1 = source_coords.T
        y2_f, x2_f = source_coords_to_target_coords_image(source_coords, flow)

        # Apply masks
        valid_mask = occlusion_mask[-y1.astype(int), x1.astype(int), 0] <= occl_threshold
        if segment_mask is not None:
            valid_mask &= segment_mask[0, 0, 0, -y1.astype(int), x1.astype(int)] > segment_threshold

        # Filter coordinates and colors based on valid_mask
        x1, y1, x2_f, y2_f = x1[valid_mask], y1[valid_mask], x2_f[valid_mask], y2_f[valid_mask]

        # Normalize and map colors
        norm = Normalize(vmin=0, vmax=np.sum(valid_mask))
        cmap = plt.get_cmap(cmap)
        colors = cmap(norm(range(np.sum(valid_mask))))

        # Create LineCollection for valid lines
        lines = np.stack((x1, y1, x2_f, y2_f), axis=1).reshape(-1, 2, 2)
        lc = LineCollection(lines, colors=colors, linewidths=0.5, alpha=0.8)
        ax2.add_collection(lc)

        # Scatter plot for source and target points
        ax1.scatter(x1, y1, color=colors, marker=marker, alpha=0.8, s=1.5)
        ax2.scatter(x2_f, y2_f, color=colors, marker=marker, alpha=0.8, s=1.5)

    def render_silhouette_overlap(self, last_rendered_silhouette, last_segment_mask, frame_idx):
        last_rendered_silhouette_binary = last_rendered_silhouette[0] > 0.5
        last_segment_mask_binary = last_segment_mask[0] > 0.5
        silh_overlap_image = torch.zeros(1, *last_segment_mask.shape[-2:], 3)
        R = torch.tensor([255.0, 0, 0])
        G = torch.tensor([0, 255.0, 0])
        Y = R + G
        # Set yellow where there is last_rendered_silhouette and last_segment_mask
        indicesG = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesG[:, 0], indicesG[:, 1]] = Y
        # Set red where there is last_rendered_silhouette and not last_segment_mask
        indicesR = torch.nonzero((last_segment_mask_binary > 0) & (last_rendered_silhouette_binary <= 0))
        silh_overlap_image[0, indicesR[:, 0], indicesR[:, 1]] = R
        # Set green where there is not last_rendered_silhouette and last_segment_mask
        indicesB = torch.nonzero((last_segment_mask_binary <= 0) & (last_rendered_silhouette_binary > 0))
        silh_overlap_image[0, indicesB[:, 0], indicesB[:, 1]] = G

        silh_overlap_image_np = silh_overlap_image[0].cpu().to(torch.uint8).numpy()
        (self.write_folder / Path('silhouette_overlap')).mkdir(exist_ok=True, parents=True)
        silhouette_overlap_path = self.write_folder / 'silhouette_overlap' / Path(f"silhouette_overlap_{frame_idx}.png")
        imageio.imwrite(silhouette_overlap_path, silh_overlap_image_np)

    def visualize_observed_data(self, frame_i):

        observed_image_annotation = RerunAnnotations.observed_image_frontview
        observed_image_segmentation_annotation = RerunAnnotations.observed_image_segmentation_frontview
        template_image_annotation = RerunAnnotations.template_image_frontview
        template_image_segmentation_annotation = RerunAnnotations.template_image_segmentation_frontview
        observed_flow_occlusion_annotation = RerunAnnotations.observed_flow_occlusion_frontview
        observed_flow_uncertainty_annotation = RerunAnnotations.observed_flow_uncertainty_frontview
        observed_flow_uncertainty_illustration_annotation = RerunAnnotations.observed_flow_with_uncertainty_frontview
        observed_flow_annotation = RerunAnnotations.observed_flow_frontview

        # Save the images to disk
        last_datagraph_node = self.data_graph.get_frame_data(frame_i)
        last_frame_observation = last_datagraph_node.frame_observation

        new_image_path = self.observations_path / Path(f'image_{frame_i}.png')
        new_segment_path = self.segmentation_path / Path(f'seg_{frame_i}.png')
        last_observed_image = last_frame_observation.observed_image.squeeze().cpu().permute(1, 2, 0)

        self.observations_path.mkdir(exist_ok=True, parents=True)
        self.segmentation_path.mkdir(exist_ok=True, parents=True)

        self.log_image(frame_i, last_observed_image, new_image_path, observed_image_annotation)

        image_255 = (last_observed_image * 255).to(torch.uint8)
        seg = last_frame_observation.observed_segmentation.squeeze().unsqueeze(0).repeat(3, 1, 1).permute(1, 2, 0).cpu()
        seg_255 = (seg * 255.).to(torch.uint8)
        imageio.imwrite(new_image_path, image_255)
        imageio.imwrite(new_segment_path, seg_255)

        if self.tracking_config.write_to_rerun_rather_than_disk:
            rr.set_time_sequence("frame", frame_i)
            image_segmentation = last_frame_observation.observed_segmentation
            rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(image_segmentation))

        data_graph_camera = self.data_graph.get_frame_data(frame_i)
        new_flow_arcs = []

        for new_flow_arcs in sorted(new_flow_arcs):
            source_frame = new_flow_arcs[0]
            target_frame = new_flow_arcs[1]

            data_graph_edge_data = self.data_graph.get_edge_observations(source_frame, target_frame)
            flow_observation = data_graph_edge_data.observed_flow
            flow_observation_image_coords = flow_observation.cast_unit_coords_to_image_coords()

            synthetic_flow_obs = data_graph_edge_data.synthetic_flow_result.cast_unit_coords_to_image_coords()

            source_frame_data = self.data_graph.get_frame_data(source_frame)
            target_frame_data = self.data_graph.get_frame_data(target_frame)

            source_frame_observation = source_frame_data.frame_observation
            target_frame_observation = target_frame_data.frame_observation

            source_frame_image = source_frame_observation.observed_image
            source_frame_segment = source_frame_observation.observed_segmentation

            if (len(self.logged_flow_tracks_inits) == 0 or
                    source_frame != self.logged_flow_tracks_inits[-1]):
                template = source_frame_image.squeeze().permute(1, 2, 0)

                template_image_path = self.observations_path / Path(f'template_img_{frame_i}.png')
                self.log_image(target_frame, template, template_image_path,
                               template_image_annotation)

                if self.tracking_config.write_to_rerun_rather_than_disk:
                    rr.set_time_sequence("frame", target_frame)
                    rr.log(template_image_segmentation_annotation, rr.SegmentationImage(source_frame_segment))

            target_frame_image = target_frame_observation.observed_image.cpu()
            target_frame_segment = target_frame_observation.observed_segmentation.cpu()

            observed_flow_reordered = flow_observation_image_coords.observed_flow.squeeze().permute(1, 2, 0).numpy()

            source_image_discrete: torch.Tensor = (source_frame_image * 255).to(torch.uint8).squeeze()
            target_image_discrete: torch.Tensor = (target_frame_image * 255).to(torch.uint8).squeeze()

            source_frame_segment_squeezed = source_frame_segment.squeeze()
            target_frame_segment_squeezed = target_frame_segment.squeeze()
            observed_flow_occlusions_squeezed = flow_observation.observed_flow_occlusion.cpu().squeeze()
            observed_flow_uncertainties_squeezed = flow_observation.observed_flow_uncertainty.cpu().squeeze()

            # TODO this computation is not mathematically justified, and serves just for visualization purposes
            observed_flow_uncertainties_0_1_range = (observed_flow_uncertainties_squeezed -
                                                     observed_flow_uncertainties_squeezed.min())
            observed_flow_uncertainties_0_1_range /= observed_flow_uncertainties_0_1_range.max()

            flow_illustration = visualize_flow_with_images([source_image_discrete], target_image_discrete,
                                                           [observed_flow_reordered], None,
                                                           gt_silhouette_current=source_frame_segment_squeezed,
                                                           gt_silhouettes_prev=[target_frame_segment_squeezed],
                                                           flow_occlusion_masks=[observed_flow_occlusions_squeezed])

            uncertainty_illustration = (
                visualize_flow_with_images([source_image_discrete], target_image_discrete,
                                           [observed_flow_reordered], None,
                                           gt_silhouette_current=source_frame_segment_squeezed,
                                           gt_silhouettes_prev=[target_frame_segment_squeezed],
                                           flow_occlusion_masks=[observed_flow_uncertainties_0_1_range]))

            # flow_errors_illustration = visualize_optical_flow_errors(source_image_discrete,
            #                                                          target_image_discrete,
            #                                                          flow_observation_image_coords,
            #                                                          synthetic_flow_obs)

            # Define output file paths
            observed_flow_path = self.observations_path / Path(f'flow_{source_frame}_{target_frame}.png')
            observed_flow_uncertainty_path = (self.observations_path /
                                              Path(f'flow_uncertainty_{source_frame}_{target_frame}.png'))
            observed_flow_errors_path = self.observations_path / Path(
                f'flow_errors_{source_frame}_{target_frame}.png')
            occlusion_path = self.observations_path / Path(f"occlusion_{source_frame}_{target_frame}.png")
            uncertainty_path = self.observations_path / Path(f"uncertainty_{source_frame}_{target_frame}.png")

            flow_occlusions_image = self.visualize_1D_feature_map_using_overlay(source_frame_image.squeeze(),
                                                                                observed_flow_occlusions_squeezed,
                                                                                alpha=0.8)
            # Uncertainty visualizations
            flow_uncertainty_image = (
                self.visualize_1D_feature_map_using_overlay(source_frame_image.squeeze(),
                                                            observed_flow_uncertainties_0_1_range, alpha=0.8))

            flow_illustration_torch = (
                torchvision.transforms.functional.pil_to_tensor(flow_illustration).permute(1, 2, 0))
            flow_illustration_uncertainty_torch = (
                torchvision.transforms.functional.pil_to_tensor(uncertainty_illustration).permute(1, 2, 0))

            # self.log_pyplot(target_frame, flow_errors_illustration, observed_flow_errors_path,
            #                 observed_flow_errors_annotations)
            self.log_image(target_frame, flow_occlusions_image, occlusion_path,
                           observed_flow_occlusion_annotation)
            self.log_image(target_frame, flow_uncertainty_image, uncertainty_path,
                           observed_flow_uncertainty_annotation)
            self.log_image(target_frame, flow_illustration_uncertainty_torch, observed_flow_uncertainty_path,
                           observed_flow_uncertainty_illustration_annotation, ignore_dimensions=True)
            self.log_image(target_frame, flow_illustration_torch, observed_flow_path, observed_flow_annotation,
                           ignore_dimensions=True)

    def visualize_1D_feature_map_using_overlay(self, source_image_rgb, flow_occlusion, alpha):
        assert flow_occlusion.shape == (self.image_height, self.image_width)
        assert source_image_rgb.shape == (3, self.image_height, self.image_width)

        occlusion_mask = flow_occlusion.detach().unsqueeze(0).repeat(3, 1, 1)
        blended_image = alpha * occlusion_mask + (1 - alpha) * source_image_rgb
        blended_image = (blended_image * 255).to(torch.uint8).squeeze().permute(1, 2, 0)

        return blended_image

    def log_image(self, frame: int, image: torch.Tensor, save_path: Path, rerun_annotation: str,
                  ignore_dimensions=False):
        if not ignore_dimensions:
            assert image.shape == (self.image_height, self.image_width, 3)

        if self.tracking_config.write_to_rerun_rather_than_disk:
            rr.set_time_sequence("frame", frame)
            rr.log(rerun_annotation, rr.Image(image))
        else:
            image_np = image.numpy(force=True)
            imageio.imwrite(save_path, image_np)

    def log_pyplot(self, frame: int, fig: plt.plot, save_path: Path, rerun_annotation: str, **kwargs):

        if self.tracking_config.write_to_rerun_rather_than_disk:
            fig.canvas.draw()

            image_bytes_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_np = image_bytes_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image_np)
            rr.set_time_sequence("frame", frame)
            rr.log(rerun_annotation, rr.Image(image))
        else:
            plt.savefig(str(save_path), **kwargs)

        plt.close()

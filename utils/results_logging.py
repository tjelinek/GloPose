import itertools
from pathlib import Path
from typing import List, Optional

import imageio
import networkx as nx
import numpy as np
import pycolmap
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image
from kornia.geometry import Se3
from matplotlib import pyplot as plt
from data_structures.data_graph import DataGraph
from data_structures.rerun_annotations import RerunAnnotations
from configs.glopose_config import GloPoseConfig
from onboarding.colmap_utils import world2cam_from_reconstruction
from utils.data_utils import load_texture, load_mesh_using_trimesh
from utils.general import normalize_vertices, extract_intrinsics_from_tensor
from visualizations.rerun_utils import (init_rerun_recording, register_matching_series_lines,
                                        visualize_certainty_map, log_matching_correspondences)
from utils.image_utils import overlay_mask


class WriteResults:

    def __init__(self, write_folder, tracking_config: GloPoseConfig, data_graph: DataGraph):

        self.data_graph: DataGraph = data_graph

        self.logged_templates_3d_space: List = list()
        self.logged_keyframe_graph: nx.DiGraph = nx.DiGraph()

        self.config: GloPoseConfig = tracking_config

        self.write_folder = Path(write_folder)

        self.observations_path = self.write_folder / Path('images')
        self.segmentation_path = self.write_folder / Path('segments')
        self.ransac_path = self.write_folder / Path('ransac')
        self.exported_mesh_path = self.write_folder / Path('3d_model')

        self.init_directories()

        self.template_fields: List[str] = []

        self.rerun_init()

    def init_directories(self):
        if not self.config.visualization.write_to_rerun:
            self.observations_path.mkdir(exist_ok=True, parents=True)
            self.segmentation_path.mkdir(exist_ok=True, parents=True)
            self.ransac_path.mkdir(exist_ok=True, parents=True)
            self.exported_mesh_path.mkdir(exist_ok=True, parents=True)

    def rerun_init(self):
        rerun_file = (self.write_folder /
                      f'rerun_{self.config.run.experiment_name}_{self.config.run.sequence}_{self.config.run.special_hash}.rrd')

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

        if self.config.onboarding.frame_filter == 'dense_matching':
            match_reliability_statistics = rrb.TimeSeriesView(
                name=f"{self.config.onboarding.frame_filter} Matching Reliability",
                origin=RerunAnnotations.matching_reliability_plot,
                axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                      zoom_lock=True),
                plot_legend=rrb.PlotLegend(visible=True))
        else:
            max_range = 3.0 * self.config.onboarding.sift_filter_good_to_add_matches
            match_reliability_statistics = rrb.TimeSeriesView(name="SIFT Num of Matches",
                                                              origin=RerunAnnotations.matching_reliability_plot,
                                                              axis_y=rrb.ScalarAxis(range=(0.0, max_range),
                                                                                    zoom_lock=False),
                                                              plot_legend=rrb.PlotLegend(visible=True))

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                contents=[
                    rrb.Tabs(
                        contents=[
                            rrb.Vertical(
                                contents=[
                                    rrb.Horizontal(
                                        contents=[
                                            rrb.Spatial2DView(name="Template Image Current",
                                                              origin=RerunAnnotations.template_image),
                                            rrb.Spatial2DView(name="Observed Image",
                                                              origin=RerunAnnotations.observed_image),
                                        ],
                                        name='Observed Images'
                                    ),
                                    rrb.Grid(
                                        contents=[
                                            rrb.Spatial2DView(name=f"Keyframe {i}",
                                                              origin=f'{RerunAnnotations.keyframe_images}/{i}')
                                            for i in range(27)
                                        ],
                                        grid_columns=9,
                                        name='Templates'
                                    ),
                                    rrb.GraphView(
                                        name='View Graph',
                                        origin=RerunAnnotations.view_graph,
                                    ),
                                ],
                                row_shares=[0.3, 0.5, 0.2],
                                name='Keyframe Images'
                            ),
                            rrb.GraphView(
                                name='Keyframe Graph',
                                origin=RerunAnnotations.keyframe_graph,
                            ),
                            rrb.GraphView(
                                name='View Graph',
                                origin=RerunAnnotations.view_graph,
                            ),
                        ],
                        name='Keyframes'
                    ),
                    rrb.Tabs(
                        contents=[
                            rrb.Spatial3DView(
                                origin=RerunAnnotations.colmap_visualization,
                                name='COLMAP',
                                background=[255, 255, 255]
                            ),
                            rrb.Spatial3DView(
                                origin=RerunAnnotations.space_visualization,
                                name='3D Ground Truth',
                                background=[255, 255, 255]
                            ),
                        ],
                        name='3D Space'
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.TimeSeriesView(name="Pose Estimation (w.o. flow computation)",
                                               origin=RerunAnnotations.pose_estimation_timing),
                        ],
                        grid_columns=2,
                        name='Timings'
                    ),
                    rrb.Tabs(
                        contents=[
                            rrb.Vertical(
                                contents=[
                                    rrb.Horizontal(
                                        contents=[
                                            rrb.Spatial2DView(
                                                name=f"{self.config.onboarding.filter_matcher} Matches High Certainty",
                                                origin=RerunAnnotations.matches_high_certainty),
                                            rrb.Spatial2DView(
                                                name=f"{self.config.onboarding.filter_matcher} Matches Low Certainty",
                                                origin=RerunAnnotations.matches_low_certainty),
                                            *([rrb.Spatial2DView(
                                                name=f"{self.config.onboarding.filter_matcher} Matching Certainty",
                                                origin=RerunAnnotations.matching_certainty)]
                                              if self.config.onboarding.frame_filter == 'dense_matching' else [])
                                        ],
                                        name='Matching'
                                    ),
                                    match_reliability_statistics,
                                ],
                                row_shares=[0.8, 0.2],
                                name='Matching'
                            ),
                            *([rrb.Vertical(
                                contents=[
                                    rrb.Horizontal(
                                        contents=[
                                            rrb.Spatial2DView(
                                                name=f"{self.config.onboarding.filter_matcher} Matches High Certainty",
                                                origin=RerunAnnotations.matches_high_certainty_matchable),
                                            rrb.Spatial2DView(
                                                name=f"{self.config.onboarding.filter_matcher} Matches Low Certainty",
                                                origin=RerunAnnotations.matches_low_certainty_matchable),
                                            rrb.Spatial2DView(name="Template",
                                                              origin=RerunAnnotations.matchability)

                                        ],
                                        name='Matching'
                                    ),
                                    rrb.TimeSeriesView(name="Matchable Area Share",
                                                       origin=RerunAnnotations.matching_matchability_plot,
                                                       axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                             zoom_lock=True),
                                                       plot_legend=rrb.PlotLegend(visible=True)),
                                    rrb.TimeSeriesView(name=f"{self.config.onboarding.filter_matcher} Min Certainty",
                                                       origin=RerunAnnotations.matching_min_roma_certainty_plot,
                                                       axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                             zoom_lock=True),
                                                       plot_legend=rrb.PlotLegend(visible=True)),
                                ],
                                row_shares=[4, 1, 1],
                                name='Matchability'
                            )] if self.config.onboarding.frame_filter == 'dense_matching' and self.config.onboarding.matchability_based_reliability
                              else []),
                        ],
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
                        ],
                        name='Pose'
                    ),
                ],
                name=f'Results - {self.config.run.sequence}'
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

        rerun_name = f'{self.config.run.sequence}-{self.config.run.experiment_name}'
        init_rerun_recording(rerun_name, rerun_file, blueprint)

        if self.config.onboarding.frame_filter == 'dense_matching':
            register_matching_series_lines()
        elif self.config.onboarding.frame_filter == 'SIFT':
            rr.log(RerunAnnotations.min_matches_sift,
                   rr.SeriesLine(color=[255, 0, 0], name="min matches"), static=True)
            rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
                   rr.SeriesLine(color=[0, 255, 0], name="good to add matches"), static=True)
            rr.log(RerunAnnotations.matches_sift,
                   rr.SeriesLine(color=[0, 0, 255], name="matches"), static=True)

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

        rr.log(RerunAnnotations.observed_image_segmentation,
               rr.AnnotationContext([(1, "blue", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotations.template_image_segmentation,
               rr.AnnotationContext([(1, "blue", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)

    def visualize_keyframes(self, frame_i: int, keyframe_graph: nx.Graph):
        rr.set_time_sequence('frame', frame_i)

        kfs = set(keyframe_graph.nodes)
        not_logged_keyframes = kfs - set(self.logged_keyframe_graph.nodes)
        for kf_idx in sorted(not_logged_keyframes):
            keyframe_node = self.data_graph.get_frame_data(kf_idx)
            template = keyframe_node.frame_observation.observed_image[0].permute(1, 2, 0).detach().cpu()

            annotation = f'{RerunAnnotations.keyframe_images}/{len(self.logged_keyframe_graph.nodes)}'
            template_path = self.write_folder / 'templates' / f'{len(keyframe_graph.nodes)}'

            self.log_image(frame_i, template, annotation, template_path)

            # Matchability logging
            if self.config.onboarding.matchability_based_reliability:
                matchability_mask = keyframe_node.matchability_mask
                matchability_image = (~matchability_mask.unsqueeze(0).permute(1, 2, 0)).to(torch.float).numpy(
                    force=True)
                template = (template.numpy(force=True) * 255.0).astype(np.uint8)
                matchability_image_overlay = overlay_mask(template, matchability_image, 1.0, color=(0, 0, 0))
                rr_matchability_image = rr.Image(template).compress(jpeg_quality=self.config.visualization.jpeg_quality)
                rr.log(RerunAnnotations.matchability, rr_matchability_image)

            self.logged_keyframe_graph.add_node(kf_idx)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphNodes(node_ids=list(keyframe_graph.nodes),
                                                              labels=[str(kf) for kf in keyframe_graph.nodes]))

        # not_logged_keyframe_edges = set(keyframe_graph.edges) - set(self.logged_keyframe_graph.edges)
        # self.logged_keyframe_graph.add_edges_from(not_logged_keyframe_edges)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphEdges(edges=[(u, v) for (u, v) in keyframe_graph.edges]))

    def visualize_pose_graph(self, frame_i: int, keyframe_graph: nx.Graph):
        rr.set_time_sequence('frame', frame_i)

        # Create a directed graph from the pose graph
        pose_graph = nx.DiGraph()
        pose_graph.add_nodes_from(self.data_graph.G.nodes)
        # pose_graph.add_edges_from((u, v) for (u, v) in self.data_graph.G.edges
        #                           if self.data_graph.get_edge_observations(u, v).is_match_reliable)
        pose_graph.add_edges_from((n, self.data_graph.get_frame_data(n).matching_source_keyframe)
                                  for n in self.data_graph.G.nodes)
        pose_graph.remove_edges_from((n, n) for n in self.data_graph.G.nodes)

        white_node = [255, 255, 255]
        red_node = [255, 0, 0]

        kfs = set(keyframe_graph.nodes)
        all_nodes = list(kfs)
        node_labels = {kf: str(kf) for kf in kfs}

        for kf in kfs:
            neighbors = sorted({e[0] for e in pose_graph.in_edges(kf)} - set(kfs))
            pose_graph.remove_edges_from((kf, n) for n in neighbors if n not in kfs)
            pose_graph.remove_edges_from((n, kf) for n in neighbors if n not in kfs)

            for k, g in itertools.groupby(enumerate(neighbors),
                                          key=lambda t: t[1] - t[0]):
                g = list(g)
                start = g[0][1]
                end = g[-1][1]
                pose_graph.add_edge(start, kf)
                all_nodes.append(start)
                if start != end:
                    node_labels[start] = f'{start}..{end}'
                else:
                    node_labels[start] = str(start)

        all_nodes_sorted = sorted(all_nodes)
        # Define y-axis positions for keyframes and ordinary frames
        positions = [
            (i * 100, 200.0) if n in kfs else (i * 100, 0.0) for i, n in enumerate(all_nodes_sorted)
        ]

        # Define colors for keyframes and ordinary frames
        colors = [
            red_node if n in kfs else white_node for n in all_nodes_sorted
        ]

        # Log nodes with their positions, colors, and labels
        rr.log(
            RerunAnnotations.view_graph,
            rr.GraphNodes(
                node_ids=all_nodes_sorted,
                positions=positions,
                labels=[node_labels[n] for n in all_nodes_sorted],
                colors=colors,
            )
        )

        all_nodes_set = set(all_nodes)
        # Log edges of the graph
        rr.log(
            RerunAnnotations.view_graph,
            rr.GraphEdges(edges=[(u, v) for (u, v) in pose_graph.edges])
        )

    @torch.no_grad()
    def write_results(self, frame_i, keyframe_graph):

        self.visualize_keyframes(frame_i, keyframe_graph)
        self.visualize_pose_graph(frame_i, keyframe_graph)
        self.visualize_observed_data(frame_i)

        self.visualize_flow_with_matching_rerun(frame_i)

        # self.visualize_3d_camera_space(frame_i, keyframe_graph)

    def visualize_colmap_track(self, frame_i: int, colmap_reconstruction: pycolmap.Reconstruction,
                               visualize_also_gt_poses: bool):
        rr.set_time_sequence("frame", frame_i)

        points_3d_coords = np.stack([p.xyz for p in colmap_reconstruction.points3D.values()], axis=0)
        points_3d_colors = np.stack([p.color for p in colmap_reconstruction.points3D.values()], axis=0)
        rr.log(RerunAnnotations.colmap_pointcloud, rr.Points3D(points_3d_coords, colors=points_3d_colors), static=True)

        all_image_names = [str(self.data_graph.get_frame_data(i).image_filename)
                           for i in range(len(self.data_graph.G.nodes))]

        pred_Se3_world2cam_colmap_frames = world2cam_from_reconstruction(colmap_reconstruction)
        pred_Se3_world2cam = {all_image_names.index(colmap_reconstruction.images[colmap_idx].name): Se3_pose
                              for colmap_idx, Se3_pose in pred_Se3_world2cam_colmap_frames.items()}

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        if visualize_also_gt_poses:
            gt_Se3_world2cam = self.accumulate_Se3_attributes(all_frames_from_0, 'gt_Se3_world2cam')

            gt_t_world2cam = gt_Se3_world2cam.inverse().translation.numpy(force=True)
            pred_t_world2cam = np.stack([pred_Se3_world2cam[frm].inverse().t.numpy(force=True)
                                         for frm in sorted(pred_Se3_world2cam)])

            cmap_gt = plt.get_cmap('Reds')
            cmap_pred = plt.get_cmap('Blues')
            gradient = np.linspace(1., 0.5, self.config.input.input_frames)
            colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)
            colors_pred = (np.asarray([cmap_pred(gradient[i])[:3] for i in range(len(pred_t_world2cam))]) * 255).astype(
                np.uint8)

            strips_gt = np.stack([gt_t_world2cam[:-1], gt_t_world2cam[1:]], axis=1)
            strips_pred = np.stack([pred_t_world2cam[:-1], pred_t_world2cam[1:]], axis=1)

            object_size = np.max(np.linalg.norm(points_3d_coords - np.mean(points_3d_coords, axis=0), axis=1))
            strips_radii = [0.005 * object_size] * n_poses

            rr.log(RerunAnnotations.colmap_gt_camera_track,
                   rr.LineStrips3D(strips=strips_gt,  # gt_t_world2cam
                                   colors=colors_gt,
                                   radii=strips_radii),
                   static=True)

            # rr.log(RerunAnnotations.colmap_pred_camera_track,
            #        rr.LineStrips3D(strips=strips_pred,  # gt_t_world2cam
            #                        colors=colors_pred,
            #                        radii=strips_radii),
            #        static=True)

        image_id_to_poses = {}
        image_name_to_image_id = {image.name: image_id for image_id, image in colmap_reconstruction.images.items()}

        G_reliable = nx.Graph()

        for image_id, image in sorted(colmap_reconstruction.images.items(), key=lambda x: x[0]):
            frame_index = all_image_names.index(image.name)

            pred_t_world2cam = torch.tensor(image.cam_from_world.translation)
            pred_q_world2cam_xyzw = torch.tensor(image.cam_from_world.rotation.quat)

            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Transform3D(translation=pred_t_world2cam,
                               rotation=rr.Quaternion(xyzw=pred_q_world2cam_xyzw),
                               from_parent=True),
                static=True
            )

            camera_params = colmap_reconstruction.cameras[1]
            rr.log(
                f'{RerunAnnotations.colmap_predicted_camera_poses}/{image_id}',
                rr.Pinhole(resolution=[camera_params.width, camera_params.height],
                           focal_length=[camera_params.params[0], camera_params.params[0]],
                           camera_xyz=None  # rr.ViewCoordinates.RUB
                           ),
                static=True
            )

            frame_node = self.data_graph.get_frame_data(frame_index)

            # gt_t_world2cam = frame_node.gt_Se3_cam2obj.t
            # print(f'Frame: {frame_index}, gt: {gt_t_world2cam.numpy(force=True).round(3)},'
            #       f'pred: {image.cam_from_world.translation.round(3)}')

            image_id_to_poses[image_id] = pred_t_world2cam

            for reliable_node_idx in frame_node.reliable_sources:
                reliable_node_data = self.data_graph.get_frame_data(reliable_node_idx)
                reliable_node_name = reliable_node_data.image_filename.name

                if reliable_node_name in image_name_to_image_id:
                    reliable_node_image_id = image_name_to_image_id[reliable_node_name]
                    G_reliable.add_edge(image_id, reliable_node_image_id)

        strips = []
        for im_id1, im_id2 in G_reliable.edges:
            im1_t = image_id_to_poses[im_id1]
            im2_t = image_id_to_poses[im_id2]

            strips.append([im1_t, im2_t])

    def visualize_3d_camera_space(self, frame_i: int, keyframe_graph: nx.DiGraph):

        rr.set_time_sequence("frame", frame_i)

        all_frames_from_0 = range(0, frame_i + 1)
        n_poses = len(all_frames_from_0)

        if (frame_i == 1 and self.config.renderer.gt_mesh_path is not None
                and self.config.renderer.gt_texture_path is not None):
            gt_texture = load_texture(Path(self.config.renderer.gt_texture_path),
                                      self.config.renderer.texture_size)
            gt_texture_int = (gt_texture[0].permute(1, 2, 0) * 255).to(torch.uint8)

            gt_mesh = load_mesh_using_trimesh(Path(self.config.renderer.gt_mesh_path))

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

        gt_cam2obj_se3 = self.accumulate_Se3_attributes(all_frames_from_0, 'gt_Se3_cam2obj')
        pred_cam2obj_se3 = self.accumulate_Se3_attributes(all_frames_from_0, 'pred_Se3_cam2obj')

        gt_q_xyzw_cam2obj = gt_cam2obj_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        pred_q_xyzw_cam2obj = pred_cam2obj_se3.quaternion.q[:, [1, 2, 3, 0]].numpy(force=True)
        gt_t_cam2obj = gt_cam2obj_se3.translation.numpy(force=True)
        pred_t_cam2obj = pred_cam2obj_se3.translation.numpy(force=True)

        rr.set_time_sequence('frame', frame_i)

        rr.log(
            RerunAnnotations.space_predicted_camera_pose,
            rr.Transform3D(translation=pred_t_cam2obj[-1],
                           rotation=rr.Quaternion(xyzw=pred_q_xyzw_cam2obj[-1]),
                           )
        )
        rr.log(
            RerunAnnotations.space_gt_camera_pose,
            rr.Transform3D(translation=gt_t_cam2obj[-1],
                           rotation=rr.Quaternion(xyzw=gt_q_xyzw_cam2obj[-1]),
                           )
        )

        cmap_gt = plt.get_cmap('Greens')
        cmap_pred = plt.get_cmap('Blues')
        gradient = np.linspace(1., 0., self.config.input.input_frames)
        colors_gt = (np.asarray([cmap_gt(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)
        colors_pred = (np.asarray([cmap_pred(gradient[i])[:3] for i in range(n_poses)]) * 255).astype(np.uint8)

        strips_gt = np.stack([gt_t_cam2obj[:-1], gt_t_cam2obj[1:]], axis=1)
        strips_pred = np.stack([pred_t_cam2obj[:-1], pred_t_cam2obj[1:]], axis=1)

        strips_radii_factor = (max(torch.max(torch.cat([gt_t_cam2obj, pred_t_cam2obj]).norm(dim=1)).item(), 5.) / 5.)
        strips_radii = [0.01 * strips_radii_factor] * n_poses

        rr.log(RerunAnnotations.space_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt,  # gt_t_cam2obj
                               colors=colors_gt,
                               radii=strips_radii))

        rr.log(RerunAnnotations.space_predicted_camera_track,
               rr.LineStrips3D(strips=strips_pred,  # pred_t_cam2obj
                               colors=colors_pred,
                               radii=strips_radii))

        datagraph_camera_node = self.data_graph.get_frame_data(frame_i)
        template_frame_idx = datagraph_camera_node.matching_source_keyframe
        datagraph_template_node = self.data_graph.get_frame_data(template_frame_idx)

        template_node_Se3_cam2obj = datagraph_template_node.pred_Se3_cam2obj
        pred_template_node_t_cam2obj = template_node_Se3_cam2obj.translation.squeeze().numpy(force=True)

        rr.log(RerunAnnotations.space_predicted_closest_keypoint,
               rr.LineStrips3D(strips=[[pred_t_cam2obj[-1],
                                        pred_template_node_t_cam2obj]],
                               colors=[[255, 0, 0]],
                               radii=[0.025 * strips_radii_factor]))

        if len(datagraph_camera_node.reliable_sources) > 1:
            for reliable_template_idx in datagraph_camera_node.reliable_sources:
                datagraph_template_node = self.data_graph.get_frame_data(reliable_template_idx)

                template_node_Se3_cam2obj = datagraph_template_node.pred_Se3_cam2obj
                pred_template_node_t_cam2obj = template_node_Se3_cam2obj.translation.squeeze().numpy(force=True)

                rr.log(f'{RerunAnnotations.space_predicted_reliable_templates}/{reliable_template_idx}',
                       rr.LineStrips3D(strips=[[pred_t_cam2obj[-1],
                                                pred_template_node_t_cam2obj]],
                                       colors=[[255, 255, 0]],
                                       radii=[0.025 * strips_radii_factor]))

        for i, keyframe_node_idx in enumerate(sorted(keyframe_graph.nodes)):

            if keyframe_node_idx not in self.logged_templates_3d_space:
                template_idx = len(self.logged_templates_3d_space)

                keyframe_node = self.data_graph.get_frame_data(keyframe_node_idx)
                template = (keyframe_node.frame_observation.observed_image[0].permute(1, 2, 0).numpy(
                    force=True) * 255.).astype(np.uint8)

                self.logged_templates_3d_space.append(keyframe_node_idx)
                template_image_grid_annotation = (f'{RerunAnnotations.space_predicted_camera_keypoints}/'
                                                  f'{template_idx}')
                rr.log(template_image_grid_annotation,
                       rr.Image(template).compress(jpeg_quality=self.config.visualization.jpeg_quality))

                for template_annotation in self.template_fields:
                    rr.log(template_annotation, rr.Scalar(0.0))

                template_frame_data = self.data_graph.get_frame_data(keyframe_node_idx)
                keyframe_pred_Se3_cam2obj = template_frame_data.pred_Se3_cam2obj

                keyframe_pred_q_cam2obj = keyframe_pred_Se3_cam2obj.quaternion.q[:, [1, 2, 3, 0]].squeeze()
                keyframe_pred_t_cam2obj = keyframe_pred_Se3_cam2obj.translation.squeeze()

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Transform3D(translation=keyframe_pred_t_cam2obj.numpy(force=True),
                                   rotation=rr.Quaternion(xyzw=keyframe_pred_q_cam2obj.numpy(force=True)))
                )
                frame_data = self.data_graph.get_frame_data(keyframe_node_idx)
                fx, fy, cx, cy = extract_intrinsics_from_tensor(frame_data.gt_pinhole_K)

                image_width = frame_data.image_shape.width
                image_height = frame_data.image_shape.height

                rr.log(
                    f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                    rr.Pinhole(
                        resolution=[image_width, image_height],
                        focal_length=[float(fx.item()),
                                      float(fy.item())],
                        camera_xyz=rr.ViewCoordinates.RUB,
                    ),
                )

    def visualize_flow_with_matching_rerun(self, frame_i):

        datagraph_camera_data = self.data_graph.get_frame_data(frame_i)
        new_flow_arc = (datagraph_camera_data.matching_source_keyframe, frame_i)
        flow_arc_source, flow_arc_target = new_flow_arc

        if self.config.onboarding.frame_filter == 'passthrough':
            return

        arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

        template_data = self.data_graph.get_frame_data(flow_arc_source)
        target_data = self.data_graph.get_frame_data(flow_arc_target)

        if self.config.onboarding.frame_filter == 'dense_matching':
            reliability = arc_observation.reliability_score
            rr.log(RerunAnnotations.matching_reliability, rr.Scalar(reliability))
            rr.log(RerunAnnotations.matching_reliability_threshold_roma,
                   rr.Scalar(target_data.current_flow_reliability_threshold))
            if self.config.onboarding.matchability_based_reliability:
                matchability_share = template_data.relative_area_matchable
                min_roma_certainty = template_data.roma_certainty_threshold
                rr.log(RerunAnnotations.matching_matchability_plot_share_matchable, rr.Scalar(matchability_share))
                rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty, rr.Scalar(min_roma_certainty))
        elif self.config.onboarding.frame_filter == 'SIFT':
            rr.log(RerunAnnotations.matches_sift, rr.Scalar(arc_observation.num_matches))
            rr.log(RerunAnnotations.min_matches_sift, rr.Scalar(self.config.onboarding.sift_filter_min_matches))
            rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
                   rr.Scalar(self.config.onboarding.sift_filter_good_to_add_matches))

        if frame_i == 0 or (frame_i % self.config.visualization.large_images_write_frequency != 0):
            return

        template_image = template_data.frame_observation.observed_image.squeeze()
        target_image = target_data.frame_observation.observed_image.squeeze()

        template_target_image = torch.cat([template_image, target_image], dim=-2)
        template_target_image_np = (template_target_image.permute(1, 2, 0).numpy(force=True) * 255.).astype(np.uint8)
        rerun_image = rr.Image(template_target_image_np).compress(jpeg_quality=self.config.visualization.jpeg_quality)
        rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
        rr.log(RerunAnnotations.matches_low_certainty, rerun_image)

        # Matchability visualization
        if self.config.onboarding.matchability_based_reliability:
            matchability_mask = template_data.matchability_mask
            matchability_mask_padding = torch.ones_like(matchability_mask)
            matchability_mask_pad = torch.cat([matchability_mask, matchability_mask_padding], dim=0)
            matchability_mask_pad_np = matchability_mask_pad.numpy(force=True)
            template_target_image_matchable_np = overlay_mask(template_target_image_np * 255.,
                                                              ~matchability_mask_pad_np, 1.0, (0, 0, 0))

            mathability_image_rerun = rr.Image(template_target_image_matchable_np).compress(
                jpeg_quality=self.config.visualization.jpeg_quality)
            rr.log(RerunAnnotations.matches_high_certainty_matchable, mathability_image_rerun)
            rr.log(RerunAnnotations.matches_low_certainty_matchable, mathability_image_rerun)

        if self.config.onboarding.frame_filter == 'dense_matching':
            certainties = arc_observation.src_dst_certainty_roma.numpy(force=True)
            threshold = template_data.roma_certainty_threshold
            if threshold is None:
                threshold = self.config.onboarding.min_certainty_threshold

            src_pts_yx = arc_observation.src_pts_xy_roma[:, [1, 0]].numpy(force=True)
            dst_pts_yx = arc_observation.dst_pts_xy_roma[:, [1, 0]].numpy(force=True)

            visualize_certainty_map(arc_observation.roma_flow_warp_certainty,
                                    template_target_image.shape, template_target_image_np,
                                    RerunAnnotations.matching_certainty,
                                    self.config.visualization.jpeg_quality)
        elif self.config.onboarding.frame_filter == 'SIFT':
            src_pts_xy = arc_observation.src_pts_xy_roma
            dst_pts_xy = arc_observation.dst_pts_xy_roma

            if src_pts_xy is not None and dst_pts_xy is not None and len(src_pts_xy) > 0:
                src_pts_yx = src_pts_xy[:, [1, 0]].numpy(force=True)
                dst_pts_yx = dst_pts_xy[:, [1, 0]].numpy(force=True)
            else:
                src_pts_yx = np.zeros((0, 2))
                dst_pts_yx = np.zeros((0, 2))
            certainties = np.ones(src_pts_yx.shape[0])
            threshold = 0.0  # all SIFT matches are inliers
        else:
            return

        template_image_size = template_data.image_shape
        log_matching_correspondences(src_pts_yx, dst_pts_yx, certainties, threshold,
                                     template_image_size.height,
                                     RerunAnnotations.matches_high_certainty,
                                     RerunAnnotations.matches_low_certainty, 20)

        if self.config.onboarding.matchability_based_reliability and self.config.onboarding.frame_filter == 'dense_matching':
            matchable_certainties = arc_observation.src_dst_certainty_roma_matchable.numpy(force=True)
            matchable_src_yx = arc_observation.src_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            matchable_dst_yx = arc_observation.dst_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
            log_matching_correspondences(matchable_src_yx, matchable_dst_yx, matchable_certainties,
                                         threshold, template_image_size.height,
                                         RerunAnnotations.matches_high_certainty_matchable,
                                         RerunAnnotations.matches_low_certainty_matchable, 20)

    def accumulate_Se3_attributes(self, frame_indices, attr_name: str) -> Se3:

        Ts_cam2obj = []

        for frame in frame_indices:
            frame_data = self.data_graph.get_frame_data(frame)
            Ts_cam2obj.append(getattr(frame_data, attr_name).matrix().squeeze())

        T_cam2obj = torch.stack(Ts_cam2obj, dim=0).to(self.config.run.device)
        Se3_cam2obj = Se3.from_matrix(T_cam2obj)

        return Se3_cam2obj

    def visualize_observed_data(self, frame_i):

        if frame_i % self.config.visualization.large_images_write_frequency != 0:
            return

        observed_image_annotation = RerunAnnotations.observed_image
        observed_image_segmentation_annotation = RerunAnnotations.observed_image_segmentation

        prev_visualized_frame_idx = self.config.visualization.large_images_write_frequency
        # Save the images to disk
        prev_frame = self.data_graph.get_frame_data(frame_i - prev_visualized_frame_idx) \
            if frame_i >= prev_visualized_frame_idx else None
        current_datagraph_node = self.data_graph.get_frame_data(frame_i)
        last_frame_observation = current_datagraph_node.frame_observation

        new_image_path = self.observations_path / Path(f'image_{frame_i}.png')
        last_observed_image = last_frame_observation.observed_image.squeeze().permute(1, 2, 0)

        self.log_image(frame_i, last_observed_image, observed_image_annotation, new_image_path)

        rr.set_time_sequence("frame", frame_i)

        if frame_i == 0 or prev_frame.matching_source_keyframe != current_datagraph_node.matching_source_keyframe:
            template_frame_node = self.data_graph.get_frame_data(current_datagraph_node.matching_source_keyframe)
            template_frame_observation = template_frame_node.frame_observation
            template = template_frame_observation.observed_image.squeeze().permute(1, 2, 0)
            template_segment = template_frame_observation.observed_segmentation.numpy(force=True)
            template_path = Path('')
            self.log_image(frame_i, template, RerunAnnotations.template_image, template_path)
            rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(template_segment))

        image_segmentation = last_frame_observation.observed_segmentation.numpy(force=True)
        rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(image_segmentation))

    def log_image(self, frame: int, image: torch.Tensor, rerun_annotation: str, save_path: Optional[Path] = None,
                  ignore_dimensions=False):
        if not ignore_dimensions:
            assert len(image.shape) == 3 and image.shape[-1] == 3

        if self.config.visualization.write_to_rerun:
            rr.set_time_sequence("frame", frame)
            image_np = (image.numpy(force=True) * 255.).astype(np.uint8) if image.dtype != torch.uint8 else image.numpy(
                force=True)
            rr.log(rerun_annotation, rr.Image(image_np).compress(jpeg_quality=self.config.visualization.jpeg_quality))
        else:
            image_np = image.numpy(force=True)
            imageio.imwrite(save_path, image_np)

    def log_pyplot(self, frame: int, fig: plt.plot, save_path: Path, rerun_annotation: str, **kwargs):

        if self.config.visualization.write_to_rerun:
            fig.canvas.draw()

            image_bytes_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_np = image_bytes_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image_np)
            rr.set_time_sequence("frame", frame)
            rr.log(rerun_annotation, rr.Image(image).compress(jpeg_quality=self.config.visualization.jpeg_quality))
        else:
            plt.savefig(str(save_path), **kwargs)

        plt.close()

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
import torchvision
from PIL import Image
from kornia.geometry import Se3
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from data_structures.data_graph import DataGraph
from data_structures.rerun_annotations import RerunAnnotations
from flow import (source_coords_to_target_coords_image)
from pose.glomap import world2cam_from_reconstruction
from tracker_config import TrackerConfig
from utils.data_utils import load_texture, load_mesh_using_trimesh
from utils.general import normalize_vertices, extract_intrinsics_from_tensor
from utils.image_utils import overlay_mask


class WriteResults:

    def __init__(self, write_folder, tracking_config: TrackerConfig, data_graph: DataGraph):

        self.data_graph: DataGraph = data_graph

        self.logged_templates_3d_space: List = list()
        self.logged_keyframe_graph: nx.DiGraph = nx.DiGraph()

        self.config: TrackerConfig = tracking_config

        self.write_folder = Path(write_folder)

        self.observations_path = self.write_folder / Path('images')
        self.segmentation_path = self.write_folder / Path('segments')
        self.ransac_path = self.write_folder / Path('ransac')
        self.exported_mesh_path = self.write_folder / Path('3d_model')

        self.init_directories()

        self.template_fields: List[str] = []

        self.rerun_init()

    def init_directories(self):
        if not self.config.write_to_rerun_rather_than_disk:
            self.observations_path.mkdir(exist_ok=True, parents=True)
            self.segmentation_path.mkdir(exist_ok=True, parents=True)
            self.ransac_path.mkdir(exist_ok=True, parents=True)
            self.exported_mesh_path.mkdir(exist_ok=True, parents=True)

    def rerun_init(self):
        rr.init(f'{self.config.sequence}-{self.config.experiment_name}')
        rerun_file = (self.write_folder /
                      f'rerun_{self.config.experiment_name}_{self.config.sequence}.rrd')
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

        if self.config.frame_filter == 'RoMa':
            match_reliability_statistics = rrb.TimeSeriesView(name="RoMa Matching Reliability",
                                                              origin=RerunAnnotations.matching_reliability_plot,
                                                              axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                                    zoom_lock=True),
                                                              plot_legend=rrb.PlotLegend(visible=True))
        else:
            max_range = 3.0 * self.config.sift_filter_good_to_add_matches
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
                                            rrb.Spatial2DView(name="RoMa Matches High Certainty",
                                                              origin=RerunAnnotations.matches_high_certainty),
                                            rrb.Spatial2DView(name="RoMa Matches Low Certainty",
                                                              origin=RerunAnnotations.matches_low_certainty),
                                            *([rrb.Spatial2DView(name="RoMa Matching Certainty",
                                                                 origin=RerunAnnotations.matching_certainty)]
                                              if self.config.frame_filter == 'RoMa' else [])
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
                                            rrb.Spatial2DView(name="RoMa Matches High Certainty",
                                                              origin=RerunAnnotations.matches_high_certainty_matchable),
                                            rrb.Spatial2DView(name="RoMa Matches Low Certainty",
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
                                    rrb.TimeSeriesView(name="RoMa Min Certainty",
                                                       origin=RerunAnnotations.matching_min_roma_certainty_plot,
                                                       axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                             zoom_lock=True),
                                                       plot_legend=rrb.PlotLegend(visible=True)),
                                ],
                                row_shares=[4, 1, 1],
                                name='Matchability'
                            )] if self.config.frame_filter == 'RoMa' and self.config.matchability_based_reliability
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
                name=f'Results - {self.config.sequence}'
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

        if self.config.frame_filter == 'RoMa':
            rr.log(RerunAnnotations.matching_reliability_threshold_roma,
                   rr.SeriesLine(color=[255, 0, 0], name="min reliability"), static=True)
            rr.log(RerunAnnotations.matching_reliability, rr.SeriesLine(color=[0, 0, 255], name="reliability"),
                   static=True)
            rr.log(RerunAnnotations.matching_matchability_plot_share_matchable,
                   rr.SeriesLine(color=[255, 0, 0], name="share of matchable fg"), static=True)
            rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty,
                   rr.SeriesLine(color=[0, 0, 255], name="min roma certainty"),
                   static=True)
        elif self.config.frame_filter == 'SIFT':
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

        rr.send_blueprint(blueprint)

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
            if self.config.matchability_based_reliability:
                matchability_mask = keyframe_node.matchability_mask
                matchability_image = (~matchability_mask.unsqueeze(0).permute(1, 2, 0)).to(torch.float).numpy(force=True)
                template = template.numpy(force=True) * 255.0
                matchability_image_overlay = overlay_mask(template, matchability_image, 1.0, color=(0, 0, 0))
                rr_matchability_image = rr.Image(template)
                rr.log(RerunAnnotations.matchability, rr_matchability_image)

            self.logged_keyframe_graph.add_node(kf_idx)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphNodes(node_ids=list(keyframe_graph.nodes),
                                                              labels=[str(kf) for kf in keyframe_graph.nodes]))

        # not_logged_keyframe_edges = set(keyframe_graph.edges) - set(self.logged_keyframe_graph.edges)
        # self.logged_keyframe_graph.add_edges_from(not_logged_keyframe_edges)

        rr.log(RerunAnnotations.keyframe_graph, rr.GraphEdges(edges=[(u, v) for (u, v) in keyframe_graph.edges]))

    def log_keyframe_info(self, keyframe_graph: nx.Graph):
        log_path = self.write_folder / 'keyframe_log.txt'

        with open(log_path, 'w') as f:
            def log_print(*args, **kwargs):
                print(*args, **kwargs)
                print(*args, **kwargs, file=f)

            log_print(f'{"Keyframe":<10} {"Start":<10} {"End":<10}')
            current_keyframe = min(keyframe_graph.nodes)
            log_print(f'{current_keyframe:<10} {0:<10}', end='')

            for i in range(len(self.data_graph.G.nodes)):
                template_current = self.data_graph.get_frame_data(i).matching_source_keyframe
                if template_current != current_keyframe:
                    current_keyframe = template_current
                    log_print(f'{i - 1:<10}')
                    log_print(f'{current_keyframe:<10} {i:<10}', end='')

            log_print(f'{i:<10}')

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
            neighbors = sorted({e[0] for e in pose_graph.in_edges(kf)})
            pose_graph.remove_edges_from((kf, n) for n in neighbors if n not in kfs)
            pose_graph.remove_edges_from((n, kf) for n in neighbors if n not in kfs)

            for k, g in itertools.groupby(enumerate(neighbors),
                                          key=lambda t: t[1] - t[0]):
                g = list(g)
                start = g[0][1]
                end = g[-1][1]
                pose_graph.add_edge(start, kf)
                all_nodes.append(start)
                node_labels[start] = f'{start}..{end}'

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

    def visualize_colmap_track(self, frame_i: int, colmap_reconstruction: pycolmap.Reconstruction):
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

        gt_Se3_world2cam = self.accumulate_Se3_attributes(all_frames_from_0, 'gt_Se3_world2cam')

        gt_t_world2cam = gt_Se3_world2cam.inverse().translation.numpy(force=True)
        pred_t_world2cam = np.stack([pred_Se3_world2cam[frm].inverse().t.numpy(force=True)
                                     for frm in sorted(pred_Se3_world2cam)])

        cmap_gt = plt.get_cmap('Reds')
        cmap_pred = plt.get_cmap('Blues')
        gradient = np.linspace(1., 0.5, self.config.input_frames)
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

        if (frame_i == 1 and self.config.gt_mesh_path is not None
                and self.config.gt_texture_path is not None):
            gt_texture = load_texture(Path(self.config.gt_texture_path),
                                      self.config.texture_size)
            gt_texture_int = (gt_texture[0].permute(1, 2, 0) * 255).to(torch.uint8)

            gt_mesh = load_mesh_using_trimesh(Path(self.config.gt_mesh_path))

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
        gradient = np.linspace(1., 0., self.config.input_frames)
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
                template = keyframe_node.frame_observation.observed_image[0].permute(1, 2, 0).numpy(force=True)

                self.logged_templates_3d_space.append(keyframe_node_idx)
                template_image_grid_annotation = (f'{RerunAnnotations.space_predicted_camera_keypoints}/'
                                                  f'{template_idx}')
                rr.log(template_image_grid_annotation, rr.Image(template))

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

        arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target)

        template_data = self.data_graph.get_frame_data(flow_arc_source)
        target_data = self.data_graph.get_frame_data(flow_arc_target)

        if self.config.frame_filter == 'RoMa':
            reliability = arc_observation.reliability_score
            rr.log(RerunAnnotations.matching_reliability, rr.Scalar(reliability))
            rr.log(RerunAnnotations.matching_reliability_threshold_roma,
                   rr.Scalar(target_data.current_flow_reliability_threshold))
            if self.config.matchability_based_reliability:
                matchability_share = template_data.relative_area_matchable
                min_roma_certainty = template_data.roma_certainty_threshold
                rr.log(RerunAnnotations.matching_matchability_plot_share_matchable, rr.Scalar(matchability_share))
                rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty, rr.Scalar(min_roma_certainty))
        elif self.config.frame_filter == 'SIFT':
            rr.log(RerunAnnotations.matches_sift, rr.Scalar(arc_observation.num_matches))
            rr.log(RerunAnnotations.min_matches_sift, rr.Scalar(self.config.sift_filter_min_matches))
            rr.log(RerunAnnotations.good_to_add_number_of_matches_sift,
                   rr.Scalar(self.config.sift_filter_good_to_add_matches))

        if frame_i == 0 or (frame_i % self.config.large_images_results_write_frequency != 0):
            return

        template_image = template_data.frame_observation.observed_image.squeeze()
        target_image = target_data.frame_observation.observed_image.squeeze()

        template_target_image = torch.cat([template_image, target_image], dim=-2)
        template_target_image_np = template_target_image.permute(1, 2, 0).numpy(force=True)
        rerun_image = rr.Image(template_target_image_np)
        rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
        rr.log(RerunAnnotations.matches_low_certainty, rerun_image)

        # Matchability visualization
        if self.config.matchability_based_reliability:
            matchability_mask = template_data.matchability_mask
            matchability_mask_padding = torch.ones_like(matchability_mask)
            matchability_mask_pad = torch.cat([matchability_mask, matchability_mask_padding], dim=0)
            matchability_mask_pad_np = matchability_mask_pad.numpy(force=True)
            template_target_image_matchable_np = overlay_mask(template_target_image_np * 255.,
                                                              ~matchability_mask_pad_np,  1.0, (0, 0, 0))

            mathability_image_rerun = rr.Image(template_target_image_matchable_np)
            rr.log(RerunAnnotations.matches_high_certainty_matchable, mathability_image_rerun)
            rr.log(RerunAnnotations.matches_low_certainty_matchable, mathability_image_rerun)

        if self.config.frame_filter == 'RoMa':
            certainties = arc_observation.src_dst_certainty_roma.numpy(force=True)
            threshold = template_data.roma_certainty_threshold
            if threshold is None:
                threshold = self.config.min_roma_certainty_threshold

            above_threshold_mask = certainties >= threshold
            src_pts_xy_roma = arc_observation.src_pts_xy_roma[:, [1, 0]].numpy(force=True)
            dst_pts_xy_roma = arc_observation.dst_pts_xy_roma[:, [1, 0]].numpy(force=True)

            inliers_source_yx = src_pts_xy_roma[above_threshold_mask]
            inliers_target_yx = dst_pts_xy_roma[above_threshold_mask]
            outliers_source_yx = src_pts_xy_roma[~above_threshold_mask]
            outliers_target_yx = dst_pts_xy_roma[~above_threshold_mask]

            if self.config.matchability_based_reliability:
                certainties_matchable = arc_observation.src_dst_certainty_roma_matchable.numpy(force=True)
                above_threshold_mask_matchable = certainties_matchable >= threshold
                src_pts_xy_roma_matchable = arc_observation.src_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
                dst_pts_xy_roma_matchable = arc_observation.dst_pts_xy_roma_matchable[:, [1, 0]].numpy(force=True)
                inliers_source_yx_matchable = src_pts_xy_roma_matchable[above_threshold_mask_matchable]
                inliers_target_yx_matchable = dst_pts_xy_roma_matchable[above_threshold_mask_matchable]
                outliers_source_yx_matchable = src_pts_xy_roma_matchable[~above_threshold_mask_matchable]
                outliers_target_yx_matchable = dst_pts_xy_roma_matchable[~above_threshold_mask_matchable]

            roma_certainty_map = arc_observation.roma_flow_warp_certainty
            roma_h, roma_w = roma_certainty_map.shape[0], roma_certainty_map.shape[1] // 2
            certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(roma_certainty_map.device)
            certainty_map_column[:roma_h, :roma_w] = roma_certainty_map[:roma_h, :roma_w]
            certainty_map_column[roma_h:, :roma_w] = roma_certainty_map[:roma_h, roma_w:]
            certainty_map_column = certainty_map_column[None]
            roma_certainty_map_image_size = (
                torchvision.transforms.functional.resize(certainty_map_column, size=template_target_image.shape[1:]))

            roma_certainty_map_im_size_np = roma_certainty_map_image_size.numpy(force=True)
            template_target_blacks = np.ones_like(template_target_image_np)
            template_target_image_certainty_np = overlay_mask(template_target_blacks, roma_certainty_map_im_size_np)

            rerun_certainty_img = rr.Image(template_target_image_certainty_np)
            rr.log(RerunAnnotations.matching_certainty, rerun_certainty_img)
        elif self.config.frame_filter == 'SIFT':
            keypoints_matching_indices = arc_observation.sift_keypoint_indices

            template_src_pts = template_data.sift_keypoints
            target_src_pts = target_data.sift_keypoints

            inliers_source_xy = template_src_pts[keypoints_matching_indices[:, 0]]
            inliers_target_xy = target_src_pts[keypoints_matching_indices[:, 1]]

            inliers_source_yx = inliers_source_xy[:, [1, 0]].numpy(force=True)
            inliers_target_yx = inliers_target_xy[:, [1, 0]].numpy(force=True)
            outliers_source_yx = np.zeros((0, 2))
            outliers_target_yx = np.zeros((0, 2))
        else:
            return

        def log_correspondences_rerun(cmap, src_yx, target_yx, rerun_annotation, source_image_height, sample_size=None):
            if sample_size is not None:
                random_indices = torch.randperm(min(sample_size, src_yx.shape[0]))
                src_yx = src_yx[random_indices]
                target_yx = target_yx[random_indices]

            if len(src_yx.shape) == 1 or src_yx.shape[1] == 0:
                return  # No matches to draw
            target_yx_2nd_image = target_yx
            target_yx_2nd_image[:, 0] = source_image_height + target_yx_2nd_image[:, 0]

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

        template_image_size = template_data.image_shape
        cmap_inliers = plt.get_cmap('Greens')
        log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                                  RerunAnnotations.matches_high_certainty, template_image_size.height, 20)
        cmap_outliers = plt.get_cmap('Reds')
        log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                                  RerunAnnotations.matches_low_certainty, template_image_size.height, 20)

        if self.config.matchability_based_reliability and self.config.frame_filter == 'RoMa':
            log_correspondences_rerun(cmap_inliers, inliers_source_yx_matchable, inliers_target_yx_matchable,
                                      RerunAnnotations.matches_high_certainty_matchable, template_image_size.height, 20)
            log_correspondences_rerun(cmap_outliers, outliers_source_yx_matchable, outliers_target_yx_matchable,
                                      RerunAnnotations.matches_low_certainty_matchable, template_image_size.height, 20)


    def accumulate_Se3_attributes(self, frame_indices, attr_name: str) -> Se3:

        Ts_cam2obj = []

        for frame in frame_indices:
            frame_data = self.data_graph.get_frame_data(frame)
            Ts_cam2obj.append(getattr(frame_data, attr_name).matrix().squeeze())

        T_cam2obj = torch.stack(Ts_cam2obj, dim=0).to(self.config.device)
        Se3_cam2obj = Se3.from_matrix(T_cam2obj)

        return Se3_cam2obj

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

        if frame_i % self.config.large_images_results_write_frequency != 0:
            return

        observed_image_annotation = RerunAnnotations.observed_image
        observed_image_segmentation_annotation = RerunAnnotations.observed_image_segmentation

        prev_visualized_frame_idx = self.config.large_images_results_write_frequency
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

        if self.config.write_to_rerun_rather_than_disk:
            rr.set_time_sequence("frame", frame)
            rr.log(rerun_annotation, rr.Image(image))
        else:
            image_np = image.numpy(force=True)
            imageio.imwrite(save_path, image_np)

    def log_pyplot(self, frame: int, fig: plt.plot, save_path: Path, rerun_annotation: str, **kwargs):

        if self.config.write_to_rerun_rather_than_disk:
            fig.canvas.draw()

            image_bytes_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_np = image_bytes_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = Image.fromarray(image_np)
            rr.set_time_sequence("frame", frame)
            rr.log(rerun_annotation, rr.Image(image))
        else:
            plt.savefig(str(save_path), **kwargs)

        plt.close()

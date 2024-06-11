from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from typing import Dict, Tuple, List

import math

import h5py
import torch
import imageio
import csv
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
import seaborn as sns
import torchvision
from PIL import Image
from kornia.geometry import normalize_quaternion
from matplotlib import pyplot as plt, gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Patch
from torch import nn
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from kornia.geometry.conversions import quaternion_to_axis_angle, axis_angle_to_quaternion
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.io import save_ply

from keyframe_buffer import FrameObservation, FlowObservation, KeyframeBuffer
from models.loss import iou_loss, FMOLoss
from tracker_config import TrackerConfig
from auxiliary_scripts.data_structures import DataGraph
from auxiliary_scripts.cameras import Cameras
from utils import coordinates_xy_to_tensor_index
from auxiliary_scripts.math_utils import quaternion_angular_difference
from models.rendering import infer_normalized_renderings, RenderingKaolin
from models.kaolin_wrapper import write_obj_mesh
from models.encoder import EncoderResult, Encoder
from flow import visualize_flow_with_images, compare_flows_with_images, flow_unit_coords_to_image_coords, \
    source_coords_to_target_coords_np, get_non_occluded_foreground_correspondences, source_coords_to_target_coords, \
    get_correct_correspondences_mask


@dataclass
class RerunAnnotations:
    template_image: str = '/observations/template_image_front'
    observed_image: str = '/observations/observed_image_front'
    observed_flow: str = '/observed_flow/observed_flow_front'
    observed_flow_with_uncertainty: str = '/observed_flow/observed_flow_front_uncertainty'
    observed_flow_occlusion: str = '/observed_flow/occlusion_front'
    observed_flow_uncertainty: str = '/observed_flow/uncertainty_front'
    
    # Triangulated points RANSAC
    triangulated_points_gt_Rt_gt_flow: str = '/point_clouds/triangulated_points_gt_Rt_gt_flow'
    triangulated_points_gt_Rt_mft_flow: str = '/point_clouds/triangulated_points_gt_Rt_mft_flow'
    point_cloud_dust3r_im1: str = '/point_clouds/point_cloud_dust3r_im1'
    point_cloud_dust3r_im2: str = '/point_clouds/point_cloud_dust3r_im2'

    # Ransac
    matching_correspondences: str = '/epipolar/matching'

    ransac_stats_old: str = '/epipolar/ransac_stats_img'
    ransac_stats_frontview: str = '/epipolar/ransac_stats_frontview'
    ransac_stats_frontview_visible: str = '/epipolar/ransac_stats_frontview/visible'
    ransac_stats_frontview_predicted_as_visible: str = '/epipolar/ransac_stats_frontview/predicted_as_visible'
    ransac_stats_frontview_correctly_predicted_flows: str = '/epipolar/ransac_stats_frontview/correctly_predicted_flows'
    ransac_stats_frontview_ransac_predicted_inliers: str = '/epipolar/ransac_stats_frontview/ransac_predicted_inliers'
    ransac_stats_frontview_correctly_predicted_inliers: str = '/epipolar/ransac_stats_frontview/correctly_predicted_inliers'
    ransac_stats_frontview_ransac_inlier_ratio: str = '/epipolar/ransac_stats_frontview/ransac_inlier_ratio'

    ransac_stats_backview: str = '/epipolar/ransac_stats_backview'
    ransac_stats_backview_visible: str = '/epipolar/ransac_stats_backview/visible'
    ransac_stats_backview_predicted_as_visible: str = '/epipolar/ransac_stats_backview/predicted_as_visible'
    ransac_stats_backview_correctly_predicted_flows: str = '/epipolar/ransac_stats_backview/correctly_predicted_flows'
    ransac_stats_backview_ransac_predicted_inliers: str = '/epipolar/ransac_stats_backview/ransac_predicted_inliers'
    ransac_stats_backview_correctly_predicted_inliers: str = '/epipolar/ransac_stats_backview/correctly_predicted_inliers'
    ransac_stats_backview_ransac_inlier_ratio: str = '/epipolar/ransac_stats_backview/ransac_inlier_ratio'

    # Pose
    pose_rotation: str = '/pose/rotation'
    pose_rotation_x: str = '/pose/rotation/x_axis'
    pose_rotation_x_gt: str = '/pose/rotation/x_axis_gt'
    pose_rotation_y: str = '/pose/rotation/y_axis'
    pose_rotation_y_gt: str = '/pose/rotation/y_axis_gt'
    pose_rotation_z: str = '/pose/rotation/z_axis'
    pose_rotation_z_gt: str = '/pose/rotation/z_axis_gt'
    
    pose_translation: str = '/pose/translation'
    pose_translation_x: str = '/pose/translation/x_axis'
    pose_translation_x_gt: str = '/pose/translation/x_axis_gt'
    pose_translation_y: str = '/pose/translation/y_axis'
    pose_translation_y_gt: str = '/pose/translation/y_axis_gt'
    pose_translation_z: str = '/pose/translation/z_axis'
    pose_translation_z_gt: str = '/pose/translation/z_axis_gt'

    # Pose
    pose_per_frame: str = '/pose/pose_per_frame'


class WriteResults:

    def __init__(self, write_folder, shape, tracking_config: TrackerConfig, rendering, rendering_backview,
                 gt_encoder, deep_encoder, rgb_encoder, data_graph: DataGraph):

        self.image_height = shape[0]
        self.image_width = shape[1]

        self.data_graph: DataGraph = data_graph

        self.rendering: RenderingKaolin = rendering
        self.rendering_backview: RenderingKaolin = rendering_backview
        self.gt_encoder: Encoder = gt_encoder
        self.deep_encoder: Encoder = deep_encoder
        self.rgb_encoder: Encoder = rgb_encoder

        self.past_frame_renderings: Dict = {}

        self.tracking_config: TrackerConfig = tracking_config
        self.output_size: torch.Size = shape
        self.baseline_iou = -np.ones((self.tracking_config.input_frames - 1, 1))
        self.our_iou = -np.ones((self.tracking_config.input_frames - 1, 1))
        self.tracking_log = open(Path(write_folder) / "tracking_log.txt", "w")
        self.metrics_log = open(Path(write_folder) / "tracking_metrics_log.txt", "w")

        self.write_folder = Path(write_folder)

        self.observations_path = self.write_folder / Path('observed_input')
        self.gt_values_path = self.write_folder / Path('gt_output')
        self.optimized_values_path = self.write_folder / Path('predicted_output')

        self.rerun_log_path = self.write_folder / Path('rerun')
        self.ransac_path = self.write_folder / Path('ransac')
        self.point_clouds_path = self.write_folder / Path('point_clouds')

        self.exported_mesh_path = self.write_folder / Path('3d_model')

        self.observations_path.mkdir(exist_ok=True, parents=True)
        self.gt_values_path.mkdir(exist_ok=True, parents=True)
        self.optimized_values_path.mkdir(exist_ok=True, parents=True)

        self.rerun_log_path.mkdir(exist_ok=True, parents=True)
        self.ransac_path.mkdir(exist_ok=True, parents=True)
        self.point_clouds_path.mkdir(exist_ok=True, parents=True)

        self.exported_mesh_path.mkdir(exist_ok=True, parents=True)

        self.rerun_init()

        self.metrics_writer = csv.writer(self.metrics_log)

        self.metrics_writer.writerow(["Frame", "mIoU", "lastIoU", "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
                                      "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"])

        self.tensorboard_log_dir = Path(write_folder) / Path("logs")
        self.tensorboard_log_dir.mkdir(exist_ok=True, parents=True)
        self.tensorboard_log = None

        self.correspondences_log_file = self.write_folder / (f'correspondences_{self.tracking_config.sequence}'
                                                             f'_flow_{self.tracking_config.gt_flow_source}.h5')
        self.correspondences_log_write_common_data()

    def rerun_init(self):
        rr.init(f'{self.tracking_config.sequence}-{self.tracking_config.experiment_name}')
        rr.save(self.rerun_log_path / f'rerun_{self.tracking_config.experiment_name}_{self.tracking_config.sequence}.rrd')

        blueprint = rrb.Blueprint(
            rrb.Tabs(
                contents=
                [
                    rrb.Grid(
                        contents=[
                            rrb.Spatial2DView(name="Observed Flow Occlusion",
                                              origin=RerunAnnotations.observed_flow),
                            rrb.Spatial2DView(name="Observed Flow Uncertainty",
                                              origin=RerunAnnotations.observed_flow_with_uncertainty),
                            # rrb.Spatial2DView(name="Observed Flow Occlusion",
                            #                   origin=RerunAnnotations.observed_flow_occlusion),
                            # rrb.Spatial2DView(name="Observed Flow Uncertainty",
                            #                   origin=RerunAnnotations.observed_flow_uncertainty)
                        ],
                        name='Observed Input'
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.Spatial2DView(name="Rendered Flow Occlusion", origin="/rendered_flow/occlusion")
                        ],
                        name='Output After Optimization'
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
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(name="Matching Visualization",
                                              origin=RerunAnnotations.matching_correspondences),
                        ],
                        name='Matching'
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.TimeSeriesView(name="RANSAC - Frontview",
                                               origin=RerunAnnotations.ransac_stats_frontview
                                               ),
                            rrb.TimeSeriesView(name="RANSAC - Backview",
                                               origin=RerunAnnotations.ransac_stats_backview
                                               ),
                            rrb.TimeSeriesView(name="Pose - Rotation",
                                               origin=RerunAnnotations.pose_rotation
                                               ),
                            rrb.TimeSeriesView(name="Pose - Translation",
                                               origin=RerunAnnotations.pose_translation
                                               ),
                        ],
                        name='Epipolar'
                    ),
                    rrb.Grid(
                        contents=[
                            rrb.Spatial2DView(name="RANSAC Stats",
                                              origin=RerunAnnotations.ransac_stats_old),
                        ],
                        name='Epipolar (old)'
                    ),
                ],
                name=f'Results - {self.tracking_config.sequence}'
            )
        )

        rr.send_blueprint(blueprint)

    def correspondences_log_write_common_data(self):

        data = {
            "sequence": self.tracking_config.sequence,
            "flow_source": self.tracking_config.gt_flow_source,
            "camera_intrinsics": self.rendering.camera_intrinsics.numpy(force=True),
            "field_of_view_rad": self.rendering.fov,
            "image_width": self.rendering.width,
            "image_height": self.rendering.height,
            "camera_translation": self.rendering.camera_trans[0].numpy(force=True),
            "camera_rotation_matrix": self.rendering.camera_rot.numpy(force=True),
            "camera_translation_backview": self.rendering.camera_trans_backview[0].numpy(force=True),
            "camera_rotation_backview": self.rendering.camera_rot_backview.numpy(force=True),
        }

        with h5py.File(self.correspondences_log_file, 'w') as f:
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                else:
                    f.create_dataset(key, data=[value])

    def __del__(self):

        self.tracking_log.close()
        self.metrics_log.close()

    def visualize_loss_landscape(self, observations: FrameObservation, flow_observations: FlowObservation, tracking6d,
                                 stepi, relative_mode=False):
        """
        Visualizes the loss landscape by computing the joint losses across varying translations and rotations.

        Parameters:
        - tracking6d (Tracking6DClassType): The 6D tracking data, containing ground truth rotations, translations,
                                            and other relevant properties.
        - observations (FrameObservation): Observations.
        - flow_observations (FlowObservation): Flow observations.
        - stepi (int): Current step or iteration.

        - relative_mode (bool, optional): If True and stepi >= 1, it will compute the ground truth marker with respect
                                          the previous predicted value.

        Behavior:
        - The function computes joint losses over a grid defined by translations and rotations around different axes.
        - It visualizes these joint losses as a 2D heatmap, overlaying paths of SGD iterations and ground truth values.
        - The resultant visualization is saved as an EPS file in a 'loss_landscapes' directory under
          `tracking6d.write_folder`.

        Notes:
        - Only certain combinations of translation and rotation axes are considered.
        - The range of translations and rotations are derived from the ground truth values and are limited to specific
          intervals for visualization.
        - Ground truth, start and end points of the SGD path, and contour lines for the loss values are overlaid on the
        heatmap.

        Returns:
        None
        """

        num_translations = 25
        num_rotations = 25

        trans_axes = ['x', 'y', 'z']
        rot_axes = ['x', 'y', 'z']

        for trans_axis_idx, rot_axis_idx in product(range(len(trans_axes)), range(len(rot_axes))):

            if trans_axis_idx in [1, 2] or rot_axis_idx in [0, 2]:
                continue

            if relative_mode and stepi > 1:
                gt_rotation_deg_prev = torch.rad2deg(tracking6d.gt_rotations[0, stepi - 1]).cpu()
                gt_translation_prev = tracking6d.gt_translations[0, 0, stepi - 1].cpu()

                gt_rotation_deg_current = torch.rad2deg(tracking6d.gt_rotations[0, stepi]).cpu()
                gt_translation_current = tracking6d.gt_translations[0, 0, stepi].cpu()

                gt_translation_diff = gt_translation_current - gt_translation_prev
                gt_rotation_diff = gt_rotation_deg_current - gt_rotation_deg_prev

                gt_translation = tracking6d.logged_sgd_translations[0][0, 0, 0].detach().cpu() + gt_translation_diff

                gt_rotation_quaternion = tracking6d.logged_sgd_quaternions[0].detach().cpu()
                gt_rotation_rad = quaternion_to_axis_angle(gt_rotation_quaternion)
                last_rotation_deg = torch.rad2deg(gt_rotation_rad)[0, 0]

                gt_rotation_deg = last_rotation_deg + gt_rotation_diff
            else:
                gt_rotation_deg = torch.rad2deg(tracking6d.gt_rotations[0, stepi]).cpu()
                gt_translation = tracking6d.gt_translations[0, 0, stepi].cpu()

            translations_space = np.linspace(gt_translation[trans_axis_idx] - 0.3,
                                             gt_translation[trans_axis_idx] + 0.3, num=num_translations)
            rotations_space = np.linspace(gt_rotation_deg[rot_axis_idx] - 7,
                                          gt_rotation_deg[rot_axis_idx] + 7, num=num_rotations)

            print(f"Visualizing loss landscape for translation axis {trans_axes[trans_axis_idx]} "
                  f"and rotation axis {rot_axes[rot_axis_idx]}")

            joint_losses: np.ndarray = self.compute_loss_landscape(flow_observations, observations, tracking6d,
                                                                   rotations_space, translations_space, trans_axis_idx,
                                                                   rot_axis_idx, stepi)

            min_val = np.min(joint_losses)
            max_val = np.max(joint_losses)

            plt.figure(figsize=(10, 8))
            plt.imshow(joint_losses.T,
                       extent=[translations_space[0], translations_space[-1], rotations_space[-1], rotations_space[0]],
                       aspect='auto', cmap='hot', interpolation='none', vmin=min_val, vmax=max_val)
            cbar = plt.colorbar(label='Joint Loss')
            ticks = list(cbar.get_ticks())
            ticks.append(min_val)
            ticks.append(max_val)
            ticks = sorted(ticks)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels([f'{tick:.2f}' for tick in ticks])

            plt.xlabel(f'Translation ({trans_axes[trans_axis_idx]}-axis)')
            plt.ylabel(f'Rotation (degrees) ({rot_axes[rot_axis_idx]}-axis)')

            # Set y-ticks for all integral numbers based on the range of rotations_space
            plt.yticks(np.arange(int(rotations_space[0]), int(rotations_space[-1]) + 1, 1))

            # Set x-ticks at every 0.1 step based on the range of translations_space
            plt.xticks(np.arange(translations_space[0], translations_space[-1] + 0.1, 0.1))

            # Add grid
            plt.grid(True, which='both', linestyle='--', linewidth=0.25)

            prev_iteration_x_translation = None
            prev_iteration_y_rotation_deg = None

            for i in range(len(tracking6d.logged_sgd_translations)):
                iteration_translation = tracking6d.logged_sgd_translations[i][0, 0, -1, trans_axis_idx].detach().cpu()
                iteration_rotation_quaternion = tracking6d.logged_sgd_quaternions[i].detach().cpu()
                iteration_rotation_rad = quaternion_to_axis_angle(iteration_rotation_quaternion)
                iteration_rotation_deg = torch.rad2deg(iteration_rotation_rad)[0, -1, rot_axis_idx]

                if i == 0:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='white', marker='x', label='Start')
                    plt.text(iteration_translation, iteration_rotation_deg, 'Start', verticalalignment='bottom',
                             color='white')
                elif i == len(tracking6d.logged_sgd_translations) - 1:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='yellow', marker='x', label='End')
                    plt.text(iteration_translation, iteration_rotation_deg, 'End', verticalalignment='bottom',
                             color='yellow')
                    plt.plot([prev_iteration_x_translation, iteration_translation],
                             [prev_iteration_y_rotation_deg, iteration_rotation_deg], color='yellow', linewidth=0.1,
                             linestyle='dotted')
                else:
                    plt.scatter(iteration_translation, iteration_rotation_deg, color='orange', marker='.', label=str(i))
                    plt.plot([prev_iteration_x_translation, iteration_translation],
                             [prev_iteration_y_rotation_deg, iteration_rotation_deg], color='yellow', linewidth=0.1,
                             linestyle='dotted')

                prev_iteration_x_translation = iteration_translation
                prev_iteration_y_rotation_deg = iteration_rotation_deg

            plt.scatter(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                        color='green', marker='x', label='GT')
            if relative_mode:
                plt.text(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                         'Relative optimum', verticalalignment='top', color='green')
            else:
                plt.text(gt_translation[trans_axis_idx], gt_rotation_deg[rot_axis_idx],
                         'GT', verticalalignment='top', color='green')

            # 2) Show contours of the values
            contours = plt.contour(translations_space, rotations_space, joint_losses.T, levels=20)
            plt.clabel(contours, inline=True, fontsize=10)

            # 3) Visualize the gradient
            # gradient = np.gradient(joint_losses.T)
            # plt.quiver(translations_space, rotations_space, -gradient[1], gradient[0], color='white', width=0.003)

            plt.title('Joint Losses')
            loss_landscape_folder = Path(tracking6d.write_folder / 'loss_landscapes')
            loss_landscape_folder.mkdir(parents=True, exist_ok=True)
            plt.savefig(loss_landscape_folder /
                        f'joint_loss_landscape_{stepi}_trans-{trans_axes[trans_axis_idx]}'
                        f'_rot-{rot_axes[rot_axis_idx]}.eps', format='eps')

    @staticmethod
    def compute_loss_landscape(flow_observations: FlowObservation, observations: FrameObservation, tracking6d,
                               rotation_space, translation_space, trans_axis_idx, rot_axis_idx, stepi):

        joint_losses = np.zeros((translation_space.shape[0], rotation_space.shape[0]))

        for i, translation in enumerate(translation_space):
            for j, rotation_deg in enumerate(rotation_space):
                translation_tensor = torch.Tensor([0, 0, 0])
                translation_tensor[trans_axis_idx] = translation
                sampled_translation = translation_tensor[None, None, None].cuda()

                rotation_tensor_deg = torch.Tensor([0, 0, 0])
                rotation_tensor_deg[rot_axis_idx] = rotation_deg
                rotation_tensor_rad = torch.deg2rad(rotation_tensor_deg)
                rotation_tensor_quaternion = axis_angle_to_quaternion(rotation_tensor_rad)

                sampled_rotation = rotation_tensor_quaternion[None, None].cuda()

                encoder_result, encoder_result_flow_frames = \
                    tracking6d.frames_and_flow_frames_inference([stepi], [stepi - 1], encoder_type='deep_features')

                encoder_result = encoder_result._replace(translations=sampled_translation, quaternions=sampled_rotation)
                flow_arcs_indices = [(-1, -1)]

                inference_result = infer_normalized_renderings(tracking6d.rendering, tracking6d.encoder.face_features,
                                                               encoder_result, encoder_result_flow_frames,
                                                               flow_arcs_indices,
                                                               tracking6d.shape[-1], tracking6d.shape[-2])
                renders, rendered_silhouettes, rendered_flow_result = inference_result

                loss_function: FMOLoss = tracking6d.loss_function
                loss_result = loss_function.forward(rendered_images=renders,
                                                    observed_images=observations.observed_image_features,
                                                    rendered_silhouettes=rendered_silhouettes,
                                                    observed_silhouettes=observations.observed_segmentation,
                                                    rendered_flow=rendered_flow_result.theoretical_flow,
                                                    observed_flow=flow_observations.observed_flow,
                                                    observed_flow_segmentation=flow_observations.observed_flow_segmentation,
                                                    rendered_flow_segmentation=rendered_flow_result.rendered_flow_segmentation,
                                                    observed_flow_occlusion=flow_observations.observed_flow_occlusion,
                                                    rendered_flow_occlusion=rendered_flow_result.rendered_flow_occlusion,
                                                    observed_flow_uncertainties=flow_observations.observed_flow_uncertainty,
                                                    keyframes_encoder_result=encoder_result,
                                                    last_keyframes_encoder_result=encoder_result)

                losses_all, losses, joint_loss, per_pixel_error = loss_result

                joint_losses[i, j] = joint_loss

        return joint_losses

    def set_tensorboard_log_for_frame(self, frame_i):
        self.tensorboard_log = SummaryWriter(str(self.tensorboard_log_dir / f'Frame_{frame_i}'))

    def write_into_tensorboard_log(self, sgd_iter: int, values_dict: Dict):

        for field_name, value in values_dict.items():
            self.tensorboard_log.add_scalar(field_name, value, sgd_iter)

    def write_tensor_into_bbox(self, image, bounding_box):
        image_with_margins = torch.zeros(image.shape[:-2] + self.output_size).to(image.device)
        image_with_margins[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]] = \
            image[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
        return image_with_margins

    @torch.no_grad()
    def write_results(self, bounding_box, frame_i, tex, new_flow_arcs,
                      active_keyframes: KeyframeBuffer, active_keyframes_backview: KeyframeBuffer,
                      best_model, observations: FrameObservation, observations_backview: FrameObservation,
                      gt_rotations, gt_translations):

        observed_segmentations = observations.observed_segmentation

        self.past_frame_renderings[frame_i] = (observations.observed_image[:, [-1]].cpu(),
                                               observations_backview.observed_image[:, [-1]].cpu())

        if self.tracking_config.preinitialization_method == 'essential_matrix_decomposition':
            self.visualize_point_clouds_from_ransac(frame_i)

        if frame_i % self.tracking_config.write_results_frequency == 0:
            self.visualize_optimized_values(bounding_box=bounding_box, keyframe_buffer=active_keyframes,
                                            new_flow_arcs=new_flow_arcs)

            self.visualize_observed_data(active_keyframes, new_flow_arcs)

            self.visualize_flow_with_matching(active_keyframes, active_keyframes_backview, new_flow_arcs)
            self.visualize_rotations_per_epoch(frame_i)

        encoder_result = self.data_graph.get_frame_data(frame_i).encoder_result
        detached_result = EncoderResult(*[it.clone().detach() if type(it) is torch.Tensor else it
                                          for it in encoder_result])

        self.dump_correspondences(active_keyframes, active_keyframes_backview, new_flow_arcs, gt_rotations,
                                  gt_translations)

        self.visualize_logged_metrics(plot_losses=False)

        self.analyze_ransac_matchings_errors(frame_i)
        self.analyze_ransac_matchings(frame_i)

        print(f"Keyframes: {active_keyframes.keyframes}, "
              f"flow arcs: {sorted(active_keyframes.G.edges, key=lambda x: x[::-1])}")

        self.tracking_log.write(f"Step {frame_i}:\n")
        self.tracking_log.write(f"Keyframes: {active_keyframes.keyframes}\n")

        self.write_keyframe_rotations(detached_result, active_keyframes.keyframes)
        self.write_all_encoder_rotations(self.deep_encoder, max(active_keyframes.keyframes) + 1)

        if self.tracking_config.features == 'rgb':
            tex = detached_result.texture_maps

        rgb_renders_result = self.rendering.forward(detached_result.translations, detached_result.quaternions,
                                                    detached_result.vertices, self.deep_encoder.face_features,
                                                    tex, detached_result.lights)

        renders = rgb_renders_result.rendered_image
        rendered_silhouette = rgb_renders_result.rendered_image_segmentation

        self.render_silhouette_overlap(rendered_silhouette[:, [-1]],
                                       observed_segmentations[:, [-1]], frame_i)

        self.save_3d_model(frame_i, tex, best_model, detached_result)

        for tmpi in range(renders.shape[1]):

            segmentations_discrete = (observed_segmentations[:, -1:, [-1]] > 0).to(observed_segmentations.dtype)
            self.baseline_iou[frame_i - 1] = 1 - iou_loss(segmentations_discrete,
                                                          observed_segmentations[:, -1:, [-1]]).detach().cpu()
            self.our_iou[frame_i - 1] = 1 - iou_loss(rendered_silhouette[:, [-1]],
                                                     observed_segmentations[:, -1:, [-1]]).detach().cpu()

        print('Baseline IoU {}, our IoU {}'.format(self.baseline_iou[frame_i - 1], self.our_iou[frame_i - 1]))

    def save_3d_model(self, frame_i, tex, best_model, detached_result):
        # write_obj_mesh(detached_result.vertices[0].cpu().numpy(), best_model["faces"],
        #                self.deep_encoder.face_features[0].cpu().numpy(),
        #                os.path.join(self.write_folder, f'mesh_{frame_i}.obj'), "model_" + str(frame_i) + ".mtl")

        mesh_i_th_path = self.exported_mesh_path / f'mesh_{frame_i}.obj'
        tex_path = self.exported_mesh_path / 'tex_deep.png'
        tex_i_th_path = self.exported_mesh_path / f'tex_{frame_i}.png'
        model_path = self.exported_mesh_path / 'model.mtl'
        model_i_th_path = self.exported_mesh_path / f"model_{frame_i}.mtl"

        write_obj_mesh(detached_result.vertices[0].numpy(force=True), best_model["faces"],
                       self.deep_encoder.face_features[0].numpy(force=True), mesh_i_th_path, str(model_i_th_path.name))
                       
        save_image(detached_result.texture_maps[:, :3], tex_path)
        save_image(tex, tex_i_th_path)

        with open(model_path, "r") as file:
            lines = file.readlines()
        # Replace the last line
        lines[-1] = f"map_Kd tex_{frame_i}.png\n"
        # Write the result to a new file
        with open(model_i_th_path, "w") as file:
            file.writelines(lines)

    def visualize_point_clouds_from_ransac(self, frame_i):

        arc_data = self.data_graph.get_edge_observations(0, frame_i, Cameras.FRONTVIEW)

        rr.set_time_sequence("frame", frame_i)

        triangulated_point_cloud_gt_flow = arc_data.ransac_triangulated_points_gt_Rt_gt_flow[0]
        triangulated_point_cloud_pred_flow = arc_data.ransac_triangulated_points_gt_Rt[0]
        triangulated_point_cloud_dust3r_im1 = arc_data.dust3r_point_cloud_im1
        triangulated_point_cloud_dust3r_im2 = arc_data.dust3r_point_cloud_im2

        triangulated_point_cloud_gt_flow_path = (self.point_clouds_path /
                                                 f'triangulated_point_cloud_gt_Rt_gt_flow_{frame_i}.ply')
        triangulated_point_cloud_pred_flow_path = (self.point_clouds_path /
                                                   f'triangulated_point_cloud_gt_Rt_pred_flow_{frame_i}.ply')
        if self.tracking_config.write_to_rerun_rather_than_disk:
            if triangulated_point_cloud_gt_flow is not None:
                rr.log(RerunAnnotations.triangulated_points_gt_Rt_gt_flow,
                       rr.Points3D(triangulated_point_cloud_gt_flow.numpy()))
            if triangulated_point_cloud_pred_flow is not None:
                rr.log(RerunAnnotations.triangulated_points_gt_Rt_mft_flow,
                       rr.Points3D(triangulated_point_cloud_pred_flow.numpy()))
            if triangulated_point_cloud_dust3r_im1 is not None:
                rr.log(RerunAnnotations.point_cloud_dust3r_im1,
                       rr.Points3D(triangulated_point_cloud_dust3r_im1.numpy()))
            if triangulated_point_cloud_dust3r_im2 is not None:
                rr.log(RerunAnnotations.point_cloud_dust3r_im2,
                       rr.Points3D(triangulated_point_cloud_dust3r_im2.numpy()))
        else:
            save_ply(triangulated_point_cloud_gt_flow_path, triangulated_point_cloud_gt_flow)
            save_ply(triangulated_point_cloud_pred_flow_path, triangulated_point_cloud_pred_flow)

    def measure_ransac_stats(self, frame_i, view: str = 'front'):
        correct_threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold
        results = defaultdict(list)

        for i in range(1, frame_i + 1):
            flow_arc = (0, i)

            camera = Cameras.FRONTVIEW if view == 'front' else Cameras.BACKVIEW
            arc_data = self.data_graph.get_edge_observations(*flow_arc, camera=camera)

            pred_inlier_ratio = arc_data.ransac_inlier_ratio
            inlier_mask = arc_data.ransac_inliers_mask

            observed_flow_image = flow_unit_coords_to_image_coords(arc_data.observed_flow.observed_flow)
            gt_flow_image = flow_unit_coords_to_image_coords(arc_data.gt_flow_result.theoretical_flow)

            src_pts_pred_visible_yx = arc_data.observed_visible_fg_points_mask.nonzero()
            dst_pts_pred_visible_yx = source_coords_to_target_coords(src_pts_pred_visible_yx.permute(1, 0),
                                                                     observed_flow_image).permute(1, 0)
            dst_pts_pred_visible_yx_gt = source_coords_to_target_coords(src_pts_pred_visible_yx.permute(1, 0),
                                                                        gt_flow_image).permute(1, 0)

            correct_flows_epe = torch.linalg.norm(dst_pts_pred_visible_yx - dst_pts_pred_visible_yx_gt, dim=1)

            correct_flows = (correct_flows_epe < correct_threshold)

            dst_pts_pred_visible_yx_small_errors = arc_data.dst_pts_yx
            dst_pts_pred_visible_yx_gt_small_errors = arc_data.dst_pts_yx_gt

            inliers_errors = torch.linalg.norm(arc_data.dst_pts_yx[inlier_mask] -
                                               arc_data.dst_pts_yx_gt[inlier_mask], dim=-1)
            correct_inliers = inliers_errors < correct_threshold

            fg_points_num = float(arc_data.observed_flow_segmentation.sum())
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
            results['model_obtained_from'].append(not arc_data.is_source_of_matching)
            results['ransac_inlier_ratio'].append(pred_inlier_ratio)
            results['mft_flow_gt_flow_difference'].append(dst_pts_pred_visible_yx_small_errors -
                                                          dst_pts_pred_visible_yx_gt_small_errors)

        return results

    def analyze_ransac_matchings_errors(self, frame_i):

        if (frame_i >= 5 and frame_i % 5 == 0) or frame_i >= self.tracking_config.input_frames:

            front_results = self.measure_ransac_stats(frame_i, 'front')
            back_results = self.measure_ransac_stats(frame_i, 'back')

            mft_flow_gt_flow_difference_front = front_results.pop('mft_flow_gt_flow_difference')
            mft_flow_gt_flow_difference_back = back_results.pop('mft_flow_gt_flow_difference')

            if self.tracking_config.plot_mft_flow_kde_error_plot:
                self.plot_distribution_of_inliers_errors(mft_flow_gt_flow_difference_front)

    def analyze_ransac_matchings(self, frame_i):

        front_results = self.measure_ransac_stats(frame_i, 'front')
        back_results = self.measure_ransac_stats(frame_i, 'back')

        front_results.pop('mft_flow_gt_flow_difference')
        back_results.pop('mft_flow_gt_flow_difference')

        colors = {'visible': 'darkgreen',
                  'predicted_as_visible': 'lime',
                  'correctly_predicted_flows': 'yellow',
                  'ransac_predicted_inliers': 'navy',
                  'correctly_predicted_inliers': 'blue',
                  'model_obtained_from': None,  # Not plotted
                  'ransac_inlier_ratio': 'deeppink',
                  }

        for view_type, results in zip(['frontview', 'backview'], [front_results, back_results]):

            for i, metric in enumerate(results.keys()):
                if metric == 'model_obtained_from':
                    continue

                rerun_time_series_entity = getattr(RerunAnnotations, 'ransac_stats_' + view_type + '_' + metric)

                rr.set_time_sequence("frame", frame_i)
                metric_val: float = results[metric][-1]
                rr.log(rerun_time_series_entity, rr.Scalar(metric_val))

        save_freq = 5
        if (frame_i >= save_freq and frame_i % save_freq == 0) or frame_i >= self.tracking_config.input_frames:

            # We want each line to have its assigned color
            assert sorted(colors.keys()) == sorted(front_results.keys()) == sorted(back_results.keys())

            fig = plt.figure(figsize=(20, 10))
            gs = gridspec.GridSpec(2, 3, figure=fig)

            axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
                   fig.add_subplot(gs[1, 2])]

            self.visualize_logged_metrics(rotation_ax=fig.add_subplot(gs[1, 0]),
                                          translation_ax=fig.add_subplot(gs[1, 1]), plot_losses=False)

            axs[0].set_title('Front View')
            axs[1].set_title('Back View')
            axs[2].set_title('Template Front')
            axs[3].set_title('Template Back')

            template_image_f, template_image_b = self.past_frame_renderings[1]
            template_image_f = template_image_f[0, 0].permute(1, 2, 0).numpy(force=True)
            template_image_b = template_image_b[0, 0].permute(1, 2, 0).numpy(force=True)

            axs[2].imshow(template_image_f, aspect='equal')
            axs[3].imshow(template_image_b, aspect='equal')

            handles, labels = [], []

            for ax in [axs[2], axs[3]]:
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

            for ax, results in zip(axs, [front_results, back_results]):

                xs = np.arange(1, frame_i + 1)
                step = 1 if frame_i < 30 else 5
                x_ticks = np.arange(1, frame_i + 1, step)
                x_labels = np.arange(1, frame_i + 1, step)

                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels)

                ax.set_yticks(np.arange(0., 1.05, 0.1))
                ax.set_ylim([0, 1.05])

                for i, metric in enumerate(results.keys()):
                    color = colors[metric]
                    if metric == 'model_obtained_from':
                        continue
                    if metric == 'ransac_inlier_ratio':
                        line, = ax.plot(xs, results[metric], label=metric, linestyle='dashed', color=color)
                    else:
                        line, = ax.plot(xs, results[metric], label=metric, color=color)

                    if ax == axs[0]:
                        handles.append(line)
                        labels.append(metric)

            for ax, results in zip(axs, [front_results, back_results]):
                ylim = ax.get_ylim()
                for i, is_foreground in enumerate(front_results['model_obtained_from']):
                    if not is_foreground:
                        ax.fill_betweenx(ylim, i + 0.5, i + 1.5, color='yellow', alpha=0.3)

            for ax in axs:
                ax.set_xlabel('Frame')
                ax.set_ylabel('Percentage')

            fig.legend(handles, labels, loc='upper left')

            plt.subplots_adjust(right=0.8)  # Adjust the right margin to make space for the legend
            plt.tight_layout(rect=[0.13, 0, 1, 1])

            fig_path = self.ransac_path / 'ransac_stats.svg'

            self.log_pyplot(frame_i, fig, fig_path, RerunAnnotations.ransac_stats_old)

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

    def visualize_flow_with_matching(self, active_keyframes, active_keyframes_backview, new_flow_arcs):

        for new_flow_arc in new_flow_arcs:

            flow_arc_source, flow_arc_target = new_flow_arc

            new_flow_arc_data_front = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target,
                                                                            camera=Cameras.FRONTVIEW)
            new_flow_arc_data_back = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target,
                                                                           camera=Cameras.BACKVIEW)

            if flow_arc_source != 0:
                continue
                # TODO not the most elegant thing to do

            arc_observation_front = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target,
                                                                          Cameras.FRONTVIEW)
            arc_observation_back = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target,
                                                                         Cameras.BACKVIEW)

            rendered_flow_res = arc_observation_front.gt_flow_result
            rendered_flow_res_back = arc_observation_back.gt_flow_result

            rend_flow = flow_unit_coords_to_image_coords(rendered_flow_res.theoretical_flow)
            rend_flow_np = rend_flow.numpy(force=True)
            rend_flow_back = flow_unit_coords_to_image_coords(rendered_flow_res_back.theoretical_flow)
            rend_flow_back_np = rend_flow_back.numpy(force=True)

            values_frontview = self.get_values_for_matching(active_keyframes, flow_arc_source, flow_arc_target)
            (occlusion_mask_front, seg_mask_front, target_front,
             template_front, template_front_overlay, step, flow_frontview) = values_frontview

            display_bounds = (0, target_front.shape[0], 0, target_front.shape[1])

            fig, axs = plt.subplots(3, 2, figsize=(8, 8), dpi=600)

            flow_source_label = self.tracking_config.gt_flow_source
            if flow_source_label == 'FlowNetwork':
                flow_source_label = self.tracking_config.long_flow_model

            heading_text = (f"Frames {new_flow_arc}\n"
                            f"Flow: {flow_source_label}")
            axs[0, 0].text(1.05, 1, heading_text, transform=axs[0, 0].transAxes, verticalalignment='top',
                           fontsize='medium')

            for ax in axs.flat:
                ax.axis('off')

            darkening_factor = 0.5
            axs[0, 0].imshow(template_front * darkening_factor, extent=display_bounds)
            axs[0, 0].set_title('Template Front')

            axs[1, 0].imshow(template_front_overlay * darkening_factor, extent=display_bounds)
            axs[1, 0].set_title('Template Front Occlusion')

            axs[2, 0].imshow(target_front * darkening_factor, extent=display_bounds)
            axs[2, 0].set_title('Target Front')

            height, width = template_front.shape[:2]  # Assuming these are the dimensions of your images
            x, y = np.meshgrid(np.arange(width, step=step), np.arange(height, step=step))
            template_coords = np.stack((y, x), axis=0).reshape(2, -1)  # Shape: [2, height*width]

            # Plot lines on the target front and back view subplots
            occlusion_threshold = self.tracking_config.occlusion_coef_threshold

            flow_frontview_np = flow_frontview.numpy(force=True)

            self.visualize_inliers_outliers_matching(axs[1, 0], axs[2, 0], flow_frontview_np,
                                                     rend_flow_np, seg_mask_front, occlusion_mask_front,
                                                     new_flow_arc_data_front.ransac_inliers,
                                                     new_flow_arc_data_front.ransac_outliers)

            self.visualize_outliers_distribution(new_flow_arc)

            legend_elements = [Patch(facecolor='green', edgecolor='green', label='TP inliers'),
                               Patch(facecolor='red', edgecolor='red', label='FP inliers'),
                               Patch(facecolor='blue', edgecolor='blue', label='Predicted outliers'), ]

            axs[2, 0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

            self.plot_matched_lines(axs[1, 0], axs[2, 0], template_coords, occlusion_mask_front, occlusion_threshold,
                                    flow_frontview_np, cmap='spring', marker='o', segment_mask=seg_mask_front)

            if self.tracking_config.matching_target_to_backview:
                values_backview = self.get_values_for_matching(active_keyframes_backview, flow_arc_source,
                                                               flow_arc_target)
                (occlusion_mask_back, seg_mask_back, target_back, template_back,
                 template_back_overlay, step, flow_backview) = values_backview

                assert template_back_overlay.shape == template_front_overlay.shape
                assert target_back.shape == target_front.shape

                axs[0, 1].imshow(template_back * darkening_factor, extent=display_bounds)
                axs[0, 1].set_title('Template Back')

                axs[1, 1].imshow(template_back_overlay * darkening_factor, extent=display_bounds)
                axs[1, 1].set_title('Template Back Occlusion')

                axs[2, 1].imshow(target_front * darkening_factor, extent=display_bounds)
                axs[2, 1].set_title('Target Back')

                flow_backview_np = flow_backview.numpy(force=True)

                self.visualize_inliers_outliers_matching(axs[1, 1], axs[2, 1], flow_backview_np,
                                                         rend_flow_back_np, seg_mask_back, occlusion_mask_back,
                                                         new_flow_arc_data_back.ransac_inliers,
                                                         new_flow_arc_data_back.ransac_outliers)

                self.plot_matched_lines(axs[1, 1], axs[2, 1], template_coords, occlusion_mask_back, occlusion_threshold,
                                        flow_backview_np, cmap='cool', marker='x', segment_mask=seg_mask_back)

            destination_path = self.ransac_path / f'matching_gt_flow_{flow_arc_source}_{flow_arc_target}.png'

            self.log_pyplot(flow_arc_target, fig, destination_path, RerunAnnotations.matching_correspondences,
                            dpi=600, bbox_inches='tight')

    def visualize_outliers_distribution(self, new_flow_arc):

        new_flow_arc_data = self.data_graph.get_edge_observations(*new_flow_arc, camera=Cameras.FRONTVIEW)
        gt_flow = new_flow_arc_data.gt_flow_result.theoretical_flow

        inlier_list = torch.nonzero(new_flow_arc_data.ransac_inliers_mask)[:, 0]
        outlier_list = torch.nonzero(~new_flow_arc_data.ransac_inliers_mask)[:, 0]

        src_pts_front = new_flow_arc_data.src_pts_yx
        dst_pts_front = new_flow_arc_data.dst_pts_yx
        dst_pts_gt_flow_front = source_coords_to_target_coords(src_pts_front.T, gt_flow).T

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
        matching_text = f'ransac method: {self.tracking_config.ransac_essential_matrix_algorithm}\n'
        if inliers is not None:
            inliers = inliers.numpy(force=True).T  # Ensure shape is (2, N)
            self.draw_cross_axes_flow_matches(inliers, occlusion, flow_np, rendered_flow,
                                              ax_source, axs_target, 'Greens', 'Reds', 'inliers',
                                              max_points=20)
            matching_text += f'inliers: {inliers.shape[1]}\n'
        if outliers is not None:
            outliers = outliers.numpy(force=True).T  # Ensure shape is (2, N)
            self.draw_cross_axes_flow_matches(outliers, occlusion, flow_np, rendered_flow, ax_source,
                                              axs_target, 'Blues', 'Oranges', 'outliers',
                                              max_points=10)
            matching_text += f'outliers: {outliers.shape[1]}'
        ax_source.text(0.95, 0.95, matching_text, transform=ax_source.transAxes, fontsize=4,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

    def dump_correspondences(self, keyframes: KeyframeBuffer, keyframes_backview: KeyframeBuffer, new_flow_arcs,
                             gt_rotations, gt_translations):

        with h5py.File(self.correspondences_log_file, 'a') as f:

            for flow_arc in new_flow_arcs:
                source_frame, target_frame = flow_arc

                flow_observation_frontview = keyframes.get_flows_between_frames(source_frame, target_frame)
                flow_observation_frontview.observed_flow = flow_unit_coords_to_image_coords(
                    flow_observation_frontview.observed_flow)

                dst_pts_xy_frontview, occlusion_score_frontview, src_pts_xy_frontview, gt_inliers_frontview = \
                    self.get_correspondences_from_observations(flow_observation_frontview, source_frame, target_frame)

                if self.tracking_config.matching_target_to_backview:
                    flow_observation_backview = keyframes_backview.get_flows_between_frames(source_frame, target_frame)
                    flow_observation_backview.observed_flow = flow_unit_coords_to_image_coords(
                        flow_observation_backview.observed_flow)

                    dst_pts_xy_backview, occlusion_score_backview, src_pts_xy_backview, gt_inliers_backview = \
                        self.get_correspondences_from_observations(flow_observation_backview, source_frame,
                                                                   target_frame)

                else:
                    dst_pts_xy_backview = torch.zeros_like(dst_pts_xy_frontview)
                    occlusion_score_backview = torch.zeros_like(occlusion_score_frontview)
                    src_pts_xy_backview = torch.zeros_like(src_pts_xy_frontview)
                    gt_inliers_backview = torch.zeros_like(gt_inliers_frontview)

                frame_data = {
                    "flow_arc": flow_arc,
                    "gt_translation": gt_translations[0, 0, target_frame].numpy(force=True),
                    "gt_rotation_change": torch.rad2deg(gt_rotations[0, target_frame]).numpy(force=True),
                    "source_points_xy_frontview": src_pts_xy_frontview.numpy(force=True),
                    "target_points_xy_frontview": dst_pts_xy_frontview.numpy(force=True),
                    "gt_inliers_frontview": gt_inliers_frontview.numpy(force=True),
                    "frontview_matching_occlusion": occlusion_score_frontview.numpy(force=True),
                    "source_points_xy_backview": src_pts_xy_backview.numpy(force=True),
                    "target_points_xy_backview": dst_pts_xy_backview.numpy(force=True),
                    "gt_inliers_backview": gt_inliers_backview.numpy(force=True),
                    "backview_matching_occlusion": occlusion_score_backview.numpy(force=True),
                }

                if 'correspondences' not in f:
                    frames_grp = f.create_group('correspondences')
                else:
                    frames_grp = f['correspondences']

                frame_grp_name = f'frame_{flow_arc}'
                frame_grp = frames_grp.create_group(frame_grp_name)

                for key, value in frame_data.items():
                    frame_grp.create_dataset(key, data=value)

    def get_correspondences_from_observations(self, flow_observation: FlowObservation, source_frame: int,
                                              target_frame: int):
        src_pts_xy_frontview, dst_pts_xy_frontview = \
            get_non_occluded_foreground_correspondences(flow_observation.observed_flow_occlusion,
                                                        flow_observation.observed_flow_segmentation,
                                                        flow_observation.observed_flow,
                                                        self.tracking_config.segmentation_mask_threshold,
                                                        self.tracking_config.occlusion_coef_threshold)
        src_pts_yx_frontview = coordinates_xy_to_tensor_index(src_pts_xy_frontview).to(torch.long)
        occlusion_score_frontview = (
            flow_observation.observed_flow_occlusion[
                ..., src_pts_yx_frontview[:, 0], src_pts_yx_frontview[:, 1]
            ].squeeze()
        )

        gt_flow_observation_frontview = self.rendering.render_flow_for_frame(self.gt_encoder, source_frame,
                                                                             target_frame)

        threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold

        gt_flow_frontview = flow_unit_coords_to_image_coords(gt_flow_observation_frontview.theoretical_flow)
        inlier_indices = get_correct_correspondences_mask(gt_flow_frontview,
                                                          src_pts_xy_frontview[:, [1, 0]],
                                                          dst_pts_xy_frontview[:, [1, 0]],
                                                          threshold)

        return dst_pts_xy_frontview, occlusion_score_frontview, src_pts_xy_frontview, inlier_indices

    def draw_cross_axes_flow_matches(self, source_coords, occlusion_mask, flow_np, flow_np_from_movement,
                                     axs1, axs2, cmap_correct, cmap_incorrect, point_type, max_points=30):

        outlier_pixel_threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold

        total_points = source_coords.shape[1]

        assert self.tracking_config.matching_visualization_type in {'matching', 'dots'}

        if total_points > max_points and self.tracking_config.matching_visualization_type == 'matching':
            random_sample = np.random.default_rng(seed=42).permutation(total_points)[:max_points]
            source_coords = source_coords[:, random_sample]

        source_coords[0, :] = flow_np.shape[-2] - source_coords[0, :]
        target_coords = source_coords_to_target_coords_np(source_coords, flow_np)
        target_coords_from_pred_movement = source_coords_to_target_coords_np(source_coords, flow_np_from_movement)

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

    def get_values_for_matching(self, active_keyframes, flow_arc_source, flow_arc_target):
        flow_observation_frontview = active_keyframes.get_flows_between_frames(flow_arc_source, flow_arc_target)
        flow_frontview = flow_observation_frontview.observed_flow
        occlusion_mask_front = self.convert_observation_to_numpy(flow_observation_frontview.observed_flow_occlusion)
        seg_mask_front = flow_observation_frontview.observed_flow_segmentation.numpy(force=True)
        flow_frontview = flow_unit_coords_to_image_coords(flow_frontview.clone())
        template_observation_frontview = active_keyframes.get_observations_for_keyframe(flow_arc_source)
        target_observation_frontview = active_keyframes.get_observations_for_keyframe(flow_arc_target)
        template_front = self.convert_observation_to_numpy(template_observation_frontview.observed_image)
        target_front = self.convert_observation_to_numpy(target_observation_frontview.observed_image)

        N = 20
        step = template_front.shape[0] // N

        template_front_overlay = self.overlay_occlusion(template_front, occlusion_mask_front >=
                                                        self.tracking_config.occlusion_coef_threshold)

        return (occlusion_mask_front, seg_mask_front, target_front,
                template_front, template_front_overlay, step, flow_frontview)

    @staticmethod
    def add_loss_plot(ax_, frame_losses_, indices=None):
        if indices is None:
            indices = range(len(frame_losses_))
        ax_loss = ax_.twinx()
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(indices, frame_losses_, color='red', label='Loss')
        ax_loss.spines.right.set_position(("axes", 1.15))
        ax_loss.legend(loc='upper left')

    def visualize_logged_metrics(self, rotation_ax=None, translation_ax=None, plot_losses=True):

        using_own_axes = False
        fig = None
        if rotation_ax is None and translation_ax is None:
            using_own_axes = True
            fig, axs = plt.subplots(2, 1, figsize=(12, 15))
            fig.subplots_adjust(hspace=0.4)
            rotation_ax, translation_ax = axs

        frame_indices = sorted(self.data_graph.G.nodes)[1:]
        losses = [self.data_graph.get_frame_data(frame).frame_losses for frame in frame_indices]

        rotations = []
        translations = []
        gt_rotations = []
        gt_translations = []

        for frame in frame_indices:
            frame_data = self.data_graph.get_frame_data(frame)
            last_quaternion = quaternion_to_axis_angle(frame_data.quaternions_during_optimization[-1])
            last_rotation = np.rad2deg(last_quaternion[0, -1].numpy(force=True))
            last_translation = frame_data.translations_during_optimization[-1][0, 0, -1].numpy(force=True)
            rotations.append(last_rotation)
            translations.append(last_translation)

            frame_data = self.data_graph.get_frame_data(frame)
            gt_rotation = np.rad2deg(frame_data.gt_rot_axis_angle.squeeze().numpy(force=True))
            gt_translation = frame_data.gt_translation.squeeze().numpy(force=True)
            gt_rotations.append(gt_rotation)
            gt_translations.append(gt_translation)

        rotations = np.array(rotations)
        translations = np.array(translations)
        gt_rotations = ((np.array(gt_rotations) - 180) % 360) - 180
        gt_translations = np.array(gt_translations)
        
        # Rerun
        rr.set_time_sequence("frame", max(frame_indices))

        rr.log(RerunAnnotations.pose_rotation_x, rr.Scalar(rotations[-1][0]))
        rr.log(RerunAnnotations.pose_rotation_y, rr.Scalar(rotations[-1][1]))
        rr.log(RerunAnnotations.pose_rotation_z, rr.Scalar(rotations[-1][2]))
        
        rr.log(RerunAnnotations.pose_rotation_x_gt, rr.Scalar(gt_rotations[-1][0]))
        rr.log(RerunAnnotations.pose_rotation_y_gt, rr.Scalar(gt_rotations[-1][1]))
        rr.log(RerunAnnotations.pose_rotation_z_gt, rr.Scalar(gt_rotations[-1][2]))

        rr.log(RerunAnnotations.pose_translation_x, rr.Scalar(translations[-1][0]))
        rr.log(RerunAnnotations.pose_translation_y, rr.Scalar(translations[-1][1]))
        rr.log(RerunAnnotations.pose_translation_z, rr.Scalar(translations[-1][2]))

        rr.log(RerunAnnotations.pose_translation_x_gt, rr.Scalar(gt_translations[-1][0]))
        rr.log(RerunAnnotations.pose_translation_y_gt, rr.Scalar(gt_translations[-1][1]))
        rr.log(RerunAnnotations.pose_translation_z_gt, rr.Scalar(gt_translations[-1][2]))

        # Plot Rotation
        colors = ['yellow', 'green', 'blue']
        ticks = list(frame_indices) if len(list(frame_indices)) < 30 else list(frame_indices)[::5]

        def plot_motion(ax, frame_indices_, data, gt_data, labels, title, ylabel):
            if ax is not None:
                ax.set_xticks(ticks)
                for i, axis_label in enumerate(labels):
                    ax.plot(frame_indices_, data[:, i], label=f'{axis_label}', color=colors[i])
                    ax.plot(frame_indices_, gt_data[:, i], '--', label=f'GT {axis_label}', alpha=0.5, color=colors[i])
                ax.set_title(title)
                ax.set_xlabel('Frame Index')
                ax.set_ylabel(ylabel)
                if len(data[:, i]) > 36:
                    ax.legend(loc='lower left', fontsize='small')
                else:
                    ax.legend(loc='upper left', fontsize='small')

        flow_source_text = self.tracking_config.gt_flow_source if self.tracking_config.gt_flow_source != 'FlowNetwork' \
            else self.tracking_config.long_flow_model

        min_data = min(rotations.min(), gt_rotations.min())
        max_data = max(gt_rotations.max(), gt_rotations.max())

        yticks_frequency = 20 if (max_data - min_data) / 20 <= 20 else 30

        lower_bound = yticks_frequency * np.floor(min_data / yticks_frequency)
        upper_bound = yticks_frequency * np.ceil(max_data / yticks_frequency)

        yticks = np.arange(lower_bound, upper_bound + yticks_frequency, yticks_frequency)
        rotation_ax.set_yticks(yticks)

        plot_motion(rotation_ax, frame_indices, rotations, gt_rotations,
                    ['X-axis Rotation', 'Y-axis Rotation', 'Z-axis Rotation'],
                    f'Rotation per Frame {flow_source_text}', 'Rotation')

        plot_motion(translation_ax, frame_indices, translations, gt_translations,
                    ['X-axis Translation', 'Y-axis Translation', 'Z-axis Translation'],
                    f'Translation per Frame {flow_source_text}', 'Translation')

        if plot_losses is True:
            if rotation_ax is not None:
                self.add_loss_plot(rotation_ax, losses, indices=frame_indices)
            if translation_ax is not None:
                self.add_loss_plot(translation_ax, losses, indices=frame_indices)

        (Path(self.write_folder) / Path('rotations_by_epoch')).mkdir(exist_ok=True, parents=True)
        fig_path = Path(self.write_folder) / Path('rotations_by_epoch') / f'pose_per_frame.svg'
        if using_own_axes:
            if fig is None:
                raise ValueError("Variable 'fig' is None. This should not have happened as it is set if"
                                 "no axes are provided and in this case, this code should not have been called.")

            self.log_pyplot(max(frame_indices), fig, fig_path, RerunAnnotations.pose_per_frame)
            plt.savefig(fig_path)
            plt.close()

    @staticmethod
    def convert_observation_to_numpy(observation):
        return observation[0, 0].permute(1, 2, 0).numpy(force=True)

    @staticmethod
    def overlay_occlusion(image, occlusion_mask):
        """
        Overlay an occlusion mask on an image.

        Args:
        - image: The original image as a numpy array of shape (H, W, C).
        - occlusion_mask: The occlusion mask as a numpy array of shape (H, W, 1), values in [0, 1].

        Returns:
        - The image with the occlusion mask overlay as a numpy array of shape (H, W, C).
        """
        occlusion_mask = occlusion_mask.squeeze() * 0.2  # Remove the singleton dimension if present
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):  # Grayscale or single-channel
            image = np.dstack([image] * 3)  # Convert to 3-channel for coloring

        white_overlay = np.ones_like(image) * 255
        overlay_image = (1 - occlusion_mask[..., np.newaxis]) * image + occlusion_mask[..., np.newaxis] * white_overlay

        return overlay_image.astype(image.dtype)

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
        y1, x1 = source_coords
        y2_f, x2_f = source_coords_to_target_coords_np(source_coords, flow)

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

    @torch.no_grad()
    def evaluate_metrics(self, stepi, tracking6d, keyframes, predicted_vertices, predicted_quaternion,
                         predicted_translation, predicted_mask, gt_vertices=None, gt_rotation=None, gt_translation=None,
                         gt_object_mask=None):

        encoder_result_all_frames, _ = self.encoder_result_all_frames(tracking6d.encoder, max(keyframes) + 1)

        chamfer_dist = "NA"
        iou_3d = "NA"
        miou_2d = "NA"
        last_iou_2d = "NA"
        mTransAll = "NA"
        mTransKF = "NA"
        transLast = "NA"
        mAngDiffAll = "NA"
        mAngDiffKF = "NA"
        angDiffLast = "NA"

        if gt_vertices is not None:
            chamfer_dist = float(chamfer_distance(predicted_vertices, gt_vertices)[0])

        if gt_rotation is not None:
            gt_quaternion = axis_angle_to_quaternion(gt_rotation)

            pred_quaternion_all_frames = encoder_result_all_frames.quaternions
            gt_quaternion_all_frames = gt_quaternion[:, :stepi + 1]

            pred_quaternion_keyframes = predicted_quaternion
            gt_quaternion_keyframes = gt_quaternion[:, keyframes]

            pred_quaternion_last = predicted_quaternion[None, :, -1]
            gt_quaternion_last = gt_quaternion[None, :, stepi]

            ang_diff_all_frames = quaternion_angular_difference(pred_quaternion_all_frames,
                                                                gt_quaternion_all_frames)
            mAngDiffAll = float(ang_diff_all_frames.mean())

            ang_diff_keyframes = quaternion_angular_difference(pred_quaternion_keyframes,
                                                               gt_quaternion_keyframes)
            mAngDiffKF = float(ang_diff_keyframes.mean())

            ang_diff_last_frame = quaternion_angular_difference(pred_quaternion_last,
                                                                gt_quaternion_last)
            angDiffLast = float(ang_diff_last_frame.mean())

        if gt_translation is not None:
            pred_translation_all_frames = encoder_result_all_frames.translations
            gt_translation_all_frames = gt_translation[:, :, :stepi + 1]

            pred_translation_keyframes = predicted_translation
            gt_translation_keyframes = gt_translation[:, :, keyframes]

            pred_translation_last = predicted_translation[None, :, :, -1]
            gt_translation_last = gt_translation[None, :, :, stepi]

            # Compute L2 norm for all frames
            translation_l2_diff_all_frames = torch.norm(pred_translation_all_frames - gt_translation_all_frames,
                                                        dim=-1)
            mTransAll = float(translation_l2_diff_all_frames.mean())

            # Compute L2 norm for keyframes
            translation_l2_diff_keyframes = torch.norm(pred_translation_keyframes - gt_translation_keyframes,
                                                       dim=-1)
            mTransKF = float(translation_l2_diff_keyframes.mean())

            # Compute L2 norm for the last frame
            translation_l2_diff_last = torch.norm(pred_translation_last - gt_translation_last, dim=-1)
            transLast = float(translation_l2_diff_last.mean())

        if gt_object_mask is not None:

            ious = torch.zeros(gt_object_mask.shape[1])
            for frame_i in range(gt_object_mask.shape[1]):
                frame_iou = 1 - iou_loss(predicted_mask[None, None, :, frame_i],
                                         gt_object_mask[None, None, :, frame_i])
                ious[frame_i] = frame_iou

            last_iou_2d = float(frame_iou)
            miou_2d = float(torch.mean(ious))

        # ["Frame", "mIoU", "lastIoU" "mIoU_3D", "ChamferDistance", "mTransAll", "mTransKF",
        #  "transLast", "mAngDiffAll", "mAngDiffKF", "angDiffLast"]
        row_results = [stepi, miou_2d, last_iou_2d, iou_3d, chamfer_dist, mTransAll, mTransKF, transLast,
                       mAngDiffAll, mAngDiffKF, angDiffLast]

        row_results_rounded = [round(res, 3) if type(res) is float else res for res in row_results]

        self.metrics_writer.writerow(row_results_rounded)
        self.metrics_log.flush()

    def visualize_rotations_per_epoch(self, frame_i):
        frame_data = self.data_graph.get_frame_data(frame_i)
        logged_sgd_translations = frame_data.translations_during_optimization
        logged_sgd_quaternions = frame_data.quaternions_during_optimization
        frame_losses = frame_data.frame_losses

        gt_rotation = frame_data.gt_rot_axis_angle
        gt_translation = frame_data.gt_translation

        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 12))  # Adjusted for two subplots, one above the other
        fig.subplots_adjust(left=0.25, right=0.85, hspace=0.5)

        # Current rotation and translation values
        translation_tensors = [t[0, 0, -1].detach().cpu() for t in logged_sgd_translations]
        rotation_tensors = [
            torch.rad2deg(quaternion_to_axis_angle(q.detach().cpu()))[0, -1]
            for q in logged_sgd_quaternions
        ]

        # Plot rotations
        axis_labels = ['X-axis rotation', 'Y-axis rotation', 'Z-axis rotation']
        colors = ['yellow', 'green', 'blue']
        for i in range(3):
            values = [tensor[i].item() for tensor in rotation_tensors]
            ax1.plot(range(len(rotation_tensors)), values, label=axis_labels[i], color=colors[i])
            # Plot GT rotations if provided
            if gt_rotation is not None:
                gt_rotation_deg = torch.rad2deg(gt_rotation)
                gt_rotation_values = gt_rotation_deg.numpy(force=True).repeat(len(rotation_tensors))
                ax1.plot(range(gt_rotation_values.shape[0]), gt_rotation_values, linestyle='dashed',
                         label=f"GT {axis_labels[i]}", alpha=0.5, color=colors[i])

        ax1.set_xlabel('Gradient descend iteration')
        ax1.set_ylabel('Rotation [degrees]')
        ax1.legend(loc='lower left')

        # Plot translations
        translation_axis_labels = ['X-axis translation', 'Y-axis translation', 'Z-axis translation']
        for i in range(3):
            values = [tensor[i].item() for tensor in translation_tensors]
            ax3.plot(range(len(translation_tensors)), values, label=translation_axis_labels[i], color=colors[i])
            # Plot GT translations if provided
            if gt_translation is not None:
                gt_translation_values = gt_translation.squeeze().numpy(force=True).repeat(len(translation_tensors))
                ax3.plot(range(gt_translation_values.shape[0]), gt_translation_values, linestyle='dashed',
                         label=f"GT {translation_axis_labels[i]}", alpha=0.5, color=colors[i])

        ax3.set_xlabel('Gradient descend iteration')
        ax3.set_ylabel('Translation')
        ax3.legend(loc='lower left')

        # Adjust the loss plot on the rotation axis for clarity
        self.add_loss_plot(ax1, frame_losses)
        self.add_loss_plot(ax3, frame_losses)

        # Saving the figure
        (Path(self.write_folder) / Path('rotations_by_epoch')).mkdir(exist_ok=True, parents=True)
        fig_path = (Path(self.write_folder) / Path('rotations_by_epoch') / f'rotations_by_epoch_frame_{frame_i}.svg')
        plt.savefig(fig_path)
        plt.close()

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

    def write_keyframe_rotations(self, detached_result, keyframes):
        quaternions = detached_result.quaternions[0]  # Assuming shape is (1, N, 4)
        for k in range(quaternions.shape[0]):
            quaternions[k] = normalize_quaternion(quaternions[k])
        # Convert quaternions to Euler angles
        angles_rad = quaternion_to_axis_angle(quaternions)
        # Convert radians to degrees
        angles_deg = angles_rad * 180.0 / math.pi
        rot_axes = ['X-axis: ', 'Y-axis: ', 'Z-axis: ']
        for k in range(angles_rad.shape[0]):
            rotations = [rot_axes[i] + str(round(float(angles_deg[k, i]), 3))
                         for i in range(3)]
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} rotation: " + str(rotations) + '\n')
        for k in range(detached_result.quaternions.shape[1]):
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} translation: str{detached_result.translations[0, 0, k]}\n")
        self.tracking_log.write('\n')
        self.tracking_log.flush()

    def write_all_encoder_rotations(self, encoder: Encoder, last_keyframe_idx):
        self.tracking_log.write("============================================\n")
        self.tracking_log.write("Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n")
        encoder_result_prime, keyframes_prime = self.encoder_result_all_frames(encoder, last_keyframe_idx)
        self.write_keyframe_rotations(encoder_result_prime, keyframes_prime)
        self.tracking_log.write("============================================\n")
        self.tracking_log.write("END of Writing all the states of the encoder\n")
        self.tracking_log.write("============================================\n\n\n")

    @staticmethod
    def encoder_result_all_frames(encoder: Encoder, last_keyframe_idx: int):
        keyframes_prime = list(range(last_keyframe_idx))
        encoder_result_prime = encoder(keyframes_prime)
        return encoder_result_prime, keyframes_prime

    def visualize_optimized_values(self, bounding_box, keyframe_buffer: KeyframeBuffer,
                                   new_flow_arcs: List[Tuple[int, int]]):
        for flow_arc_idx, flow_arc in enumerate(new_flow_arcs):

            source_frame = flow_arc[0]
            target_frame = flow_arc[1]

            # Get optical flow frames
            keyframes = [source_frame, target_frame]
            flow_frames = [source_frame, target_frame]

            # Compute estimated shape
            encoder_result, encoder_result_flow_frames = self.deep_encoder.frames_and_flow_frames_inference(keyframes,
                                                                                                            flow_frames)

            # Get texture map
            tex_rgb = nn.Sigmoid()(self.rgb_encoder.texture_map)

            # Render keyframe images
            rendering_result = self.rendering.forward(encoder_result.translations, encoder_result.quaternions,
                                                      encoder_result.vertices, self.deep_encoder.face_features, tex_rgb,
                                                      encoder_result.lights)

            rendering_rgb = rendering_result.rendered_image

            rendered_flow_result = self.rendering.compute_theoretical_flow(encoder_result, encoder_result_flow_frames,
                                                                           flow_arcs_indices=[(0, 1)])

            rendered_keyframe_images = self.write_tensor_into_bbox(rendering_rgb, bounding_box)

            # Extract current and previous rendered images
            source_rendered_image_rgb = rendered_keyframe_images[0, -2]
            target_rendered_image_rgb = rendered_keyframe_images[0, -1]

            theoretical_flow_path = self.optimized_values_path / Path(
                f"predicted_flow_{source_frame}_{target_frame}.png")
            flow_difference_path = self.optimized_values_path / Path(
                f"flow_difference_{source_frame}_{target_frame}.png")
            rendering_path = self.optimized_values_path / Path(f"rendering_{target_frame}.png")
            occlusion_path = self.optimized_values_path / Path(f"occlusion_{source_frame}_{target_frame}.png")

            rendered_occlusion_squeezed = rendered_flow_result.rendered_flow_occlusion.squeeze()
            self.visualize_1D_feature_map_using_overlay(target_frame, occlusion_path, source_rendered_image_rgb.squeeze(),
                                                        rendered_occlusion_squeezed, alpha=0.8,
                                                        rerun_annotation='/optimized_values/occlusion')

            # Save rendered images
            if flow_arc_idx == 0:
                target_rendered_image = (target_rendered_image_rgb * 255).permute(1, 2, 0).to(torch.uint8)
                self.log_image(target_frame, target_rendered_image, rendering_path, "/optimized_values/rendering")

            # Adjust (0, 1) range to pixel range
            theoretical_flow = rendered_flow_result.theoretical_flow[:, [-1]]
            theoretical_flow = flow_unit_coords_to_image_coords(theoretical_flow).squeeze().detach().clone().cpu()

            # Adjust (0, 1) range to pixel range
            observed_flow = keyframe_buffer.get_flows_between_frames(source_frame, target_frame).observed_flow
            observed_flow = flow_unit_coords_to_image_coords(observed_flow)
            observed_flow = observed_flow.squeeze().detach().clone().cpu()

            observed_flow_np = observed_flow.permute(1, 2, 0).numpy()
            theoretical_flow_np = theoretical_flow.permute(1, 2, 0).numpy()

            target_frame_observation = keyframe_buffer.get_observations_for_keyframe(target_frame)
            source_frame_observation = keyframe_buffer.get_observations_for_keyframe(source_frame)

            target_image_segmentation = target_frame_observation.observed_segmentation[
                0, 0, 0].detach().clone().cpu()
            source_image_segmentation = source_frame_observation.observed_segmentation[
                0, 0, 0].detach().clone().cpu()

            rendered_flow_occlusion_mask = rendered_flow_result.rendered_flow_occlusion[
                0, 0, 0].detach().clone().cpu()

            # Visualize flow and flow difference
            flow_illustration = visualize_flow_with_images([source_rendered_image_rgb],
                                                           target_rendered_image_rgb, observed_flows=None,
                                                           gt_flows=theoretical_flow_np,
                                                           gt_silhouette_current=target_image_segmentation,
                                                           gt_silhouettes_prev=[source_image_segmentation],
                                                           flow_occlusion_masks=[rendered_flow_occlusion_mask])

            flow_difference_illustration = compare_flows_with_images([source_rendered_image_rgb],
                                                                     target_rendered_image_rgb,
                                                                     [observed_flow_np], theoretical_flow_np,
                                                                     gt_silhouette_current=target_image_segmentation,
                                                                     gt_silhouette_prev=[source_image_segmentation])

            # Save flow illustrations
            imageio.imwrite(theoretical_flow_path, flow_illustration)
            imageio.imwrite(flow_difference_path, flow_difference_illustration)

    def visualize_observed_data(self, keyframe_buffer: KeyframeBuffer, flow_arcs):
        for flow_arcs in flow_arcs:

            source_frame = flow_arcs[0]
            target_frame = flow_arcs[1]

            flow_observation = keyframe_buffer.get_flows_between_frames(source_frame, target_frame)
            source_frame_observation = keyframe_buffer.get_observations_for_keyframe(source_frame)
            target_frame_observation = keyframe_buffer.get_observations_for_keyframe(target_frame)

            observed_flow = flow_observation.observed_flow.cpu()
            observed_flow_occlusions = flow_observation.observed_flow_occlusion.cpu()
            observed_flow_uncertainties = flow_observation.observed_flow_uncertainty.cpu()
            source_frame_image = source_frame_observation.observed_image.cpu()
            source_frame_segment = source_frame_observation.observed_segmentation.cpu()

            target_frame_image = target_frame_observation.observed_image.cpu()
            target_frame_segment = target_frame_observation.observed_segmentation.cpu()

            observed_flow = flow_unit_coords_to_image_coords(observed_flow)
            observed_flow_reordered = observed_flow.squeeze().permute(1, 2, 0).numpy()

            source_image_discrete: torch.Tensor = (source_frame_image * 255).to(torch.uint8).squeeze()
            target_image_discrete: torch.Tensor = (target_frame_image * 255).to(torch.uint8).squeeze()

            source_frame_segment_squeezed = source_frame_segment.squeeze()
            target_frame_segment_squeezed = target_frame_segment.squeeze()
            observed_flow_occlusions_squeezed = observed_flow_occlusions.squeeze()
            observed_flow_uncertainties_squeezed = observed_flow_uncertainties.squeeze()

            # TODO this computation is not mathematically justified, and serves just for visualization purposes
            observed_flow_uncertainties_0_1_range = (observed_flow_uncertainties_squeezed -
                                                     observed_flow_uncertainties.min())
            observed_flow_uncertainties_0_1_range /= observed_flow_uncertainties_0_1_range.max()

            flow_illustration = visualize_flow_with_images([source_image_discrete], target_image_discrete,
                                                           [observed_flow_reordered], None,
                                                           gt_silhouette_current=source_frame_segment_squeezed,
                                                           gt_silhouettes_prev=[target_frame_segment_squeezed],
                                                           flow_occlusion_masks=[observed_flow_occlusions_squeezed])

            uncertainty_illustration = visualize_flow_with_images([source_image_discrete], target_image_discrete,
                                                           [observed_flow_reordered], None,
                                                           gt_silhouette_current=source_frame_segment_squeezed,
                                                           gt_silhouettes_prev=[target_frame_segment_squeezed],
                                                           flow_occlusion_masks=[observed_flow_uncertainties_0_1_range])

            # Define output file paths
            template_image_path = self.observations_path / Path(f'template_img_{source_frame}_{target_frame}.png')
            new_image_path = self.observations_path / Path(f'gt_img_{source_frame}_{target_frame}.png')
            observed_flow_path = self.observations_path / Path(f'flow_{source_frame}_{target_frame}.png')
            observed_flow_uncertainty_path = (self.observations_path /
                                              Path(f'flow_uncertainty_{source_frame}_{target_frame}.png'))
            occlusion_path = self.observations_path / Path(f"occlusion_{source_frame}_{target_frame}.png")
            uncertainty_path = self.observations_path / Path(f"uncertainty_{source_frame}_{target_frame}.png")

            self.visualize_1D_feature_map_using_overlay(target_frame, occlusion_path, source_frame_image.squeeze(),
                                                        observed_flow_occlusions_squeezed, alpha=0.8,
                                                        rerun_annotation=RerunAnnotations.observed_flow_occlusion)
            # Uncertainty visualizations
            self.visualize_1D_feature_map_using_overlay(target_frame, uncertainty_path, source_frame_image.squeeze(),
                                                        observed_flow_uncertainties_0_1_range, alpha=0.8,
                                                        rerun_annotation=RerunAnnotations.observed_flow_uncertainty)

            # Save the images to disk
            self.log_image(target_frame, target_image_discrete.permute(1, 2, 0), new_image_path,
                           RerunAnnotations.observed_image)
            self.log_image(target_frame, source_image_discrete.permute(1, 2, 0), template_image_path,
                           RerunAnnotations.template_image)

            flow_illustration_torch = (
                torchvision.transforms.functional.pil_to_tensor(flow_illustration).permute(1, 2, 0))
            flow_illustration_uncertainty_torch = (
                torchvision.transforms.functional.pil_to_tensor(uncertainty_illustration).permute(1, 2, 0))

            self.log_image(target_frame, flow_illustration_uncertainty_torch, observed_flow_uncertainty_path,
                           RerunAnnotations.observed_flow_with_uncertainty, ignore_dimensions=True)

            self.log_image(target_frame, flow_illustration_torch, observed_flow_path, RerunAnnotations.observed_flow,
                           ignore_dimensions=True)

    def visualize_1D_feature_map_using_overlay(self, flow_target_frame, occlusion_path, source_image_rgb, flow_occlusion, alpha,
                                               rerun_annotation):
        assert flow_occlusion.shape == (self.image_height, self.image_width)
        assert source_image_rgb.shape == (3, self.image_height, self.image_width)

        occlusion_mask = flow_occlusion.detach().unsqueeze(0).repeat(3, 1, 1)
        blended_image = alpha * occlusion_mask + (1 - alpha) * source_image_rgb
        blended_image = (blended_image * 255).to(torch.uint8).squeeze().permute(1, 2, 0)

        self.log_image(flow_target_frame, blended_image, occlusion_path, rerun_annotation)

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
            rr.log(rerun_annotation, rr.Image(image))
            rr.set_time_sequence("frame", frame)
        else:
            plt.savefig(str(save_path), **kwargs)
            plt.close()

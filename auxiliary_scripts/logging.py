from collections import defaultdict
from dataclasses import dataclass
from itertools import product

from typing import Dict, Tuple, List, Any

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
from kornia.geometry import normalize_quaternion, Se3, Quaternion, PinholeCamera
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch, Patch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from kornia.geometry.conversions import quaternion_to_axis_angle, axis_angle_to_quaternion
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.io import save_ply

from auxiliary_scripts.data_utils import load_texture, load_mesh_using_trimesh
from auxiliary_scripts.image_utils import ImageShape, overlay_occlusion
from auxiliary_scripts.visualizations import visualize_optical_flow_errors
from data_structures.keyframe_buffer import FrameObservation, FlowObservation, KeyframeBuffer
from data_structures.pose_icosphere import PoseIcosphere
from models.loss import iou_loss, FMOLoss
from tracker_config import TrackerConfig
from data_structures.data_graph import DataGraph
from auxiliary_scripts.cameras import Cameras
from utils import coordinates_xy_to_tensor_index, normalize_vertices
from auxiliary_scripts.math_utils import quaternion_angular_difference, Se3_last_cam_to_world_from_Se3_obj
from models.rendering import infer_normalized_renderings, RenderingKaolin
from models.encoder import EncoderResult, Encoder
from flow import visualize_flow_with_images, compare_flows_with_images, flow_unit_coords_to_image_coords, \
    source_coords_to_target_coords_image, get_non_occluded_foreground_correspondences, source_coords_to_target_coords, \
    get_correct_correspondences_mask, source_coords_to_target_coords_np


@dataclass
class RerunAnnotations:
    # Observations
    space_visualization: str = '/3d_space'
    space_gt_mesh: str = '/3d_space/gt_mesh'
    space_gt_camera_pose: str = '/3d_space/gt_camera_pose'
    space_predicted_camera_pose: str = '/3d_space/predicted_camera_pose'
    space_gt_camera_track: str = '/3d_space/gt_camera_track'
    space_predicted_camera_track: str = '/3d_space/predicted_camera_track'
    space_predicted_camera_keypoints: str = '/3d_space/predicted_camera_keypoints'
    space_predicted_closest_keypoint: str = '/3d_space/predicted_closest_keypoint'

    template_image_frontview: str = '/observations/template_image_frontview'

    template_image_segmentation_frontview: str = '/observations/template_image_frontview/segment'

    observed_image_frontview: str = '/observations/observed_image_frontview'

    observed_image_segmentation_frontview: str = '/observations/observed_image_frontview/segment'

    observed_flow_frontview: str = '/observed_flow/observed_flow_frontview'
    observed_flow_occlusion_frontview: str = '/observed_flow/occlusion_frontview'
    observed_flow_uncertainty_frontview: str = '/observed_flow/uncertainty_frontview'
    observed_flow_with_uncertainty_frontview: str = '/observed_flow/observed_flow_front_uncertainty'
    observed_flow_errors_frontview: str = '/observed_flow/observed_flow_gt_disparity'

    # Optimized model visualizations
    optimized_model_occlusion: str = '/optimized_values/occlusion'
    optimized_model_render: str = '/optimized_values/rendering'

    # Triangulated points RANSAC
    triangulated_points_gt_Rt_gt_flow: str = '/point_clouds/triangulated_points_gt_Rt_gt_flow'
    triangulated_points_gt_Rt_mft_flow: str = '/point_clouds/triangulated_points_gt_Rt_mft_flow'
    point_cloud_dust3r_im1: str = '/point_clouds/point_cloud_dust3r_im1'
    point_cloud_dust3r_im2: str = '/point_clouds/point_cloud_dust3r_im2'

    # Ransac
    matching_correspondences_inliers: str = '/epipolar/matching/correspondences_inliers'
    matching_correspondences_outliers: str = '/epipolar/matching/correspondences_outliers'

    ransac_stats_frontview: str = '/epipolar/ransac_stats_frontview'
    ransac_stats_frontview_visible: str = '/epipolar/ransac_stats_frontview/visible'
    ransac_stats_frontview_predicted_as_visible: str = '/epipolar/ransac_stats_frontview/predicted_as_visible'
    ransac_stats_frontview_correctly_predicted_flows: str = '/epipolar/ransac_stats_frontview/correctly_predicted_flows'
    ransac_stats_frontview_ransac_predicted_inliers: str = '/epipolar/ransac_stats_frontview/ransac_predicted_inliers'
    ransac_stats_frontview_correctly_predicted_inliers: str = '/epipolar/ransac_stats_frontview/correctly_predicted_inliers'
    ransac_stats_frontview_ransac_inlier_ratio: str = '/epipolar/ransac_stats_frontview/ransac_inlier_ratio'

    # Pose
    pose_estimation_timing: str = '/pose/timing/'
    pose_estimation_time: str = '/pose/timing/pose_estimation_time'
    long_short_chain_remaining_pts: str = '/pose/chaining/remaining_pts/remaining_percent'
    long_short_chain_diff_template: str = '/pose/chaining/template'
    chained_pose_polar: str = '/pose/chaining/polar_angle'
    chained_pose_polar_template: str = '/pose/chaining/polar_angle/template'
    chained_pose_long_flow_polar: str = '/pose/chaining/polar_angle/long_flow'
    chained_pose_short_flow_polar: str = '/pose/chaining/polar_angle/short_flow'
    long_short_chain_diff: str = '/pose/chaining/polar_angle/difference'

    chained_pose_long_flow: str = '/pose/chaining/long_flow'
    chained_pose_long_flow_template: str = '/pose/chaining/long_flow/template'
    chained_pose_long_flow_x: str = '/pose/chaining/long_flow/x_axis'
    chained_pose_long_flow_y: str = '/pose/chaining/long_flow/y_axis'
    chained_pose_long_flow_z: str = '/pose/chaining/long_flow/z_axis'

    chained_pose_short_flow: str = '/pose/chaining/short_flow'
    chained_pose_short_flow_template: str = '/pose/chaining/short_flow/template'
    chained_pose_short_flow_x: str = '/pose/chaining/short_flow/x_axis'
    chained_pose_short_flow_y: str = '/pose/chaining/short_flow/y_axis'
    chained_pose_short_flow_z: str = '/pose/chaining/short_flow/z_axis'

    cam_delta_short_flow: str = '/pose/cam/short_flow/'
    cam_delta_short_flow_template = '/pose/cam/short_flow/template'
    cam_delta_short_flow_zaragoza: str = '/pose/cam/short_flow/cam_pose_delta_zaragoza'
    cam_delta_short_flow_RANSAC: str = '/pose/cam/short_flow/cam_pose_delta_RANSAC'

    cam_delta_long_flow: str = '/pose/cam/long_flow/'
    cam_delta_long_flow_template = '/pose/cam/long_flow/template'
    cam_delta_long_flow_zaragoza: str = '/pose/cam/long_flow/cam_pose_delta_zaragoza'
    cam_delta_long_flow_RANSAC: str = '/pose/cam/long_flow/cam_pose_delta_RANSAC'

    obj_rot_1st_to_last: str = '/pose/rotation'
    obj_rot_1st_to_last_x: str = '/pose/rotation/x_axis'
    obj_rot_1st_to_last_x_gt: str = '/pose/rotation/x_axis_gt'
    obj_rot_1st_to_last_y: str = '/pose/rotation/y_axis'
    obj_rot_1st_to_last_y_gt: str = '/pose/rotation/y_axis_gt'
    obj_rot_1st_to_last_z: str = '/pose/rotation/z_axis'
    obj_rot_1st_to_last_z_gt: str = '/pose/rotation/z_axis_gt'

    obj_tran_1st_to_last: str = '/pose/translation'
    obj_tran_1st_to_last_x: str = '/pose/translation/x_axis'
    obj_tran_1st_to_last_x_gt: str = '/pose/translation/x_axis_gt'
    obj_tran_1st_to_last_y: str = '/pose/translation/y_axis'
    obj_tran_1st_to_last_y_gt: str = '/pose/translation/y_axis_gt'
    obj_tran_1st_to_last_z: str = '/pose/translation/z_axis'
    obj_tran_1st_to_last_z_gt: str = '/pose/translation/z_axis_gt'

    cam_rot_ref_to_last: str = '/pose/cam_rot_ref_to_last'
    cam_rot_ref_to_last_template: str = '/pose/cam_rot_ref_to_last/template'
    cam_rot_ref_to_last_x: str = '/pose/cam_rot_ref_to_last/x_axis'
    cam_rot_ref_to_last_x_gt: str = '/pose/cam_rot_ref_to_last/x_axis_gt'
    cam_rot_ref_to_last_y: str = '/pose/cam_rot_ref_to_last/y_axis'
    cam_rot_ref_to_last_y_gt: str = '/pose/cam_rot_ref_to_last/y_axis_gt'
    cam_rot_ref_to_last_z: str = '/pose/cam_rot_ref_to_last/z_axis'
    cam_rot_ref_to_last_z_gt: str = '/pose/cam_rot_ref_to_last/z_axis_gt'

    cam_tran_ref_to_last: str = '/pose/cam_tran_ref_to_last'
    cam_tran_ref_to_last_x: str = '/pose/cam_tran_ref_to_last/x_axis'
    cam_tran_ref_to_last_x_gt: str = '/pose/cam_tran_ref_to_last/x_axis_gt'
    cam_tran_ref_to_last_y: str = '/pose/cam_tran_ref_to_last/y_axis'
    cam_tran_ref_to_last_y_gt: str = '/pose/cam_tran_ref_to_last/y_axis_gt'
    cam_tran_ref_to_last_z: str = '/pose/cam_tran_ref_to_last/z_axis'
    cam_tran_ref_to_last_z_gt: str = '/pose/cam_tran_ref_to_last/z_axis_gt'

    obj_rot_ref_to_last: str = '/pose/obj_rot_ref_to_last'
    obj_rot_ref_to_last_template: str = '/pose/obj_rot_ref_to_last/template'
    obj_rot_ref_to_last_x: str = '/pose/obj_rot_ref_to_last/x_axis'
    obj_rot_ref_to_last_x_gt: str = '/pose/obj_rot_ref_to_last/x_axis_gt'
    obj_rot_ref_to_last_y: str = '/pose/obj_rot_ref_to_last/y_axis'
    obj_rot_ref_to_last_y_gt: str = '/pose/obj_rot_ref_to_last/y_axis_gt'
    obj_rot_ref_to_last_z: str = '/pose/obj_rot_ref_to_last/z_axis'
    obj_rot_ref_to_last_z_gt: str = '/pose/obj_rot_ref_to_last/z_axis_gt'

    obj_tran_ref_to_last: str = '/pose/obj_tran_ref_to_last'
    obj_tran_ref_to_last_x: str = '/pose/obj_tran_ref_to_last/x_axis'
    obj_tran_ref_to_last_x_gt: str = '/pose/obj_tran_ref_to_last/x_axis_gt'
    obj_tran_ref_to_last_y: str = '/pose/obj_tran_ref_to_last/y_axis'
    obj_tran_ref_to_last_y_gt: str = '/pose/obj_tran_ref_to_last/y_axis_gt'
    obj_tran_ref_to_last_z: str = '/pose/obj_tran_ref_to_last/z_axis'
    obj_tran_ref_to_last_z_gt: str = '/pose/obj_tran_ref_to_last/z_axis_gt'

    # Pose
    pose_per_frame: str = '/pose/pose_per_frame'


class WriteResults:

    def __init__(self, write_folder, shape: ImageShape, tracking_config: TrackerConfig, rendering, gt_encoder,
                 deep_encoder, rgb_encoder, data_graph: DataGraph, cameras: List[Cameras],
                 pinhole_params):

        self.image_height = shape.height
        self.image_width = shape.width

        self.cameras: List[Cameras] = cameras
        self.pinhole_params: Dict[Cameras, PinholeCamera] = pinhole_params

        self.data_graph: DataGraph = data_graph

        self.rendering: RenderingKaolin = rendering
        self.gt_encoder: Encoder = gt_encoder
        self.deep_encoder: Encoder = deep_encoder
        self.rgb_encoder: Encoder = rgb_encoder

        self.logged_flow_tracks_inits: Dict[Cameras, List] = defaultdict(list)

        self.tracking_config: TrackerConfig = tracking_config
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

        self.init_directories()

        self.template_fields: List[str] = []

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

    def init_directories(self):
        self.observations_path.mkdir(exist_ok=True, parents=True)
        self.gt_values_path.mkdir(exist_ok=True, parents=True)
        self.optimized_values_path.mkdir(exist_ok=True, parents=True)
        self.rerun_log_path.mkdir(exist_ok=True, parents=True)
        self.ransac_path.mkdir(exist_ok=True, parents=True)
        self.point_clouds_path.mkdir(exist_ok=True, parents=True)
        self.exported_mesh_path.mkdir(exist_ok=True, parents=True)

    def rerun_init(self):
        rr.init(f'{self.tracking_config.sequence}-{self.tracking_config.experiment_name}')
        rr.save(
            self.rerun_log_path / f'rerun_{self.tracking_config.experiment_name}_{self.tracking_config.sequence}.rrd')

        self.template_fields = {
            RerunAnnotations.chained_pose_polar_template,
            RerunAnnotations.chained_pose_long_flow_template,
            RerunAnnotations.cam_delta_short_flow_template,
            RerunAnnotations.long_short_chain_diff_template,
            RerunAnnotations.cam_rot_ref_to_last_template,
            RerunAnnotations.chained_pose_short_flow_template,
            RerunAnnotations.obj_rot_ref_to_last_template,
            RerunAnnotations.cam_delta_long_flow_template,
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
                                                       origin=RerunAnnotations.ransac_stats_frontview
                                                       ),
                                    rrb.TimeSeriesView(name="Pose - Rotation",
                                                       origin=RerunAnnotations.obj_rot_1st_to_last
                                                       ),
                                    # rrb.TimeSeriesView(name="Pose - Translation",
                                    #                    origin=RerunAnnotations.obj_tran_1st_to_last
                                    #                    ),
                                ],
                                grid_columns=1,
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
                                    rrb.TimeSeriesView(name="Cam Delta Short Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_short_flow
                                                       ),
                                    rrb.TimeSeriesView(name="Cam Delta Long Flow Zaragoza vs RANSAC",
                                                       origin=RerunAnnotations.cam_delta_long_flow
                                                       ),
                                ],
                                grid_columns=1,
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
            for movement_type in ['rot', 'tran']:
                annotations |= set(map(
                    lambda annotation: (annotation, c),
                    [
                        getattr(RerunAnnotations, f'obj_{movement_type}_1st_to_last_{axis}'),
                        getattr(RerunAnnotations, f'obj_{movement_type}_ref_to_last_{axis}'),
                        getattr(RerunAnnotations, f'cam_{movement_type}_ref_to_last_{axis}'),
                        getattr(RerunAnnotations, f'chained_pose_long_flow_{axis}'),
                        getattr(RerunAnnotations, f'chained_pose_short_flow_{axis}'),
                    ]
                ))

        for axis, c in gt_axes_colors.items():
            for movement_type in ['rot', 'tran']:
                annotations |= set(map(
                    lambda annotation: (annotation, c),
                    [
                        getattr(RerunAnnotations, f'obj_{movement_type}_1st_to_last_{axis}_gt'),
                        getattr(RerunAnnotations, f'obj_{movement_type}_ref_to_last_{axis}_gt'),
                        getattr(RerunAnnotations, f'cam_{movement_type}_ref_to_last_{axis}_gt'),
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
                       extent=(float(translations_space[0]), float(translations_space[-1]),
                               float(rotations_space[-1]), float(rotations_space[0])),
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
                iteration_rotation_deg = float(torch.rad2deg(iteration_rotation_rad)[0, -1, rot_axis_idx])

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
                                                    keyframes_encoder_result=encoder_result)

                losses_all, losses, joint_loss, per_pixel_error = loss_result

                joint_losses[i, j] = joint_loss

        return joint_losses

    def set_tensorboard_log_for_frame(self, frame_i):
        self.tensorboard_log = SummaryWriter(str(self.tensorboard_log_dir / f'Frame_{frame_i}'))

    def write_into_tensorboard_log(self, sgd_iter: int, values_dict: Dict):

        for field_name, value in values_dict.items():
            self.tensorboard_log.add_scalar(field_name, value, sgd_iter)

    def write_tensor_into_bbox(self, image, bounding_box):
        image_with_margins = torch.zeros(image.shape[:-2] + self.image_width).to(image.device)
        image_with_margins[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]] = \
            image[..., bounding_box[0]:bounding_box[1], bounding_box[2]:bounding_box[3]]
        return image_with_margins

    @torch.no_grad()
    def write_results(self, frame_i, tex, active_keyframes: KeyframeBuffer, best_model,
                      observations: FrameObservation, flow_tracks_inits: List[int],
                      pose_icosphere: PoseIcosphere):

        observed_segmentations = observations.observed_segmentation

        for camera in self.cameras:
            self.visualize_observed_data(active_keyframes, frame_i, camera)

            if not self.tracking_config.write_to_rerun_rather_than_disk:
                self.visualize_flow_with_matching(frame_i)
            self.visualize_rotations_per_epoch(frame_i)

        if self.tracking_config.visualize_outliers_distribution:
            datagraph_camera_data = self.data_graph.get_camera_specific_frame_data(frame_i, Cameras.FRONTVIEW)
            new_flow_arc = (datagraph_camera_data.long_jump_source, frame_i)
            self.visualize_outliers_distribution(new_flow_arc)

        self.visualize_3d_camera_space(frame_i, pose_icosphere)

        encoder_result = self.data_graph.get_frame_data(frame_i).encoder_result
        detached_result = EncoderResult(*[it.clone().detach() if type(it) is torch.Tensor else it
                                          for it in encoder_result])
        if self.tracking_config.save_3d_model:
            self.save_3d_model(frame_i, tex, best_model, detached_result)

        if self.tracking_config.write_to_rerun_rather_than_disk:
            self.log_poses_into_rerun(frame_i)
            self.visualize_flow_with_matching_rerun(frame_i)

        if self.tracking_config.preinitialization_method == 'essential_matrix_decomposition':
            if self.tracking_config.analyze_ransac_matching_errors:
                self.analyze_ransac_matchings_errors(frame_i)

            if (self.tracking_config.analyze_ransac_matchings and
                    frame_i % self.tracking_config.analyze_ransac_matchings_frequency == 0):
                self.analyze_ransac_matchings(frame_i, flow_tracks_inits)

            if self.tracking_config.visualize_point_clouds_from_ransac:
                self.visualize_point_clouds_from_ransac(frame_i)

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

        # self.render_silhouette_overlap(rendered_silhouette[:, [-1]],
        #                                observed_segmentations[:, [-1]], frame_i)

        for tmpi in range(renders.shape[1]):
            segmentations_discrete = (observed_segmentations[:, -1:, [-1]] > 0).to(observed_segmentations.dtype)
            self.baseline_iou[frame_i - 1] = 1 - iou_loss(segmentations_discrete,
                                                          observed_segmentations[:, -1:, [-1]]).detach().cpu()
            self.our_iou[frame_i - 1] = 1 - iou_loss(rendered_silhouette[:, [-1]],
                                                     observed_segmentations[:, -1:, [-1]]).detach().cpu()

        print('Baseline IoU {}, our IoU {}'.format(self.baseline_iou[frame_i - 1], self.our_iou[frame_i - 1]))

    def visualize_3d_camera_space(self, frame_i: int, pose_icosphere: PoseIcosphere):

        rr.set_time_sequence(RerunAnnotations.space_visualization, frame_i)

        all_frames_from_0 = range(0, frame_i+1)
        n_poses = len(all_frames_from_0)

        T_world_to_cam_se3 = Se3.from_matrix(self.pinhole_params[Cameras.FRONTVIEW].extrinsics)
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

        strips_radii_factor = (max(torch.max(gt_translations.norm(dim=1)).item(), 5.) / 5.)
        strips_radii = [0.01 * strips_radii_factor] * n_poses

        rr.log(RerunAnnotations.space_gt_camera_track,
               rr.LineStrips3D(strips=strips_gt,  # gt_t_cam
                               colors=colors_gt,
                               radii=strips_radii))

        rr.log(RerunAnnotations.space_predicted_camera_track,
               rr.LineStrips3D(strips=strips_pred,  # pred_t_cam
                               colors=colors_pred,
                               radii=strips_radii))

        datagraph_camera_node = self.data_graph.get_camera_specific_frame_data(frame_i, Cameras.FRONTVIEW)
        template_frame_idx = datagraph_camera_node.long_jump_source
        datagraph_template_node = self.data_graph.get_frame_data(template_frame_idx)

        template_node_Se3 = datagraph_template_node.predicted_object_se3_total
        template_node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(template_node_Se3, T_world_to_cam_se3)

        rr.log(RerunAnnotations.space_predicted_closest_keypoint,
               rr.LineStrips3D(strips=[[pred_t_cam[-1],
                                        template_node_cam_se3.translation.squeeze().numpy(force=True)]],
                               colors=[[255, 0, 0]],
                               radii=[0.025 * strips_radii_factor]))

        for i, icosphere_node in enumerate(pose_icosphere.reference_poses):

            if icosphere_node.keyframe_idx_observed not in self.logged_flow_tracks_inits[Cameras.FRONTVIEW]:
                template_idx = len(self.logged_flow_tracks_inits[Cameras.FRONTVIEW])

                template = icosphere_node.observation.observed_image[0, 0].permute(1, 2, 0).numpy(force=True)

                self.logged_flow_tracks_inits[Cameras.FRONTVIEW].append(icosphere_node.keyframe_idx_observed)
                template_image_grid_annotation = (f'{RerunAnnotations.space_predicted_camera_keypoints}/'
                                                  f'{template_idx}')
                rr.log(template_image_grid_annotation, rr.Image(template))

                for template_annotation in self.template_fields:
                    rr.log(template_annotation, rr.Scalar(0.0))

            node_Se3 = Se3(icosphere_node.quaternion, torch.zeros(1, 3).cuda())
            node_cam_se3 = Se3_last_cam_to_world_from_Se3_obj(node_Se3, T_world_to_cam_se3)
            node_cam_q_xyzw = node_cam_se3.quaternion.q[:, [1, 2, 3, 0]]

            rr.log(
                f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                rr.Transform3D(translation=node_cam_se3.translation.squeeze().numpy(force=True),
                               rotation=rr.Quaternion(xyzw=node_cam_q_xyzw.squeeze().numpy(force=True)))
            )

            rr.log(
                f'{RerunAnnotations.space_predicted_camera_keypoints}/{i}',
                rr.Pinhole(
                    resolution=[self.image_width, self.image_height],
                    focal_length=[float(self.pinhole_params[Cameras.FRONTVIEW].fx.item()),
                                  float(self.pinhole_params[Cameras.FRONTVIEW].fy.item())],
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

    def save_3d_model(self, frame_i, tex, best_model, detached_result):

        mesh_i_th_path = self.exported_mesh_path / f'mesh_{frame_i}.obj'
        tex_path = self.exported_mesh_path / 'tex_deep.png'
        tex_i_th_path = self.exported_mesh_path / f'tex_{frame_i}.png'
        model_path = self.exported_mesh_path / 'model.mtl'
        model_i_th_path = self.exported_mesh_path / f"model_{frame_i}.mtl"

        self.write_obj_mesh(detached_result.vertices[0].numpy(force=True), best_model["faces"],
                            self.deep_encoder.face_features[0].numpy(force=True), mesh_i_th_path,
                            str(model_i_th_path.name))

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

        in_edges = self.data_graph.G.in_edges(frame_i, data=False)
        flow_source = int(min(e[0] for e in in_edges))

        arc_data = self.data_graph.get_edge_observations(flow_source, frame_i, Cameras.FRONTVIEW)

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

    def measure_ransac_stats(self, frame_i, camera=Cameras.FRONTVIEW):
        correct_threshold = self.tracking_config.ransac_feed_only_inlier_flow_epe_threshold
        results = defaultdict(list)

        for i in range(1, frame_i + 1):
            in_edges = self.data_graph.G.in_edges(i, data=False)
            flow_arc = (min(e[0] for e in in_edges), i)

            arc_data = self.data_graph.get_edge_observations(*flow_arc, camera=camera)

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

            for camera in self.cameras:
                front_results = self.measure_ransac_stats(frame_i, camera)

                mft_flow_gt_flow_difference_front = front_results.pop('mft_flow_gt_flow_difference')

                if self.tracking_config.plot_mft_flow_kde_error_plot and camera == Cameras.FRONTVIEW:
                    self.plot_distribution_of_inliers_errors(mft_flow_gt_flow_difference_front)

    def analyze_ransac_matchings(self, frame_i, flow_tracks_inits):

        if frame_i % 10 == 0:
            return

        for camera in self.cameras:
            ransac_stats = self.measure_ransac_stats(frame_i, camera)

            ransac_stats.pop('mft_flow_gt_flow_difference')

            # We want each line to have its assigned color
            for i, metric in enumerate(ransac_stats.keys()):
                if metric == 'model_obtained_from':
                    continue

                rerun_time_series_entity = getattr(RerunAnnotations, f'ransac_stats_{camera.value}_{metric}')

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

        datagraph_frontview_data = self.data_graph.get_camera_specific_frame_data(frame_i, Cameras.FRONTVIEW)
        new_flow_arcs = [(datagraph_frontview_data.long_jump_source, frame_i)]

        for new_flow_arc in new_flow_arcs:

            flow_arc_source, flow_arc_target = new_flow_arc

            fig, axs = plt.subplots(3, len(self.cameras), figsize=(8, 12), dpi=600)
            axs: Any = axs

            flow_source_label = self.tracking_config.gt_flow_source
            if flow_source_label == 'FlowNetwork':
                flow_source_label = self.tracking_config.long_flow_model

            heading_text = (f"Frames {new_flow_arc}\n"
                            f"Flow: {flow_source_label}")

            if len(self.cameras) == 1:
                axs = np.atleast_2d(axs).T

            axs[0, 0].text(1.05, 1, heading_text, transform=axs[0, 0].transAxes, verticalalignment='top',
                           fontsize='medium')

            for i, camera in enumerate(self.cameras):

                arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target, camera)

                rendered_flow_res = arc_observation.synthetic_flow_result

                rend_flow = flow_unit_coords_to_image_coords(rendered_flow_res.observed_flow)
                rend_flow_np = rend_flow.numpy(force=True)

                flow_observation = arc_observation.observed_flow
                opt_flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)
                occlusion_mask = self.convert_observation_to_numpy(flow_observation.observed_flow_occlusion)
                occlusion_mask_thresh = np.greater_equal(occlusion_mask, self.tracking_config.occlusion_coef_threshold)
                segmentation_mask = flow_observation.observed_flow_segmentation.numpy(force=True)

                template_data = self.data_graph.get_camera_specific_frame_data(flow_arc_source, camera)
                target_data = self.data_graph.get_camera_specific_frame_data(flow_arc_target, camera)
                template_observation_frontview = template_data.frame_observation
                target_observation_frontview = target_data.frame_observation
                template_image = self.convert_observation_to_numpy(template_observation_frontview.observed_image)
                target_image = self.convert_observation_to_numpy(target_observation_frontview.observed_image)

                template_overlay = overlay_occlusion(template_image, occlusion_mask_thresh.astype(np.float32))

                display_bounds = (0, self.image_width, 0, self.image_height)

                for ax in axs.flat:
                    ax.axis('off')

                darkening_factor = 0.5
                axs[0, i].imshow(template_overlay * darkening_factor, extent=display_bounds)
                axs[0, i].set_title(f'Template {camera} occlusion')

                axs[1, i].imshow(template_image * darkening_factor, extent=display_bounds)
                axs[1, i].set_title(f'Template {camera}')

                axs[2, i].imshow(target_image * darkening_factor, extent=display_bounds)
                axs[2, i].set_title(f'Target {camera}')

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

        datagraph_camera_data = self.data_graph.get_camera_specific_frame_data(frame_i, Cameras.FRONTVIEW)
        new_flow_arc = (datagraph_camera_data.long_jump_source, frame_i)
        flow_arc_source, flow_arc_target = new_flow_arc

        rr.set_time_sequence('frame', flow_arc_target)

        camera = Cameras.FRONTVIEW
        arc_observation = self.data_graph.get_edge_observations(flow_arc_source, flow_arc_target, camera)

        flow_observation = arc_observation.observed_flow
        opt_flow = flow_unit_coords_to_image_coords(flow_observation.observed_flow)
        opt_flow_np = opt_flow.numpy(force=True)

        template_data = self.data_graph.get_camera_specific_frame_data(flow_arc_source, camera)
        target_data = self.data_graph.get_camera_specific_frame_data(flow_arc_target, camera)
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

        new_flow_arc_data = self.data_graph.get_edge_observations(*new_flow_arc, camera=Cameras.FRONTVIEW)
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

    def dump_correspondences(self, keyframes: KeyframeBuffer, new_flow_arcs, gt_rotations, gt_translations):

        with h5py.File(self.correspondences_log_file, 'a') as f:

            for flow_arc in new_flow_arcs:
                source_frame, target_frame = flow_arc

                flow_observation_frontview = keyframes.get_flows_between_frames(source_frame, target_frame)
                flow_observation_frontview.observed_flow = flow_unit_coords_to_image_coords(
                    flow_observation_frontview.observed_flow)

                dst_pts_xy_frontview, occlusion_score_frontview, src_pts_xy_frontview, gt_inliers_frontview = \
                    self.get_correspondences_from_observations(flow_observation_frontview, source_frame, target_frame)

                frame_data = {
                    "flow_arc": flow_arc,
                    "gt_translation": gt_translations[0, 0, target_frame].numpy(force=True),
                    "gt_rotation_change": torch.rad2deg(gt_rotations[0, target_frame]).numpy(force=True),
                    "source_points_xy_frontview": src_pts_xy_frontview.numpy(force=True),
                    "target_points_xy_frontview": dst_pts_xy_frontview.numpy(force=True),
                    "gt_inliers_frontview": gt_inliers_frontview.numpy(force=True),
                    "frontview_matching_occlusion": occlusion_score_frontview.numpy(force=True),
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

    @staticmethod
    def add_loss_plot(ax_, frame_losses_, indices=None):
        if indices is None:
            indices = range(len(frame_losses_))
        ax_loss = ax_.twinx()
        ax_loss.set_ylabel('Loss')
        ax_loss.plot(indices, frame_losses_, color='red', label='Loss')
        ax_loss.spines.right.set_position(("axes", 1.15))
        ax_loss.legend(loc='upper left')

    def log_poses_into_rerun(self, frame_i: int):

        data_graph_node = self.data_graph.get_frame_data(frame_i)
        camera_specific_graph_node = self.data_graph.get_camera_specific_frame_data(frame_i)

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

        pred_obj_quat_ref_to_last = datagraph_long_edge.predicted_obj_delta_se3.quaternion
        pred_cam_quat_ref_to_last = datagraph_long_edge.predicted_cam_delta_se3.quaternion

        pred_cam_RANSAC_quat_ref_to_last = datagraph_long_edge.predicted_cam_delta_se3_ransac.quaternion
        pred_cam_quat_prev_to_last = datagraph_short_edge.predicted_cam_delta_se3.quaternion
        pred_cam_RANSAC_quat_prev_to_last = datagraph_short_edge.predicted_cam_delta_se3_ransac.quaternion

        rr.log(RerunAnnotations.cam_delta_long_flow_zaragoza,
               rr.Scalar(torch.rad2deg(2 * pred_cam_quat_ref_to_last.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_long_flow_RANSAC,
               rr.Scalar(torch.rad2deg(2 * pred_cam_RANSAC_quat_ref_to_last.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_short_flow_zaragoza,
               rr.Scalar(torch.rad2deg(2 * pred_cam_quat_prev_to_last.polar_angle).cpu()))
        rr.log(RerunAnnotations.cam_delta_short_flow_RANSAC,
               rr.Scalar(torch.rad2deg(2 * pred_cam_RANSAC_quat_prev_to_last.polar_angle).cpu()))

        pred_obj_rot_ref_to_last = torch.rad2deg(quaternion_to_axis_angle(pred_obj_quat_ref_to_last.q)).cpu().squeeze()
        pred_cam_rot_ref_to_last = torch.rad2deg(quaternion_to_axis_angle(pred_cam_quat_ref_to_last.q)).cpu().squeeze()

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

        for axis, axis_label in enumerate(['x', 'y', 'z']):
            rr.log(getattr(RerunAnnotations, f'obj_rot_1st_to_last_{axis_label}'),
                   rr.Scalar(obj_rot_1st_to_last[axis]))
            rr.log(getattr(RerunAnnotations, f'obj_rot_1st_to_last_{axis_label}_gt'),
                   rr.Scalar(obj_rot_1st_to_last_gt[axis]))
            rr.log(getattr(RerunAnnotations, f'obj_tran_1st_to_last_{axis_label}'),
                   rr.Scalar(obj_tran_1st_to_last[axis]))
            rr.log(getattr(RerunAnnotations, f'obj_tran_1st_to_last_{axis_label}_gt'),
                   rr.Scalar(obj_tran_1st_to_last_gt[axis]))

            rr.log(getattr(RerunAnnotations, f'obj_rot_ref_to_last_{axis_label}'),
                   rr.Scalar(pred_obj_rot_ref_to_last[axis]))
            rr.log(getattr(RerunAnnotations, f'cam_rot_ref_to_last_{axis_label}'),
                   rr.Scalar(pred_cam_rot_ref_to_last[axis]))

            rr.log(getattr(RerunAnnotations, f'chained_pose_long_flow_{axis_label}'),
                   rr.Scalar(long_jumps_pose_axis_angle[axis].item()))
            rr.log(getattr(RerunAnnotations, f'chained_pose_short_flow_{axis_label}'),
                   rr.Scalar(short_jumps_pose_axis_angle[axis].item()))

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

            frame_iou = 0.
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
        translation_tensors = [t[-1].detach().cpu() for t in logged_sgd_translations]
        rotation_tensors = [
            torch.rad2deg(quaternion_to_axis_angle(q.detach().cpu()))[-1]
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
        quaternions = detached_result.quaternions  # Assuming shape is (1, N, 4)
        for k in range(quaternions.shape[0]):
            quaternions[k] = normalize_quaternion(quaternions[k])
        # Convert quaternions to Euler angles
        angles_rad = quaternion_to_axis_angle(quaternions)
        # Convert radians to degrees
        angles_deg = torch.rad2deg(angles_rad)
        rot_axes = ['X-axis: ', 'Y-axis: ', 'Z-axis: ']
        for k in range(angles_rad.shape[0]):
            rotations = [rot_axes[i] + str(round(float(angles_deg[k, i]), 3))
                         for i in range(3)]
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} rotation: " + str(rotations) + '\n')
        for k in range(detached_result.quaternions.shape[0]):
            self.tracking_log.write(
                f"Keyframe {keyframes[k]} translation: str{detached_result.translations[k]}\n")
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

    def visualize_observed_data(self, keyframe_buffer: KeyframeBuffer, frame_i, view=Cameras.FRONTVIEW):

        view_name = view.value

        observed_image_annotation = RerunAnnotations.observed_image_frontview
        observed_image_segmentation_annotation = RerunAnnotations.observed_image_segmentation_frontview
        template_image_annotation = RerunAnnotations.template_image_frontview
        template_image_segmentation_annotation = RerunAnnotations.template_image_segmentation_frontview
        observed_flow_errors_annotations = RerunAnnotations.observed_flow_errors_frontview
        observed_flow_occlusion_annotation = RerunAnnotations.observed_flow_occlusion_frontview
        observed_flow_uncertainty_annotation = RerunAnnotations.observed_flow_uncertainty_frontview
        observed_flow_uncertainty_illustration_annotation = RerunAnnotations.observed_flow_with_uncertainty_frontview
        observed_flow_annotation = RerunAnnotations.observed_flow_frontview

        # Save the images to disk
        last_frame_observation = keyframe_buffer.get_observations_for_keyframe(frame_i)

        new_image_path = self.observations_path / Path(f'gt_img_{frame_i}.png')
        last_observed_image = last_frame_observation.observed_image.squeeze().cpu().permute(1, 2, 0)

        self.log_image(frame_i, last_observed_image, new_image_path, observed_image_annotation)

        if self.tracking_config.write_to_rerun_rather_than_disk:
            rr.set_time_sequence("frame", frame_i)
            image_segmentation = last_frame_observation.observed_segmentation
            rr.log(observed_image_segmentation_annotation, rr.SegmentationImage(image_segmentation))

        # Visualize new flow arcs
        if view == Cameras.FRONTVIEW:
            data_graph_camera = self.data_graph.get_camera_specific_frame_data(frame_i, view)
            new_flow_arcs = [(data_graph_camera.long_jump_source, frame_i)]

            for new_flow_arcs in sorted(new_flow_arcs):
                source_frame = new_flow_arcs[0]
                target_frame = new_flow_arcs[1]

                data_graph_edge_data = self.data_graph.get_edge_observations(source_frame, target_frame,
                                                                             Cameras.FRONTVIEW)
                flow_observation = data_graph_edge_data.observed_flow
                flow_observation_image_coords = flow_observation.cast_unit_coords_to_image_coords()

                synthetic_flow_obs = data_graph_edge_data.synthetic_flow_result.cast_unit_coords_to_image_coords()

                source_frame_data = self.data_graph.get_camera_specific_frame_data(source_frame, view)
                target_frame_data = self.data_graph.get_camera_specific_frame_data(target_frame, view)

                source_frame_observation = source_frame_data.frame_observation
                target_frame_observation = target_frame_data.frame_observation

                source_frame_image = source_frame_observation.observed_image
                source_frame_segment = source_frame_observation.observed_segmentation

                if (len(self.logged_flow_tracks_inits[view]) == 0 or
                        source_frame != self.logged_flow_tracks_inits[view][-1]):
                    template = source_frame_image.squeeze().permute(1, 2, 0)

                    template_image_path = self.observations_path / Path(f'template_img_{str(view)}_{frame_i}.png')
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

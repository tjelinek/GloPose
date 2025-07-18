from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from kornia.image import ImageSize

from data_structures.rerun_annotations import RerunAnnotations
from tracker_config import TrackerConfig
from utils.image_utils import overlay_mask
from utils.results_logging import log_correspondences_rerun


class PoseEstimatorLogger:

    def __init__(self):

        self.init_rerun()
        self.write_folder = Path('/mnt/personal/jelint19/results/PoseEstimation')

    def init_rerun(self):
        rr.init(f'{0}')
        rerun_file = self.write_folder / f'rerun_pose_estimation.rrd'
        rr.save(rerun_file)

        match_reliability_statistics = rrb.TimeSeriesView(name=f"Matching Reliability",
                                                          origin=RerunAnnotations.matching_reliability_plot,
                                                          axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                                zoom_lock=True),
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
                                            rrb.Spatial2DView(
                                                name=f"Matches High Certainty",
                                                origin=RerunAnnotations.matches_high_certainty),
                                            rrb.Spatial2DView(
                                                name=f"Matches Low Certainty",
                                                origin=RerunAnnotations.matches_low_certainty),
                                            [rrb.Spatial2DView(
                                                name=f"Matching Certainty",
                                                origin=RerunAnnotations.matching_certainty)],

                                        ],
                                        name='Matching'
                                    ),
                                    match_reliability_statistics,
                                ],
                                row_shares=[0.8, 0.2],
                                name='Matching'
                            ),
                        ],
                        name='Matching'
                    ),
                ],
                name=f'Results'
            )
        )

        rr.log(RerunAnnotations.matching_reliability_threshold_roma,
               rr.SeriesLine(color=[255, 0, 0], name="min reliability"), static=True)
        rr.log(RerunAnnotations.matching_reliability, rr.SeriesLine(color=[0, 0, 255], name="reliability"),
               static=True)
        rr.log(RerunAnnotations.matching_matchability_plot_share_matchable,
               rr.SeriesLine(color=[255, 0, 0], name="share of matchable fg"), static=True)
        rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty,
               rr.SeriesLine(color=[0, 0, 255], name=f"min match certainty"),
               static=True)

        rr.send_blueprint(blueprint)


def visualize_flow_with_matching_rerun(frame_i: int, src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor,
                                       certainty: torch.Tensor, config: TrackerConfig,
                                       viewgraph_image: torch.Tensor, query_image: torch.Tensor, reliability: float,
                                       reliability_threshold: float, certainty_threshold,
                                       match_certainty_map: torch.Tensor = None,):

    template_image = viewgraph_image
    target_image = query_image

    rr.log(RerunAnnotations.matching_reliability, rr.Scalar(reliability))
    rr.log(RerunAnnotations.matching_reliability_threshold_roma,
           rr.Scalar(reliability_threshold))

    if frame_i == 0 or (frame_i % config.large_images_results_write_frequency != 0):
        return

    template_target_image = torch.cat([template_image, target_image], dim=-2)
    template_target_image_np = template_target_image.permute(1, 2, 0).numpy(force=True)
    rerun_image = rr.Image(template_target_image_np)
    rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
    rr.log(RerunAnnotations.matches_low_certainty, rerun_image)

    certainties = certainty.numpy(force=True)
    threshold = certainty_threshold
    if threshold is None:
        threshold = config.min_roma_certainty_threshold

    above_threshold_mask = certainties >= threshold
    src_pts_xy_roma = src_pts_xy[:, [1, 0]].numpy(force=True)
    dst_pts_xy_roma = dst_pts_xy[:, [1, 0]].numpy(force=True)

    inliers_source_yx = src_pts_xy_roma[above_threshold_mask]
    inliers_target_yx = dst_pts_xy_roma[above_threshold_mask]
    outliers_source_yx = src_pts_xy_roma[~above_threshold_mask]
    outliers_target_yx = dst_pts_xy_roma[~above_threshold_mask]

    roma_certainty_map = match_certainty_map if match_certainty_map is not None else torch.ones_like(template_target_image)
    roma_h, roma_w = roma_certainty_map.shape[0], roma_certainty_map.shape[1] // 2
    certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(roma_certainty_map.device)
    certainty_map_column[:roma_h, :roma_w] = roma_certainty_map[:roma_h, :roma_w]
    certainty_map_column[roma_h:, :roma_w] = roma_certainty_map[:roma_h, roma_w:]
    certainty_map_column = certainty_map_column[None]
    roma_certainty_map_image_size = TF.resize(certainty_map_column, size=template_target_image.shape[1:])

    roma_certainty_map_im_size_np = roma_certainty_map_image_size.numpy(force=True)
    template_target_blacks = np.ones_like(template_target_image_np)
    template_target_image_certainty_np = overlay_mask(template_target_blacks, roma_certainty_map_im_size_np)

    rerun_certainty_img = rr.Image(template_target_image_certainty_np)
    rr.log(RerunAnnotations.matching_certainty, rerun_certainty_img)

    template_image_size = ImageSize(template_image.shape[-2:])
    cmap_inliers = plt.get_cmap('Greens')
    log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                              RerunAnnotations.matches_high_certainty, template_image_size.height, 20)
    cmap_outliers = plt.get_cmap('Reds')
    log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                              RerunAnnotations.matches_low_certainty, template_image_size.height, 20)

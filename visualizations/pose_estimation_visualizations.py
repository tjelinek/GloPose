from pathlib import Path
from typing import Final, Dict

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt
from kornia.image import ImageSize

from repositories.cnos.src.model.utils import Detections
from utils.image_utils import overlay_mask
from utils.results_logging import log_correspondences_rerun


class RerunAnnotationsPose:
    observed_image: Final[str] = '/observations/observed_image'
    observed_image_segmentation: Final[str] = '/observations/observed_image/segment'
    observed_image_all: Final[str] = '/observations/observed_image_all'
    observed_image_segmentation_all: Final[str] = '/observations/observed_image_all/segment'

    detection_image: Final[str] = '/observations/template_image'
    detection_nearest_neighbors: Final[str] = '/observations/template_image_neighbors'

    matches_high_certainty: Final[str] = '/matching/high_certainty'
    matches_low_certainty: Final[str] = '/matching/low_certainty'
    matching_certainty: Final[str] = '/matching/certainty'
    matches_high_certainty_segmentation: Final[str] = '/matching/high_certainty/segmentation'
    matches_low_certainty_segmentation: Final[str] = '/matching/low_certainty/segmentation'
    matching_reliability_plot: Final[str] = '/matching/reliability_plot'
    matching_reliability: Final[str] = '/matching/reliability_plot/reliability'
    matching_reliability_threshold_roma: Final[str] = '/matching/reliability_plot/reliability_threshold'

    matches_high_certainty_matchable: Final[str] = '/matching/high_certainty_matchable'
    matches_low_certainty_matchable: Final[str] = '/matching/low_certainty_matchable'
    matching_matchability_plot: Final[str] = '/matching/matchability_plot'
    matching_matchability_plot_share_matchable: Final[str] = '/matching/matchability_plot/share_matchable'
    matching_min_roma_certainty_plot: Final[str] = '/matching/min_roma_certainty_plot/'
    matching_min_roma_certainty_plot_min_certainty: Final[str] = '/matching/min_roma_certainty_plot/min_certainty'

    matches_sift: Final[str] = '/matching/reliability_plot/sift_num_matches/'
    min_matches_sift: Final[str] = '/matching/reliability_plot/min_matches_sift'
    good_to_add_number_of_matches_sift: Final[str] = '/matching/reliability_plot/good_to_add_matches_sift'


def tensor2numpy(image, downsample_factor=1.0):
    _, h, w = image.shape
    image = TF.resize(image.float().unsqueeze(0),
                      [int(h * downsample_factor), int(w * downsample_factor)],
                      interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
    return image.permute(1, 2, 0).numpy(force=True) * 255.


class PoseEstimatorLogger:

    def __init__(self, output_path: Path, image_downsample: float = 0.5):
        self.init_rerun(output_path)
        self.rerun_sequence_id: int = 0
        self.image_downsample = image_downsample

    @staticmethod
    def init_rerun(output_path: Path):
        rr.init(str(output_path))
        rr.save(output_path)

        match_reliability_statistics = rrb.TimeSeriesView(name=f"Matching Reliability",
                                                          origin=RerunAnnotationsPose.matching_reliability_plot,
                                                          axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                                zoom_lock=True),
                                                          plot_legend=rrb.PlotLegend(visible=True))
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                contents=[
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name=f"Scene",
                                origin=RerunAnnotationsPose.observed_image),
                            rrb.Spatial2DView(
                                name=f"Scene - all detections",
                                origin=RerunAnnotationsPose.observed_image_all),
                        ],
                        name='Detections'
                    ),
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name=f"Template",
                                origin=RerunAnnotationsPose.detection_image),
                            rrb.Grid(
                                name=f"Scene - all detections",
                                grid_columns=2,
                                contents=[
                                    rrb.Spatial2DView(
                                        name=f"Nearest best object template {i+1}",
                                        origin=f'{RerunAnnotationsPose.detection_nearest_neighbors}/{i}',
                                        # contents=[
                                        #     rrb.TextDocumentView(
                                        #         name="score",
                                        #         origin=f"{RerunAnnotationsPose.detection_nearest_neighbors}/{i}/score"
                                        #     )
                                        # ]
                                    )
                                    for i in range(6)
                                ]
                            ),
                        ],
                        name='Detections - Closest Neighbors'
                    ),
                    rrb.Vertical(
                        contents=[
                            rrb.Horizontal(
                                contents=[
                                    rrb.Spatial2DView(
                                        name=f"Matches High Certainty",
                                        origin=RerunAnnotationsPose.matches_high_certainty),
                                    rrb.Spatial2DView(
                                        name=f"Matches Low Certainty",
                                        origin=RerunAnnotationsPose.matches_low_certainty),
                                    rrb.Spatial2DView(
                                        name=f"Matching Certainty",
                                        origin=RerunAnnotationsPose.matching_certainty),

                                ],
                                name='Matching'
                            ),
                            match_reliability_statistics,
                        ],
                        row_shares=[0.8, 0.2],
                        name='Matching'
                    ),
                ],
                name=f'Results'
            )
        )

        rr.log(RerunAnnotationsPose.matching_reliability_threshold_roma,
               rr.SeriesLine(color=[255, 0, 0], name="min reliability"), static=True)
        rr.log(RerunAnnotationsPose.matching_reliability, rr.SeriesLine(color=[0, 0, 255], name="reliability"),
               static=True)
        rr.log(RerunAnnotationsPose.matching_matchability_plot_share_matchable,
               rr.SeriesLine(color=[255, 0, 0], name="share of matchable fg"), static=True)
        rr.log(RerunAnnotationsPose.matching_min_roma_certainty_plot_min_certainty,
               rr.SeriesLine(color=[0, 0, 255], name=f"min match certainty"),
               static=True)

        rr.log(RerunAnnotationsPose.matches_high_certainty_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotationsPose.matches_low_certainty_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotationsPose.observed_image_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotationsPose.observed_image_segmentation_all,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)


        rr.send_blueprint(blueprint)

    def visualize_detections(self, all_detections_segmentations, detection_idx):
        h, w = all_detections_segmentations.shape[-2:]

        seg_all = TF.resize(all_detections_segmentations.float().unsqueeze(0),
                        [int(h * self.image_downsample), int(w * self.image_downsample)],
                        interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        query_segment_np = seg_all.squeeze(0).to(torch.float).numpy(force=True)
        rr.set_time_sequence('frame', self.rerun_sequence_id)

        rr_segment = rr.SegmentationImage(query_segment_np[detection_idx])

        segment_cumulative = query_segment_np[:detection_idx].sum(axis=0)
        rr_segment_cumulative = rr.SegmentationImage(segment_cumulative)

        rr.log(RerunAnnotationsPose.observed_image_segmentation, rr_segment)
        rr.log(f'{RerunAnnotationsPose.observed_image_segmentation_all}', rr_segment_cumulative)

    def visualize_nearest_neighbors(self, query_image: torch.Tensor, view_graph_images: Dict[int, torch.Tensor],
                                    detection_idx: int, detections: Detections, detections_scores):

        rr.set_time_sequence('frame', self.rerun_sequence_id)

        detection_bbox = detections.boxes[detection_idx]
        viewgraph_id = detections.object_ids[detection_idx].item()

        detections_scores_per_viewgraph = detections_scores[viewgraph_id]
        k = min(6, detections_scores_per_viewgraph.shape[1])
        topk_scores, topk_template_indices = torch.topk(detections_scores_per_viewgraph, k=k, dim=-1)

        detection_topk_scores = topk_scores[detection_idx]
        detection_topk_template_ids = topk_template_indices[detection_idx]

        x1, y1, x2, y2 = detection_bbox.int()
        cropped_detection = query_image[..., y1:y2, x1:x2]

        rr_detection = rr.Image(tensor2numpy(cropped_detection))
        rr.log(RerunAnnotationsPose.detection_image, rr_detection)

        template_images = view_graph_images[viewgraph_id]
        for i in range(6):
            if i < detection_topk_template_ids.shape[0]:
                template_image_idx = detection_topk_template_ids[i].item()
                template_image = template_images[template_image_idx]
                template_score = detection_topk_scores[i].item()
            else:
                template_image = torch.zeros_like(cropped_detection)
                template_score = 0.

            rr_template = rr.Image(tensor2numpy(template_image, self.image_downsample))
            rr.log(f'{RerunAnnotationsPose.detection_nearest_neighbors}/{i}', rr_template)
            # rr.log(f'{RerunAnnotationsPose.detection_nearest_neighbors}/{i}/scores',
            #        rr.TextDocument(f"score: {template_score:.3f}"))

    def visualize_image(self, query_image: torch.Tensor):
        h, w = query_image.shape[-2:]
        img = TF.resize(query_image, [int(h * self.image_downsample), int(w * self.image_downsample)])
        query_image_np = tensor2numpy(img)
        rr.set_time_sequence('frame', self.rerun_sequence_id)
        rr_image = rr.Image(query_image_np)
        rr.log(RerunAnnotationsPose.observed_image, rr_image)
        rr.log(RerunAnnotationsPose.observed_image_all, rr_image)

    def visualize_pose_matching_rerun(self, src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor, certainty: torch.Tensor,
                                      viewgraph_image: torch.Tensor, query_image: torch.Tensor, reliability: float,
                                      reliability_threshold: float, certainty_threshold,
                                      match_certainty_map: torch.Tensor = None,
                                      viewgraph_image_segment: torch.Tensor = None,
                                      query_image_segment: torch.Tensor = None):
        template_image = viewgraph_image
        target_image = query_image

        rr.set_time_sequence('frame', self.rerun_sequence_id)

        rr.log(RerunAnnotationsPose.matching_reliability, rr.Scalar(reliability))
        rr.log(RerunAnnotationsPose.matching_reliability_threshold_roma,
               rr.Scalar(reliability_threshold))

        template_target_image = torch.cat([template_image, target_image], dim=-2)
        template_target_image_segment = torch.cat([viewgraph_image_segment, query_image_segment], dim=-2)

        th, tw = template_target_image.shape[-2:]
        template_target_image = TF.resize(template_target_image,
                                          [int(th * self.image_downsample), int(tw * self.image_downsample)])
        template_target_image_segment = TF.resize(template_target_image_segment.float().unsqueeze(0),
                                                  [int(th * self.image_downsample), int(tw * self.image_downsample)],
                                                  interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        template_target_image_np = tensor2numpy(template_target_image)
        template_target_image_segment_np = template_target_image_segment.squeeze().numpy(force=True)

        rerun_image = rr.Image(template_target_image_np)
        rerun_segment = rr.SegmentationImage(template_target_image_segment_np)
        rr.log(RerunAnnotationsPose.matches_high_certainty, rerun_image)
        rr.log(RerunAnnotationsPose.matches_low_certainty, rerun_image)
        # rr.log(RerunAnnotationsPose.matches_high_certainty_segmentation, rerun_segment)
        # rr.log(RerunAnnotationsPose.matches_low_certainty_segmentation, rerun_segment)

        certainties = certainty.numpy(force=True)
        threshold = certainty_threshold

        above_threshold_mask = certainties >= threshold
        src_pts_xy_roma = src_pts_xy[:, [1, 0]].numpy(force=True) * self.image_downsample
        dst_pts_xy_roma = dst_pts_xy[:, [1, 0]].numpy(force=True) * self.image_downsample

        inliers_source_yx = src_pts_xy_roma[above_threshold_mask]
        inliers_target_yx = dst_pts_xy_roma[above_threshold_mask]
        outliers_source_yx = src_pts_xy_roma[~above_threshold_mask]
        outliers_target_yx = dst_pts_xy_roma[~above_threshold_mask]

        roma_certainty_map = match_certainty_map if match_certainty_map is not None else torch.ones_like(
            template_target_image[0, ...])
        roma_h, roma_w = roma_certainty_map.shape[-2], roma_certainty_map.shape[-1] // 2
        certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(roma_certainty_map.device)
        certainty_map_column[:roma_h, :roma_w] = roma_certainty_map[:roma_h, :roma_w]
        certainty_map_column[roma_h:, :roma_w] = roma_certainty_map[:roma_h, roma_w:]
        certainty_map_column = certainty_map_column[None]
        roma_certainty_map_image_size = TF.resize(certainty_map_column, size=list(template_target_image.shape[1:]))

        roma_certainty_map_im_size_np = roma_certainty_map_image_size.numpy(force=True)
        template_target_blacks = np.ones_like(template_target_image_np)
        template_target_image_certainty_np = overlay_mask(template_target_blacks, roma_certainty_map_im_size_np)

        rerun_certainty_img = rr.Image(template_target_image_certainty_np)
        rr.log(RerunAnnotationsPose.matching_certainty, rerun_certainty_img)

        template_image_size = ImageSize(*template_image.shape[-2:])
        cmap_inliers = plt.get_cmap('Greens')
        log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                                  RerunAnnotationsPose.matches_high_certainty,
                                  int(template_image_size.height * self.image_downsample), 2000)
        cmap_outliers = plt.get_cmap('Reds')
        log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                                  RerunAnnotationsPose.matches_low_certainty,
                                  int(template_image_size.height * self.image_downsample), 2000)

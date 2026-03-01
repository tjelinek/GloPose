from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from matplotlib import cm
from torchvision import transforms

from data_structures.rerun_annotations import RerunAnnotations
from visualizations.rerun_utils import (init_rerun_recording, register_matching_series_lines,
                                        visualize_certainty_map, log_matching_correspondences)


def tensor2numpy(image, downsample_factor=1.0):
    h, w = image.shape[-2:]
    image = TF.resize(image.float().unsqueeze(0),
                      [int(h * downsample_factor), int(w * downsample_factor)],
                      interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
    return (image.permute(1, 2, 0).numpy(force=True) * 255.).astype(np.uint8)


def add_score_overlay(image, score, similarity_metric, bounds=None):
    image = image.astype(np.uint8)
    h, w = image.shape[:2]
    box_height = int(h * 0.1)
    overlay = image.copy()

    if similarity_metric.lower() == "csls":
        min_v, max_v = (-4.0, 2.0) if bounds is None else bounds
    else:
        min_v, max_v = (-1.0, 1.0) if bounds is None else bounds

    if score == 0:
        color = (0, 0, 0)
    else:
        s = np.clip(score, min_v, max_v)
        t = (s - min_v) / (max_v - min_v)
        normalized_score = 0.5 + 0.5 * t
        inferno_cmap = cm.get_cmap('inferno')
        r, g, b, _ = inferno_cmap(normalized_score)
        color = (int(b * 255), int(g * 255), int(r * 255))

    cv2.rectangle(overlay, (0, 0), (w, box_height), color, -1)

    text = f"{similarity_metric} similarity {score:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Calculate text size and position to center it in the box
    text_size = cv2.getTextSize(text, font, 1, 2)[0]

    # Scale font size to fit the box if needed
    font_scale = min(1.0, (box_height * 0.8) / text_size[1])
    font_scale = min(font_scale, (w * 0.9) / text_size[0])
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (box_height + text_size[1]) // 2
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

    return overlay


class PoseEstimatorLogger:

    def __init__(self, output_path: Path, image_downsample: float = 0.5, rerun_jpeg_quality: int = 75):
        self.init_rerun(output_path)
        self.rerun_sequence_id: int = 0
        self.image_downsample = image_downsample
        self.rerun_jpeg_quality = rerun_jpeg_quality

    @staticmethod
    def init_rerun(output_path: Path):
        match_reliability_statistics = rrb.TimeSeriesView(name="Matching Reliability",
                                                          origin=RerunAnnotations.matching_reliability_plot,
                                                          axis_y=rrb.ScalarAxis(range=(0.0, 1.2),
                                                                                zoom_lock=True),
                                                          plot_legend=rrb.PlotLegend(visible=True))
        blueprint = rrb.Blueprint(
            rrb.Tabs(
                contents=[
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name="Scene",
                                origin=RerunAnnotations.observed_image),
                            rrb.Spatial2DView(
                                name="Scene - all detections",
                                origin=RerunAnnotations.observed_image_all),
                        ],
                        name='Detections'
                    ),
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name="Template",
                                origin=RerunAnnotations.detection_image),
                            rrb.Grid(
                                name="Scene - all detections",
                                grid_columns=2,
                                contents=[
                                    rrb.Spatial2DView(
                                        name=f"Nearest best object template {i + 1}",
                                        origin=f'{RerunAnnotations.detection_nearest_neighbors}/{i}',
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
                                        name="Matches High Certainty",
                                        origin=RerunAnnotations.matches_high_certainty),
                                    rrb.Spatial2DView(
                                        name="Matches Low Certainty",
                                        origin=RerunAnnotations.matches_low_certainty),
                                    rrb.Spatial2DView(
                                        name="Matching Certainty",
                                        origin=RerunAnnotations.matching_certainty),
                                ],
                                name='Matching'
                            ),
                            match_reliability_statistics,
                        ],
                        row_shares=[0.8, 0.2],
                        name='Matching'
                    ),
                ],
                name='Results'
            )
        )

        init_rerun_recording(str(output_path), output_path, blueprint)
        register_matching_series_lines()

        rr.log(RerunAnnotations.matches_high_certainty_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotations.matches_low_certainty_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotations.observed_image_segmentation,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)
        rr.log(RerunAnnotations.observed_image_segmentation_all,
               rr.AnnotationContext([(1, "white", (255, 255, 255)), (0, "black", (0, 0, 0))]), static=True)

    def visualize_detections(self, all_detections_segmentations, detection_idx):
        h, w = all_detections_segmentations.shape[-2:]

        seg_all = TF.resize(all_detections_segmentations.float().unsqueeze(0),
                            [int(h * self.image_downsample), int(w * self.image_downsample)],
                            interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        query_segment_np = seg_all.to(torch.float).numpy(force=True)
        rr.set_time_sequence('frame', self.rerun_sequence_id)

        rr_segment = rr.SegmentationImage(query_segment_np[detection_idx])

        segment_cumulative = query_segment_np[:detection_idx + 1].sum(axis=0)
        rr_segment_cumulative = rr.SegmentationImage(segment_cumulative)

        rr.log(RerunAnnotations.observed_image_segmentation, rr_segment)
        rr.log(f'{RerunAnnotations.observed_image_segmentation_all}', rr_segment_cumulative)

    def visualize_nearest_neighbors(self,
                                    query_image: torch.Tensor,
                                    template_images: Dict[int, List[torch.Tensor] | Path],
                                    template_masks: Dict[int, List[torch.Tensor] | Path],
                                    detection_idx: int,
                                    detections,
                                    detections_scores,
                                    similarity_metric: str):

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

        rr_detection = rr.Image(tensor2numpy(cropped_detection)).compress(jpeg_quality=self.rerun_jpeg_quality)
        rr.log(RerunAnnotations.detection_image, rr_detection)

        template_images = template_images[viewgraph_id]
        template_masks = template_masks[viewgraph_id]

        for i in range(6):
            if i < detection_topk_template_ids.shape[0]:
                template_image_idx = detection_topk_template_ids[i].item()

                if isinstance(template_images[template_image_idx], (str, Path)):
                    image_path = template_images[template_image_idx]
                    rgb_img = Image.open(image_path).convert('RGB')
                    template_image_tensor = transforms.ToTensor()(rgb_img)
                else:
                    template_image_tensor = template_images[template_image_idx]

                if isinstance(template_masks[template_image_idx], (str, Path)):
                    mask_path = template_masks[template_image_idx]
                    mask_img = Image.open(mask_path).convert('L')
                    template_mask_tensor = transforms.ToTensor()(mask_img)
                else:
                    template_mask_tensor = template_masks[template_image_idx]

                template_image = tensor2numpy(template_image_tensor, self.image_downsample)
                template_mask = tensor2numpy(template_mask_tensor[None], self.image_downsample)
                template_score = detection_topk_scores[i].item()
                template_image_with_overlay = add_score_overlay(template_image, template_score, similarity_metric)
            else:
                template_image = tensor2numpy(torch.zeros_like(cropped_detection), self.image_downsample)
                template_mask = tensor2numpy(torch.zeros_like(cropped_detection)[0:1], self.image_downsample)
                template_image_with_overlay = template_image

            rr_template = rr.Image(template_image_with_overlay).compress(jpeg_quality=self.rerun_jpeg_quality)
            rr.log(f'{RerunAnnotations.detection_nearest_neighbors}/{i}', rr_template)

            rr_mask = rr.SegmentationImage(template_mask, opacity=0.5)
            context = rr.AnnotationContext([(0, "", (0, 0, 0, 0))])  # 0 is transparent
            rr.log(f'{RerunAnnotations.detection_nearest_neighbors}/{i}/mask', rr_mask, context)

    def visualize_image(self, query_image: torch.Tensor):
        h, w = query_image.shape[-2:]
        img = TF.resize(query_image, [int(h * self.image_downsample), int(w * self.image_downsample)])
        query_image_np = tensor2numpy(img)
        rr.set_time_sequence('frame', self.rerun_sequence_id)
        rr_image = rr.Image(query_image_np).compress(jpeg_quality=self.rerun_jpeg_quality)
        rr.log(RerunAnnotations.observed_image, rr_image)
        rr.log(RerunAnnotations.observed_image_all, rr_image)

    def visualize_pose_matching_rerun(self, src_pts_xy: torch.Tensor, dst_pts_xy: torch.Tensor, certainty: torch.Tensor,
                                      viewgraph_image: torch.Tensor, query_image: torch.Tensor, reliability: float,
                                      reliability_threshold: float, certainty_threshold,
                                      match_certainty_map: torch.Tensor = None,
                                      viewgraph_image_segment: torch.Tensor = None,
                                      query_image_segment: torch.Tensor = None):
        template_image = viewgraph_image
        target_image = query_image

        rr.set_time_sequence('frame', self.rerun_sequence_id)

        rr.log(RerunAnnotations.matching_reliability, rr.Scalar(reliability))
        rr.log(RerunAnnotations.matching_reliability_threshold_roma,
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

        rerun_image = rr.Image(template_target_image_np).compress(jpeg_quality=self.rerun_jpeg_quality)
        rerun_segment = rr.SegmentationImage(template_target_image_segment_np)
        rr.log(RerunAnnotations.matches_high_certainty, rerun_image)
        rr.log(RerunAnnotations.matches_low_certainty, rerun_image)
        # rr.log(RerunAnnotations.matches_high_certainty_segmentation, rerun_segment)
        # rr.log(RerunAnnotations.matches_low_certainty_segmentation, rerun_segment)

        src_pts_yx = src_pts_xy[:, [1, 0]].numpy(force=True) * self.image_downsample
        dst_pts_yx = dst_pts_xy[:, [1, 0]].numpy(force=True) * self.image_downsample

        roma_certainty_map = match_certainty_map if match_certainty_map is not None else torch.ones_like(
            template_target_image[0, ...])
        visualize_certainty_map(roma_certainty_map, template_target_image.shape,
                                template_target_image_np, RerunAnnotations.matching_certainty,
                                self.rerun_jpeg_quality)

        template_image_height = int(template_image.shape[-2] * self.image_downsample)
        log_matching_correspondences(src_pts_yx, dst_pts_yx, certainty.numpy(force=True),
                                     certainty_threshold, template_image_height,
                                     RerunAnnotations.matches_high_certainty,
                                     RerunAnnotations.matches_low_certainty, 2000)

from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
import torchvision.transforms.functional as TF
from matplotlib import pyplot as plt

from data_structures.rerun_annotations import RerunAnnotations
from utils.image_utils import overlay_mask


def init_rerun_recording(name: str, output_path: Path, blueprint: rrb.Blueprint):
    """Initialize a rerun recording: init, save to .rrd file, and send blueprint."""
    rr.init(name)
    rr.save(output_path)
    rr.send_blueprint(blueprint)


def register_matching_series_lines():
    """Register shared series line styles for matching reliability/matchability plots."""
    rr.log(RerunAnnotations.matching_reliability_threshold_roma,
           rr.SeriesLine(color=[255, 0, 0], name="min reliability"), static=True)
    rr.log(RerunAnnotations.matching_reliability,
           rr.SeriesLine(color=[0, 0, 255], name="reliability"), static=True)
    rr.log(RerunAnnotations.matching_matchability_plot_share_matchable,
           rr.SeriesLine(color=[255, 0, 0], name="share of matchable fg"), static=True)
    rr.log(RerunAnnotations.matching_min_roma_certainty_plot_min_certainty,
           rr.SeriesLine(color=[0, 0, 255], name="min match certainty"), static=True)


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


def visualize_certainty_map(certainty_map: torch.Tensor,
                            target_image_shape: tuple,
                            template_target_image_np: np.ndarray,
                            certainty_annotation: str,
                            jpeg_quality: int = 75):
    """Reshape a (H, W*2) certainty map into column layout, resize to match the concatenated
    template+target image, overlay as heatmap, and log to rerun.

    Args:
        certainty_map: (H, W*2) or (1, H, W*2) certainty map from matcher
        target_image_shape: (C, H, W) shape of the concatenated template+target image tensor
        template_target_image_np: (H*2, W, 3) uint8 numpy image (or ones for black background)
        certainty_annotation: rerun annotation path for the certainty view
        jpeg_quality: JPEG compression quality for rerun
    """
    roma_h, roma_w = certainty_map.shape[-2], certainty_map.shape[-1] // 2
    certainty_map_column = torch.zeros(roma_h * 2, roma_w).to(certainty_map.device)
    certainty_map_column[:roma_h, :roma_w] = certainty_map[..., :roma_h, :roma_w]
    certainty_map_column[roma_h:, :roma_w] = certainty_map[..., :roma_h, roma_w:]
    certainty_map_column = certainty_map_column[None]
    certainty_map_resized = TF.resize(certainty_map_column, size=list(target_image_shape[1:]))

    certainty_map_np = certainty_map_resized.numpy(force=True)
    background = np.ones_like(template_target_image_np)
    certainty_overlay = overlay_mask(background, certainty_map_np)

    rr.log(certainty_annotation,
           rr.Image(certainty_overlay).compress(jpeg_quality=jpeg_quality))


def log_matching_correspondences(src_pts_yx: np.ndarray, dst_pts_yx: np.ndarray,
                                 certainties: np.ndarray, threshold: float,
                                 source_image_height: int,
                                 high_certainty_annotation: str,
                                 low_certainty_annotation: str,
                                 sample_size: int = 2000):
    """Split points by certainty threshold and log inlier/outlier correspondences to rerun.

    Args:
        src_pts_yx: (N, 2) source points in (y, x) order
        dst_pts_yx: (N, 2) destination points in (y, x) order
        certainties: (N,) certainty values
        threshold: certainty threshold for inlier/outlier split
        source_image_height: height of the source (template) image for vertical offset
        high_certainty_annotation: rerun annotation path for inliers
        low_certainty_annotation: rerun annotation path for outliers
        sample_size: max number of correspondences to draw per group
    """
    above_threshold_mask = certainties >= threshold

    inliers_source_yx = src_pts_yx[above_threshold_mask]
    inliers_target_yx = dst_pts_yx[above_threshold_mask]
    outliers_source_yx = src_pts_yx[~above_threshold_mask]
    outliers_target_yx = dst_pts_yx[~above_threshold_mask]

    cmap_inliers = plt.get_cmap('Greens')
    log_correspondences_rerun(cmap_inliers, inliers_source_yx, inliers_target_yx,
                              high_certainty_annotation, source_image_height, sample_size)
    cmap_outliers = plt.get_cmap('Reds')
    log_correspondences_rerun(cmap_outliers, outliers_source_yx, outliers_target_yx,
                              low_certainty_annotation, source_image_height, sample_size)

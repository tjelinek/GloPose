import numpy as np
import rerun as rr
import torch


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

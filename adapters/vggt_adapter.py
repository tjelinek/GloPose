"""Adapter for the VGGT external repository.

This is the SOLE location in GloPose that imports VGGT internals.
All other modules import from here.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import pycolmap

from onboarding.colmap_utils import add_posed_image_to_reconstruction, make_point2d_list

VGGT_REPO = Path(__file__).resolve().parent.parent / 'repositories' / 'vggt'


def _ensure_vggt_on_path():
    vggt_str = str(VGGT_REPO)
    if vggt_str not in sys.path:
        sys.path.insert(0, vggt_str)


def load_vggt_model(device: str = 'cuda'):
    """Load the VGGT-1B model."""
    _ensure_vggt_on_path()
    from vggt.models.vggt import VGGT

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    return model


def reconstruct_with_vggt(
    image_paths: list[Path],
    image_names: list[str],
    device: str = 'cuda',
    camera_K: Optional[torch.Tensor] = None,
    conf_threshold: float = 0.0,
    max_points: int = 100_000,
    segmentation_paths: Optional[list[Path]] = None,
    model=None,
) -> Optional[pycolmap.Reconstruction]:
    """Run VGGT feed-forward reconstruction on a set of images.

    Args:
        image_paths: Paths to input images (already background-masked if desired).
        image_names: COLMAP image names (e.g. '0.png', '5.png') — must match
            the names used in DataGraph.image_filename.
        device: Torch device.
        camera_K: Optional known camera intrinsics (3x3 tensor). If provided,
            VGGT-predicted intrinsics are replaced with these.
        conf_threshold: Depth confidence threshold for point filtering.
        max_points: Maximum number of 3D points to include.
        model: Pre-loaded VGGT model. If None, loads from HuggingFace.

    Returns:
        pycolmap.Reconstruction with poses and 3D points, or None on failure.
    """
    _ensure_vggt_on_path()
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues

    if len(image_paths) < 2:
        print("VGGT requires at least 2 images")
        return None

    if model is None:
        model = load_vggt_model(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    vggt_resolution = 518
    img_load_resolution = 1024

    # Load and preprocess images
    images, original_coords = load_and_preprocess_images_square(
        [str(p) for p in image_paths], img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)

    # Run VGGT
    images_resized = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images_batch = images_resized[None]  # add batch dim
            aggregated_tokens_list, ps_idx = model.aggregator(images_batch)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, images_resized.shape[-2:])
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens_list, images_batch, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()

    # Unproject depth to 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Build reconstruction in feed-forward mode (no BA)
    num_frames = len(image_paths)
    image_size = np.array([vggt_resolution, vggt_resolution])
    height, width = vggt_resolution, vggt_resolution

    # Get RGB colors at VGGT resolution
    points_rgb = F.interpolate(
        images, size=(vggt_resolution, vggt_resolution),
        mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)  # (N, H, W, 3)

    # Pixel coordinates + frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Filter by confidence
    conf_mask = depth_conf >= conf_threshold

    # Filter by segmentation masks (keep only object points)
    if segmentation_paths is not None:
        from PIL import Image
        for fidx, seg_path in enumerate(segmentation_paths):
            seg = np.array(Image.open(seg_path).convert('L').resize(
                (vggt_resolution, vggt_resolution), Image.NEAREST))
            conf_mask[fidx] &= (seg > 127)

    conf_mask = randomly_limit_trues(conf_mask, max_points)

    filtered_pts3d = points_3d[conf_mask]
    filtered_xyf = points_xyf[conf_mask]
    filtered_rgb = points_rgb[conf_mask]

    # Build pycolmap.Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add 3D points
    for idx in range(len(filtered_pts3d)):
        reconstruction.add_point3D(
            filtered_pts3d[idx], pycolmap.Track(), filtered_rgb[idx])

    # Rescale intrinsics and 2D points from VGGT's 518px padded-square space
    # to the original image pixel space.
    #
    # original_coords: [x1, y1, x2, y2, width, height] per frame
    #   (x1,y1)-(x2,y2) = bounding box of original image content in the 1024px padded square
    #   width, height = original image dimensions
    #
    # Mapping: grid_518 → padded_1024 → original_image
    #   padded = grid * (1024 / 518)
    #   original_x = (padded_x - x1) * (width / (x2 - x1))
    #   original_y = (padded_y - y1) * (height / (y2 - y1))
    original_coords_np = original_coords.cpu().numpy()
    grid_to_padded = img_load_resolution / vggt_resolution  # 1024 / 518

    for fidx in range(num_frames):
        x1, y1, x2, y2, orig_w, orig_h = original_coords_np[fidx]
        content_w = x2 - x1  # width of original image in 1024px space
        content_h = y2 - y1  # height of original image in 1024px space

        # Scale and offset from 518px grid to original image
        scale_x = grid_to_padded * orig_w / content_w
        scale_y = grid_to_padded * orig_h / content_h
        offset_x = x1 * orig_w / content_w
        offset_y = y1 * orig_h / content_h

        cam_w, cam_h = int(orig_w), int(orig_h)

        # Override with known intrinsics if provided
        if camera_K is not None:
            K_np = camera_K.cpu().numpy() if isinstance(camera_K, torch.Tensor) else camera_K
            fx, fy = K_np[0, 0], K_np[1, 1]
            cx, cy = K_np[0, 2], K_np[1, 2]
            cam_params = np.array([fx, fy, cx, cy])
        else:
            # Transform VGGT intrinsics (518px padded-square) to original image space
            K = intrinsic[fidx].copy()
            fx = K[0, 0] * scale_x
            fy = K[1, 1] * scale_y
            cx = K[0, 2] * scale_x - offset_x
            cy = K[1, 2] * scale_y - offset_y
            cam_params = np.array([fx, fy, cx, cy])

        camera = pycolmap.Camera(
            model='PINHOLE', width=cam_w, height=cam_h,
            params=cam_params, camera_id=fidx + 1)
        reconstruction.add_camera(camera)

        # Extrinsics are already camera-from-world (OpenCV convention)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsic[fidx][:3, :3]),
            extrinsic[fidx][:3, 3])

        # Add 2D point observations for points belonging to this frame
        # Transform grid coordinates to original image space
        points2D_list = []
        point2D_idx = 0
        points_in_frame = (filtered_xyf[:, 2].astype(np.int32) == fidx)
        for batch_idx in np.nonzero(points_in_frame)[0]:
            point3D_id = int(batch_idx) + 1
            gx, gy = filtered_xyf[batch_idx, :2]
            px = gx * scale_x - offset_x
            py = gy * scale_y - offset_y
            points2D_list.append(pycolmap.Point2D(np.array([px, py]), point3D_id))
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        add_posed_image_to_reconstruction(
            reconstruction, fidx + 1, fidx + 1, image_names[fidx],
            cam_from_world, points2D=make_point2d_list(points2D_list))

    print(f"VGGT reconstruction: {num_frames} images, "
          f"{len(filtered_pts3d)} 3D points")
    return reconstruction

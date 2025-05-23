from pathlib import Path

import torch

from data_providers.frame_provider import PrecomputedDepthProvider, PrecomputedFrameProvider
from tracker_config import TrackerConfig
from utils.bop_challenge import read_gt_Se3_world2cam, get_pinhole_params


def depth_to_xyz(depth, intrinsics):
    """
    depth: [H,W] tensor (meters)
    intrinsics: [3,3] camera matrix
    returns: [3, N] world points in camera frame
    """
    H, W = depth.shape
    u = torch.arange(0, W, device=depth.device, dtype=intrinsics.dtype)
    v = torch.arange(0, H, device=depth.device, dtype=intrinsics.dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv1 = torch.stack((u, v, torch.ones_like(u)), dim=-1)  # [H,W,3]
    invK = torch.inverse(intrinsics)
    rays = (invK @ uv1.reshape(-1, 3).T)  # [3,H*W]
    pts = rays * depth.reshape(1, -1)
    return pts  # in camera frame


def transform_pts(pts, cam2world):
    """
    pts: [3, N] in cam frame
    cam2world: [4,4]
    returns: [3, N] in world frame
    """
    homo = torch.cat((pts, torch.ones(1, pts.shape[1], device=pts.device)), dim=0)
    wpts = cam2world @ homo
    return wpts[:3]


def overlap_ratio(depth_a, depth_b, cam2world_a, cam2world_b, intrinsics_a, thresh=0.05):
    # 1) get A’s points in world, then into B’s camera frame
    pts_a = depth_to_xyz(depth_a, intrinsics_a)
    wpts_a = transform_pts(pts_a, cam2world_a)
    world2b = torch.inverse(cam2world_b)
    bpts = world2b @ torch.cat((wpts_a, torch.ones(1, wpts_a.shape[1], device=wpts_a.device)), dim=0)
    cam_pts = bpts[:3]
    # 2) project into B’s image
    proj = intrinsics_a @ cam_pts
    u = proj[0] / proj[2]
    v = proj[1] / proj[2]
    valid = (proj[2] > 0)
    u_int = u.round().long()
    v_int = v.round().long()
    H, W = depth_b.shape
    in_bounds = valid & (u_int >= 0) & (u_int < W) & (v_int >= 0) & (v_int < H)
    u_int = u_int[in_bounds]
    v_int = v_int[in_bounds]
    zs = cam_pts[2, in_bounds]
    # 3) compare against depth_b
    db = depth_b[v_int, u_int]
    matched = torch.abs(db - zs) < thresh
    return matched.sum().float() / in_bounds.sum().float()


def compute_all_overlaps(depths, cam2worlds, intrinsics):
    """
    depths: list of [H,W] tensors
    cam2worlds: list of [4,4] tensors
    intrinsics: [3,3] tensor
    returns: NxN matrix of overlap ratios
    """
    N = len(depths)
    M = torch.zeros((N, N), device=depths[0].device)
    for i in range(N):
        for j in range(N):
            if i != j:
                M[i, j] = overlap_ratio(depths[i], depths[j], cam2worlds[i], cam2worlds[j], intrinsics[i])
    return M


def compute_overlaps_bop(dataset_name):
    config = TrackerConfig()

    path_to_dataset = Path(f'/mnt/personal/jelint19/data/bop/{dataset_name}')
    training_set = path_to_dataset / 'train_pbr'

    for scene in training_set.iterdir():
        depths_folder = scene / 'depth'
        images_folder = scene / 'rgb'
        scene_camera_path = scene / 'scene_camera.json'

        Se3_world2cams = read_gt_Se3_world2cam(scene_camera_path, device=config.device)
        camera_K = get_pinhole_params(scene_camera_path, scale=config.image_downsample, device=config.device)

        images_paths = sorted(images_folder.iterdir())
        depths_paths = sorted(depths_folder.iterdir())
        N_frames = len(images_paths)

        image_provider = PrecomputedFrameProvider(config, images_paths)
        image_shape = image_provider.image_shape

        depth_provider = PrecomputedDepthProvider(config, image_shape, depths_paths)

        cam2worlds = [Se3_world2cams[i].inverse().matrix().squeeze() for i in range(N_frames)]
        intrinsics = [camera_K[i].camera_matrix.squeeze() for i in range(N_frames)]

        depths = [depth_provider.next_depth(i).squeeze() for i in range(N_frames)]

        compute_all_overlaps(depths, cam2worlds, intrinsics)
        pass


if __name__ == "__main__":

    compute_overlaps_bop('handal')

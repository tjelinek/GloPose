import random
from itertools import product
from pathlib import Path

import plotly.graph_objects as go

import numpy as np
import torch
from kornia.geometry import Se3, Quaternion
from tqdm import tqdm

from data_providers.frame_provider import PrecomputedDepthProvider, PrecomputedFrameProvider, \
    PrecomputedSegmentationProvider, PrecomputedDepthProvider_HO3D
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


def visualize_overlap_debug(depth_a, depth_b, cam2world_a, cam2world_b, intrinsics_a, intrinsics_b,
                            sample_rate=100, width=1000, height=700, coordinate_frame='world',
                            transformed_pts_a=None, pts_b_in_own_frame=None, transformed_cam_a_pos=None):
    """
    Interactive 3D visualization of point clouds from both cameras to debug overlap calculation.

    Args:
        depth_a, depth_b: depth maps
        cam2world_a, cam2world_b: camera-to-world transformation matrices
        intrinsics_a, intrinsics_b: camera intrinsic matrices
        sample_rate: subsample points every N pixels for visualization (default: 100)
        width, height: plot dimensions
        coordinate_frame: 'world' or 'camera_b' - which coordinate system to visualize in
        transformed_pts_a: pre-computed transformed points from camera A (when coordinate_frame='camera_b')
        pts_b_in_own_frame: pre-computed points from camera B in its own frame
    """

    if coordinate_frame == 'world':
        # Original world coordinate visualization
        # Get points from camera A (same as in your original function)
        pts_a = depth_to_xyz(depth_a, intrinsics_a)
        wpts_a = transform_pts(pts_a, cam2world_a)

        # Get points from camera B
        pts_b = depth_to_xyz(depth_b, intrinsics_b)
        wpts_b = transform_pts(pts_b, cam2world_b)

    else:  # coordinate_frame == 'camera_b'
        # Use the exact transformations computed in overlap_ratio
        if transformed_pts_a is None or pts_b_in_own_frame is None or transformed_cam_a_pos is None:
            raise ValueError(
                "For camera_b coordinate frame, must provide transformed_pts_a, pts_b_in_own_frame, and transformed_cam_a_pos")

    # Convert to numpy for visualization (subsample for performance)
    wpts_a_np = wpts_a.detach().cpu().numpy()
    wpts_b_np = wpts_b.detach().cpu().numpy()

    # Subsample points for visualization
    n_pts_a = wpts_a_np.shape[1]
    n_pts_b = wpts_b_np.shape[1]

    indices_a = np.arange(0, n_pts_a, sample_rate)
    indices_b = np.arange(0, n_pts_b, sample_rate)

    pts_a_vis = wpts_a_np[:, indices_a]
    pts_b_vis = wpts_b_np[:, indices_b]

    # Get camera positions
    cam_a_pos = cam2world_a[:3, 3].detach().cpu().numpy()
    cam_b_pos = cam2world_b[:3, 3].detach().cpu().numpy()

    # Create interactive 3D plot
    fig = go.Figure()

    # Add points from camera A (red)
    fig.add_trace(go.Scatter3d(
        x=pts_a_vis[0],
        y=pts_a_vis[1],
        z=pts_a_vis[2],
        mode='markers',
        marker=dict(
            size=2,
            color='red',
            opacity=0.6
        ),
        name='Camera A points',
        hovertemplate='Camera A<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))

    # Add points from camera B (blue)
    fig.add_trace(go.Scatter3d(
        x=pts_b_vis[0],
        y=pts_b_vis[1],
        z=pts_b_vis[2],
        mode='markers',
        marker=dict(
            size=2,
            color='blue',
            opacity=0.6
        ),
        name='Camera B points',
        hovertemplate='Camera B<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))

    # Add camera positions
    fig.add_trace(go.Scatter3d(
        x=[cam_a_pos[0]],
        y=[cam_a_pos[1]],
        z=[cam_a_pos[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='darkred',
            symbol='diamond'
        ),
        name='Camera A position',
        hovertemplate='Camera A Position<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))

    fig.add_trace(go.Scatter3d(
        x=[cam_b_pos[0]],
        y=[cam_b_pos[1]],
        z=[cam_b_pos[2]],
        mode='markers',
        marker=dict(
            size=15,
            color='darkblue',
            symbol='diamond'
        ),
        name='Camera B position',
        hovertemplate='Camera B Position<br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
    ))

    # Calculate proper axis ranges for equal aspect ratio
    all_points = np.concatenate([pts_a_vis, pts_b_vis,
                                 cam_a_pos.reshape(3, 1), cam_b_pos.reshape(3, 1)], axis=1)
    center = np.mean(all_points, axis=1)
    max_range = np.max(np.max(all_points, axis=1) - np.min(all_points, axis=1)) / 2

    # Update layout for better visualization
    fig.update_layout(
        title='Interactive 3D Point Clouds from Both Cameras<br><sub>Click and drag to rotate, scroll to zoom</sub>',
        width=width,
        height=height,
        scene=dict(
            xaxis=dict(
                title='X (world)',
                range=[center[0] - max_range, center[0] + max_range]
            ),
            yaxis=dict(
                title='Y (world)',
                range=[center[1] - max_range, center[1] + max_range]
            ),
            zaxis=dict(
                title='Z (world)',
                range=[center[2] - max_range, center[2] + max_range]
            ),
            aspectmode='cube',  # Equal aspect ratio
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Nice initial viewing angle
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    fig.show()


def overlap_ratio(depth_a, depth_b, cam2world_a, cam2world_b, intrinsics_a, intrinsics_b,
                  thresh=0.005, debug_vis=False):
    # 1) get A's points in world, then into B's camera frame
    pts_a = depth_to_xyz(depth_a, intrinsics_a)
    wpts_a = transform_pts(pts_a, cam2world_a)
    world2b = torch.inverse(cam2world_b)
    bpts = world2b @ torch.cat((wpts_a, torch.ones(1, wpts_a.shape[1], device=wpts_a.device)), dim=0)
    cam_pts = bpts[:3]
    # 2) project into B’s image
    proj = intrinsics_b @ cam_pts
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

    if debug_vis:
        visualize_overlap_debug(depth_a, depth_b, cam2world_a, cam2world_b,
                                intrinsics_a, intrinsics_b, coordinate_frame='camera_b')
        print(f"Overlap ratio: {matched.sum().float() / (in_bounds.sum().float() + 1e-8):.4f}")
        print(f"Points in bounds: {in_bounds.sum()}")
        print(f"Points matched: {matched.sum()}")

    return matched.sum().float() / (in_bounds.sum().float() + 1e-8)


def compute_all_overlaps(depths, cam2worlds, intrinsics, overlap_thresh):
    """
    depths: list of [H,W] tensors
    cam2worlds: list of [4,4] tensors
    intrinsics: [3,3] tensor
    returns: NxN matrix of overlap ratios
    """
    N = len(depths)
    M = torch.eye(N, device=depths[0].device)
    for i in tqdm(range(N), desc='i→j overlaps'):
        for j in range(N):
            if i != j:
                overlap = overlap_ratio(depths[i], depths[j], cam2worlds[i], cam2worlds[j],
                                        intrinsics[i], intrinsics[j], overlap_thresh)
                M[i, j] = overlap
                M[j, i] = overlap
    return M


def compute_overlaps_bop(dataset_name):
    config = TrackerConfig()

    path_to_dataset = Path(f'/mnt/personal/jelint19/data/bop/{dataset_name}')
    training_set = path_to_dataset / 'train_pbr'

    for scene in tqdm(sorted(training_set.iterdir()), desc='scenes'):
        depths_folder = scene / 'depth'
        images_folder = scene / 'rgb'
        scene_camera_path = scene / 'scene_camera.json'
        scene_info_path = scene / 'scene_info.npy'

        print(f'Processing scene {scene}...')

        if scene_info_path.exists():
            continue

        Se3_world2cams = read_gt_Se3_world2cam(scene_camera_path, device=config.device)
        camera_K = get_pinhole_params(scene_camera_path, scale=config.image_downsample, device=config.device)

        images_paths = sorted(images_folder.iterdir())
        depths_paths = sorted(depths_folder.iterdir())
        N_frames = len(images_paths)

        overlap_thresh = 500  # 5 mm
        if dataset_name == 'handal':
            scale = 0.001
            depth_scales = [scale] * N_frames
        elif dataset_name == 'hope':
            scale = 0.1
            depth_scales = [scale] * N_frames
        else:
            raise NotImplementedError("BOP datasets have non-uniform depth scales. Might cause problems.")
        overlap_thresh *= scale

        image_provider = PrecomputedFrameProvider(config, images_paths)
        image_shape = image_provider.image_shape

        depth_provider = PrecomputedDepthProvider(config, image_shape, depths_paths, depth_scales)

        invalid_ids = set()

        cam2worlds = {}
        intrinsics = {}
        depths = {}
        for i in range(N_frames):
            try:
                cam2worlds[i] = Se3_world2cams[i].inverse().matrix().squeeze()
                intrinsics[i] = camera_K[i].camera_matrix.squeeze()
                depths[i] = depth_provider.next_depth(i).squeeze()
            except:
                invalid_ids.add(i)

        valid_ids = sorted(set(range(N_frames)) - invalid_ids)

        depths = [depths[i] for i in valid_ids]
        intrinsics = [intrinsics[i] for i in valid_ids]
        cam2worlds = [cam2worlds[i] for i in valid_ids]

        overlap_matrix = compute_all_overlaps(depths, cam2worlds, intrinsics, overlap_thresh)

        scene_info = {'image_paths': [str(p) for p in images_paths], 'depth_paths': [str(p) for p in depths_paths],
                      'intrinsics': [K.numpy(force=True) for K in intrinsics],
                      'poses': [Se3_world2cams[i].matrix().squeeze().numpy(force=True) for i in range(len(valid_ids))],
                      'pairs': np.array([(i, j) for (i, j) in product(range(len(valid_ids)), repeat=2)
                                         if overlap_matrix[i, j] > 0])}
        scene_info['overlaps'] = np.array([overlap_matrix[i, j].item() for i, j in scene_info['pairs']])

        np.save(str(scene_info_path), scene_info, allow_pickle=True)


def compute_overlaps_ho3d(random_shuffle=True):
    config = TrackerConfig()

    path_to_dataset = Path(f'/mnt/personal/jelint19/data/HO3D/')
    training_set = path_to_dataset / 'train'

    scene_list = list(training_set.iterdir())
    if random_shuffle:
        random.shuffle(scene_list)

    for scene in tqdm(scene_list, desc='scenes'):

        scene_info_path = scene / 'scene_info.npy'

        if scene_info_path.exists():
            continue

        print(f'Processing scene {scene}...')

        images_paths = sorted((scene / 'rgb').iterdir())
        segmentation_paths = sorted((scene / 'seg').iterdir())
        depths_paths = sorted((scene / 'depth').iterdir())
        meta_files = sorted(f for f in (scene / 'meta').iterdir() if Path(f).suffix == '.npz')

        N = len(images_paths)

        image_provider = PrecomputedFrameProvider(config, images_paths)
        image_shape = image_provider.image_shape
        depth_provider = PrecomputedDepthProvider_HO3D(config, image_shape, depths_paths)
        segmentation_provider = PrecomputedSegmentationProvider(config, image_shape, segmentation_paths)

        valid_ids = []
        depths = []
        intrinsics = []
        cam2obj_Ts = []
        segmentations = []

        for i in range(N):

            try:
                segmentation = segmentation_provider.next_segmentation(i).squeeze()
                depth = depth_provider.next_depth(i).squeeze()

                meta_file = meta_files[i]
                data = np.load(meta_file, allow_pickle=True)
                data_dict = {key: data[key] for key in data}

                cam_K = torch.from_numpy(data_dict['camMat']).to(config.device).to(torch.float32)
                cam2obj_R = torch.from_numpy(data_dict['objRot']).to(config.device).squeeze()
                cam2obj_t = torch.from_numpy(data_dict['objTrans']).to(config.device)
                cam2obj_Se3 = Se3(Quaternion.from_axis_angle(cam2obj_R), cam2obj_t)
                cam2obj_T = cam2obj_Se3.matrix().squeeze()

                valid_ids.append(i)
                intrinsics.append(cam_K)
                cam2obj_Ts.append(cam2obj_T)
                depths.append(depth)
                segmentations.append(segmentation)

            except Exception:
                pass

        overlap_matrix = compute_all_overlaps(depths, cam2obj_Ts, intrinsics)

        scene_info = {'image_paths': [str(p) for p in images_paths], 'depth_paths': [str(p) for p in depths_paths],
                      'intrinsics': [K.numpy(force=True) for K in intrinsics],
                      'poses': [Se3.from_matrix(cam2obj_T).inverse().matrix().squeeze().numpy(force=True)
                                for cam2obj_T in cam2obj_Ts],
                      'pairs': np.array([(i, j) for (i, j) in product(range(len(valid_ids)), repeat=2)
                                         if overlap_matrix[i, j] > 0])}
        scene_info['overlaps'] = np.array([overlap_matrix[i, j].item() for i, j in scene_info['pairs']])

        np.save(str(scene_info_path), scene_info, allow_pickle=True)



if __name__ == "__main__":
    # compute_overlaps_bop('hope')
    compute_overlaps_ho3d()

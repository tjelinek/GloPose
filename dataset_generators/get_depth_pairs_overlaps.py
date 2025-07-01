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


def transform_pts(pts, T_4x4):
    """
    pts: [3, N] points
    T_4x4: [4, 4] transformation matrix
    returns: [3, N] transformed points
    """
    homo = torch.cat((pts, torch.ones(1, pts.shape[1], device=pts.device)), dim=0)
    pts_transformed = T_4x4 @ homo
    return pts_transformed[:3]


def visualize_overlap_debug(camA_pts_camB, camB_pts_camB, camA_camB, sample_rate=100, width=1000, height=700):

    # Convert to numpy for visualization (subsample for performance)
    camA_pts_camB_np = camA_pts_camB.numpy(force=True)
    camB_pts_camB_np = camB_pts_camB.numpy(force=True)

    # Subsample points for visualization
    n_pts_a = camA_pts_camB_np.shape[1]
    n_pts_b = camB_pts_camB_np.shape[1]

    indices_a = np.arange(0, n_pts_a, sample_rate)
    indices_b = np.arange(0, n_pts_b, sample_rate)

    pts_a_vis = camA_pts_camB_np[:, indices_a]
    pts_b_vis = camB_pts_camB_np[:, indices_b]

    # Get camera positions
    cam_a_pos = camA_camB.numpy(force=True)
    cam_b_pos = camA_camB.numpy(force=True)

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


def overlap_ratio(depthA, depthB, camA2world, camB2world, intrinsicsA, intrinsicsB,
                  thresh=0.005, debug_vis=False):
    # Naming convention: <object>_<coordinate_system>_<mode>
    camA_pts_camA = depth_to_xyz(depthA, intrinsicsA)

    T_world2camB = torch.inverse(camB2world)
    T_camA2camB = T_world2camB @ camA2world

    camA_pts_camB = transform_pts(camA_pts_camA, T_camA2camB)

    camA_pts_imB = intrinsicsB @ camA_pts_camB
    camA_pts_imB_u = camA_pts_imB[0] / camA_pts_imB[2]
    camA_pts_imB_v = camA_pts_imB[1] / camA_pts_imB[2]
    camA_pts_imB_before_cam = (camA_pts_imB[2] > 0)
    camA_pts_imB_u_int = camA_pts_imB_u.round().long()
    camA_pts_imB_v_int = camA_pts_imB_v.round().long()
    H, W = depthB.shape
    camA_pts_camB_in_bounds = (camA_pts_imB_before_cam & (camA_pts_imB_u_int >= 0) & (camA_pts_imB_u_int < W) &
                              (camA_pts_imB_v_int >= 0) & (camA_pts_imB_v_int < H))
    camA_pts_imB_u_int = camA_pts_imB_u_int[camA_pts_camB_in_bounds]
    camA_pts_imB_v_int = camA_pts_imB_v_int[camA_pts_camB_in_bounds]
    camA_pts_camB_zs = camA_pts_camB[2, camA_pts_camB_in_bounds]

    camB_pts_masked_zs = depthB[camA_pts_imB_v_int, camA_pts_imB_u_int]
    matched = torch.abs(camB_pts_masked_zs - camA_pts_camB_zs) < thresh

    if debug_vis:
        camB_pts_camB = depth_to_xyz(depthB, intrinsicsB)
        camA_camA = torch.zeros(3, 1, device=camA_pts_imB.device)
        camA_camB = transform_pts(camA_camA, T_camA2camB).squeeze()

        visualize_overlap_debug(camA_pts_camB, camB_pts_camB, camA_camB)
        print(f"Overlap ratio: {matched.sum().float() / (camA_pts_camB_in_bounds.sum().float() + 1e-8):.4f}")
        print(f"Points in bounds: {camA_pts_camB_in_bounds.sum()}")
        print(f"Points matched: {matched.sum()}")

    return matched.sum().float() / (camA_pts_camB_in_bounds.sum().float() + 1e-8)


def compute_all_overlaps(cam2worlds, intrinsics, depths, overlap_thresh, delimiters: np.ndarray = None):
    """
    depths: list of [H,W] tensors
    cam2worlds: list of [4,4] tensors
    intrinsics: [3,3] tensor
    returns: NxN matrix of overlap ratios
    """
    N = len(depths)
    M = torch.eye(N, device=depths[0].device)
    for i in tqdm(range(N), desc='i→j overlaps'):
        if delimiters is not None:
            delimiters_idx = np.argmax(delimiters > i)
            j_start = delimiters[delimiters_idx - 1]
            j_end = delimiters[delimiters_idx]
        else:
            j_start = 0
            j_end = N

        for j in range(j_start, j_end):
            if i != j:
                overlap = overlap_ratio(depths[i], depths[j], cam2worlds[i], cam2worlds[j],
                                        intrinsics[i], intrinsics[j], overlap_thresh)
                M[i, j] = overlap
                M[j, i] = overlap
    M_np = M.numpy(force=True)
    return M


def compute_overlaps_bop(dataset_name, device='cuda'):
    config = TrackerConfig()
    config.device = device

    path_to_dataset = Path(f'/mnt/personal/jelint19/data/bop/{dataset_name}')
    training_set = path_to_dataset / 'train_pbr'

    for scene in tqdm(sorted(training_set.iterdir()), desc='scenes'):
        depths_folder = scene / 'depth'
        images_folder = scene / 'rgb'
        scene_camera_path = scene / 'scene_camera.json'
        scene_info_path = scene / 'scene_info.npy'

        print(f'Processing scene {scene}...')

        # if scene_info_path.exists():
        #     continue

        camera_K = get_pinhole_params(scene_camera_path, scale=config.image_downsample, device=config.device)

        missing_intrinsics = [i for i in range(1000) if i not in camera_K.keys()]

        if len(missing_intrinsics) != 0:
            print(f"\nManually adding intrinsics {missing_intrinsics}")
            for m_idx in missing_intrinsics:
                closest = np.argmin(np.abs(m_idx - np.asarray(sorted(camera_K.keys()))))
                camera_K[m_idx] = camera_K[closest]

        images_paths = sorted(images_folder.iterdir())
        depths_paths = sorted(depths_folder.iterdir())
        N_frames = len(images_paths)

        overlap_thresh = 0.005  # 5 mm
        if dataset_name == 'handal':
            depth_scale_to_meter = 0.001
            extrinsics_input_scale = 'm'
        elif dataset_name == 'hope':
            depth_scale_to_meter = 0.0001
            extrinsics_input_scale = 'mm'
        else:
            raise NotImplementedError("BOP datasets have non-uniform depth scales. Might cause problems.")
        depth_output_unit = 'm'
        extrinsics_output_scale = 'm'

        Se3_world2cams = read_gt_Se3_world2cam(scene_camera_path, input_scale=extrinsics_input_scale,
                                               output_scale=extrinsics_output_scale, device=config.device)

        image_provider = PrecomputedFrameProvider(config, images_paths)
        image_shape = image_provider.image_shape

        depth_provider = PrecomputedDepthProvider(config, image_shape, depths_paths,
                                                  depth_scale_to_meter=depth_scale_to_meter,
                                                  output_unit=depth_output_unit)

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
        images_paths = [images_paths[i] for i in valid_ids]
        depths_paths = [depths_paths[i] for i in valid_ids]
        Se3_world2cams = [Se3_world2cams[i] for i in valid_ids]

        N_frames = len(valid_ids)
        if dataset_name == 'handal':
            delimiters = np.concatenate([np.arange(0, N_frames, 25), [N_frames]])
        elif dataset_name == 'hope':
            delimiters = np.concatenate([np.arange(0, N_frames, 25), [N_frames]])
        else:
            raise NotImplementedError("BOP datasets have non-uniform depth scales. Might cause problems.")

        overlap_matrix = compute_all_overlaps(cam2worlds, intrinsics, depths, overlap_thresh, delimiters=delimiters)

        scene_info = {'image_paths': [str(p) for p in images_paths], 'depth_paths': [str(p) for p in depths_paths],
                      'depth_scale_to_meter': depth_scale_to_meter,
                      # Multiplicative scale to convert depth scale to meter
                      'intrinsics': [K.numpy(force=True) for K in intrinsics],
                      'poses': [Se3_world2cams[i].matrix().squeeze().numpy(force=True) for i in range(len(valid_ids))],
                      'pairs': np.array([(i, j) for (i, j) in product(range(len(valid_ids)), repeat=2)
                                         if overlap_matrix[i, j] > 0])}
        scene_info['overlaps'] = np.array([overlap_matrix[i, j].item() for i, j in scene_info['pairs']])

        np.save(str(scene_info_path), scene_info, allow_pickle=True)


def compute_overlaps_ho3d(random_shuffle=True, device='cuda'):
    config = TrackerConfig()
    config.device = device

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

        overlap_matrix = compute_all_overlaps(cam2obj_Ts, intrinsics, depths, 0.005)

        scene_info = {'image_paths': [str(p) for p in images_paths], 'depth_paths': [str(p) for p in depths_paths],
                      'intrinsics': [K.numpy(force=True) for K in intrinsics],
                      'poses': [Se3.from_matrix(cam2obj_T).inverse().matrix().squeeze().numpy(force=True)
                                for cam2obj_T in cam2obj_Ts],
                      'pairs': np.array([(i, j) for (i, j) in product(range(len(valid_ids)), repeat=2)
                                         if overlap_matrix[i, j] > 0])}
        scene_info['overlaps'] = np.array([overlap_matrix[i, j].item() for i, j in scene_info['pairs']])

        np.save(str(scene_info_path), scene_info, allow_pickle=True)


if __name__ == "__main__":
    compute_overlaps_bop('hope', 'cuda')
    compute_overlaps_bop('handal', 'cuda')
    # compute_overlaps_ho3d('cuda)

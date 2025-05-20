#!/usr/bin/env python3
"""
Script to render per-frame segmentation masks of the object from HO3Dv3 evaluation splits, with debug outputs to validate transforms.

Usage:
    python render_segmentations_debug.py
"""
import os
import copy
import numpy as np
import open3d as o3d
import cv2

from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from open3d.camera import PinholeCameraIntrinsic

# Debug flag: number of frames to debug-print
DEBUG_FRAMES = 5

def find_mesh_file(mesh_root, obj_name):
    mesh_dir = os.path.join(mesh_root, obj_name)
    for fname in ['textured_simple.obj', 'textured.obj']:
        path = os.path.join(mesh_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"No mesh found for object {obj_name} in {mesh_dir}")

def render_sequence(seq_path, mesh_root, out_seq_path):
    meta_dir = os.path.join(seq_path, 'meta')
    rgb_dir  = os.path.join(seq_path, 'rgb')

    # Get image size
    first_rgb = sorted(os.listdir(rgb_dir))[0]
    img = cv2.imread(os.path.join(rgb_dir, first_rgb), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]

    # Setup renderer
    renderer = OffscreenRenderer(w, h)
    mat = MaterialRecord()
    mat.shader = "defaultLit"

    # Output directory for masks
    seg_dir = os.path.join(out_seq_path, 'segmentation_rendered')
    os.makedirs(seg_dir, exist_ok=True)

    # Collect meta files
    meta_files = sorted([f for f in os.listdir(meta_dir)
                         if f.endswith('.npz') or f.endswith('.pkl')])
    if not meta_files:
        print(f"No meta files in {meta_dir}")
        return

    # Load base mesh once
    first_meta = np.load(os.path.join(meta_dir, meta_files[0]), allow_pickle=True)
    obj_name = first_meta['objName'].item() if isinstance(first_meta['objName'], np.ndarray) else first_meta['objName']
    mesh_path = find_mesh_file(mesh_root, obj_name)
    base_mesh = o3d.io.read_triangle_mesh(mesh_path)
    base_mesh.compute_vertex_normals()
    center_local = np.asarray(base_mesh.get_center())

    for idx, mf in enumerate(meta_files):
        # Load per-frame data
        data = np.load(os.path.join(meta_dir, mf), allow_pickle=True)
        aa = data['objRot'].squeeze()
        t  = data['objTrans'].squeeze()
        camMat = data['camMat']

        # Compute rotation matrix for cam2obj
        R_co, _ = cv2.Rodrigues(aa)
        t_co = t.reshape(3,1)
        # Build cam2obj transform
        T_co = np.eye(4)
        T_co[:3,:3] = R_co
        T_co[:3, 3] = t_co.flatten()
        # Invert to get obj2cam
        T_oc = np.linalg.inv(T_co)
        R_oc = T_oc[:3,:3]
        t_oc = T_oc[:3,3]

        # Debug prints
        if idx < DEBUG_FRAMES:
            print(f"--- Frame {idx} Debug ---")
            print("Cam2Obj R_co:\n", R_co)
            print("Cam2Obj t_co:\n", t_co.flatten())
            print("Obj2Cam R_oc:\n", R_oc)
            print("Obj2Cam t_oc:\n", t_oc)
            # Project mesh center
            center_cam = R_oc @ center_local + t_oc
            X, Y, Z = center_cam
            u = camMat[0,0] * X / Z + camMat[0,2]
            v = camMat[1,1] * Y / Z + camMat[1,2]
            print(f"Center in camera coords: {center_cam}")
            print(f"Projected center (u,v): ({u:.1f}, {v:.1f})\n")

        # Render mesh in camera frame
        renderer.scene.clear_geometry()
        mesh_copy = copy.deepcopy(base_mesh)
        mesh_copy.transform(T_oc)
        renderer.scene.add_geometry("obj", mesh_copy, mat)

        # Setup camera intrinsics
        fx, fy = camMat[0,0], camMat[1,1]
        cx, cy = camMat[0,2], camMat[1,2]
        intrinsic = PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        renderer.setup_camera(intrinsic, np.eye(4))

        # Render depth
        depth_o3d = renderer.render_to_depth_image(False)
        depth = np.asarray(depth_o3d)

        # Debug depth stats
        if idx < DEBUG_FRAMES:
            nonzeros = depth[depth > 0]
            if nonzeros.size:
                print(f"Depth stats frame {idx}: min={nonzeros.min():.3f}, max={nonzeros.max():.3f}, mean={nonzeros.mean():.3f}\n")
            else:
                print("No valid depth values (all zeros).\n")

        # Segmentation mask
        mask = (depth > 0).astype(np.uint8) * 255
        seg_file = os.path.join(seg_dir, f"{idx:06d}.png")
        cv2.imwrite(seg_file, mask)

    print(f"Saved segmentation masks to {seg_dir}")


def main():
    # Hardcoded paths
    eval_root = '/mnt/personal/jelint19/data/HO3D/evaluation'
    mesh_root = '/mnt/personal/jelint19/data/HO3D/YCB_Video_Models'

    for seq in sorted(os.listdir(eval_root)):
        seq_path = os.path.join(eval_root, seq)
        if not os.path.isdir(seq_path):
            continue
        print(f"Rendering sequence: {seq}")
        render_sequence(seq_path, mesh_root, seq_path)

if __name__ == '__main__':
    main()

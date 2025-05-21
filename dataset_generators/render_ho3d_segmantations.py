#!/usr/bin/env python3
"""
Script to render per-frame segmentation masks of the object from HO3Dv3 evaluation splits using Open3D,
applying the correct transform convention (object→camera) and handling multi-camera annotations.

Usage:
    python render_segmentations_open3d.py
"""
import os
import copy
import numpy as np
import open3d as o3d
import cv2

from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from open3d.camera import PinholeCameraIntrinsic


def find_mesh_file(mesh_root, obj_name):
    mesh_dir = os.path.join(mesh_root, obj_name)
    for fname in ['textured_simple.obj', 'textured.obj']:
        path = os.path.join(mesh_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Mesh not found for {obj_name} in {mesh_dir}")


def render_sequence(seq_path, mesh_root):
    meta_dir = os.path.join(seq_path, 'meta')
    rgb_dir = os.path.join(seq_path, 'rgb')

    # create output dir
    out_dir = os.path.join(seq_path, 'segmentation_rendered')
    os.makedirs(out_dir, exist_ok=True)

    # read image size for renderer
    first_rgb = sorted(os.listdir(rgb_dir))[0]
    img = cv2.imread(os.path.join(rgb_dir, first_rgb), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]

    # offscreen renderer
    renderer = OffscreenRenderer(w, h)
    mat = MaterialRecord()
    mat.shader = "defaultUnlit"

    # collect meta files
    meta_files = sorted(f for f in os.listdir(meta_dir) if f.endswith('.npz') or f.endswith('.pkl'))
    if not meta_files:
        return

    # load base mesh once
    first_meta = np.load(os.path.join(meta_dir, meta_files[0]), allow_pickle=True)
    obj_name = first_meta['objName'].item() if isinstance(first_meta['objName'], np.ndarray) else first_meta['objName']
    mesh_path = find_mesh_file(mesh_root, obj_name)
    base_mesh = o3d.io.read_triangle_mesh(mesh_path)
    base_mesh.compute_vertex_normals()

    for idx, mf in enumerate(meta_files):
        # load metadata
        data = np.load(os.path.join(meta_dir, mf), allow_pickle=True)
        raw_rot = data['objRot']
        raw_trans = data['objTrans']

        # extract for correct camera
        aa = np.array(raw_rot)
        t = np.array(raw_trans)
        camMat = data['camMat']

        # correct transform: object->camera rotation and translation
        R_co, _ = cv2.Rodrigues(aa.astype(np.float64))
        T_final = np.eye(4)
        T_final[:3, :3] = R_co
        T_final[:3, 3] = t

        # clear and add mesh
        renderer.scene.clear_geometry()
        mesh_copy = copy.deepcopy(base_mesh)
        mesh_copy.transform(T_final)

        renderer.scene.add_geometry("obj", mesh_copy, mat)

        fx, fy = camMat[0, 0], camMat[1, 1]
        cx, cy = camMat[0, 2], camMat[1, 2]
        intrinsic = PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        renderer.setup_camera(intrinsic, np.eye(4))

        # render depth buffer (raw)
        depth_o3d = renderer.render_to_depth_image(False)
        depth = np.asarray(depth_o3d)

        depth_o3d = renderer.render_to_depth_image(True)
        depth = np.asarray(depth_o3d, dtype=np.float32)

        mask = np.isfinite(depth)

        mask_img = (mask.astype(np.uint8) * 255)
        seg_file = os.path.join(out_dir, f"{idx:06d}.png")
        cv2.imwrite(seg_file, mask_img)


def main():
    eval_root = '/mnt/personal/jelint19/data/HO3D/evaluation'
    mesh_root = '/mnt/personal/jelint19/data/HO3D/models'

    for seq in sorted(os.listdir(eval_root)):
        seq_path = os.path.join(eval_root, seq)
        if not os.path.isdir(seq_path):
            continue
        print(f"Processing sequence: {seq}")
        render_sequence(seq_path, mesh_root)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Script to render per-frame segmentation masks of the object from HO3Dv3 evaluation splits using Open3D,
applying the correct transform convention (object→camera) and handling multi-camera annotations.

Usage:
    python render_segmentations_open3d.py
"""
import os
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.append('../repositories/ho3d')
sys.path.append('../repositories/ho3d/mano')

import numpy as np
import open3d as o3d
import cv2
from kornia.image import ImageSize

from open3d.visualization.rendering import OffscreenRenderer, MaterialRecord
from open3d.camera import PinholeCameraIntrinsic
from tqdm import tqdm

from repositories.ho3d.mano.webuser.smpl_handpca_wrapper_HAND_only import load_model
from tracker_config import TrackerConfig


def find_mesh_file(mesh_root, obj_name):
    mesh_dir = os.path.join(mesh_root, obj_name)
    for fname in ['textured_simple.obj', 'textured.obj']:
        path = os.path.join(mesh_dir, fname)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Mesh not found for {obj_name} in {mesh_dir}")


def render_sequence(seq_path: Path, mesh_root: Path, evaluation_verts, evaluation_mapping_path):
    # Depth consistency check is unreliable
    meta_dir = os.path.join(seq_path, 'meta')
    rgb_dir = os.path.join(seq_path, 'rgb')

    with open(evaluation_mapping_path, 'r') as f:
        eval_mapping = f.read().strip().split('\n')

    eval_mapping_to_index = defaultdict(dict)
    for i, line in enumerate(eval_mapping):
        sequence, frame = line.split('/')
        eval_mapping_to_index[sequence][frame] = i

    # create output dir
    out_dir = os.path.join(seq_path, 'segmentation_rendered')
    os.makedirs(out_dir, exist_ok=True)

    mano_model_path = '../repositories/ho3d/mano/models/MANO_RIGHT.pkl'
    mano_model = load_model(mano_model_path)
    mano_faces = mano_model.f

    # read image size for renderer
    first_rgb = sorted(os.listdir(rgb_dir))[0]
    img = cv2.imread(os.path.join(rgb_dir, first_rgb), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]

    # offscreen renderer
    obj_renderer = OffscreenRenderer(w, h)
    hand_renderer = OffscreenRenderer(w, h)
    obj_renderer.scene.set_background([0, 0, 0, 0])  # Transparent background
    hand_renderer.scene.set_background([0, 0, 0, 0])  # Transparent background

    # obj_mat = MaterialRecord()
    # obj_mat.shader = "defaultUnlit"
    # obj_mat.base_color = [0.0, 1.0, 0.0, 1.0]  # Green for object
    #
    # hand_mat = MaterialRecord()
    # hand_mat.shader = "defaultUnlit"
    # hand_mat.base_color = [0.0, 0.0, 1.0, 1.0]  # Blue for hand

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

    for idx, mf in enumerate(tqdm(meta_files, desc=f"Frames ({os.path.basename(seq_path)})")):

        if idx > 10:
            exit()
        # load metadata
        data = np.load(os.path.join(meta_dir, mf), allow_pickle=True)
        raw_rot = data['objRot']
        raw_trans = data['objTrans']

        # extract for correct camera
        aa = np.array(raw_rot)
        t = np.array(raw_trans)
        camMat = data['camMat']

        # correct transform: object->camera rotation and translation
        try:
            R_co, _ = cv2.Rodrigues(aa.astype(np.float64))
        except:
            print(f'Frame {idx} corrupted')
            continue

        seq_name = seq_path.name
        frame_name = Path(mf).stem
        if not eval_mapping_to_index[seq_name].get(frame_name):
            print(f'Frame {idx} missing hand file')
            continue

        hand_vertices = np.asarray(evaluation_verts[eval_mapping_to_index[seq_name][frame_name]])

        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_vertices)
        hand_mesh.triangles = o3d.utility.Vector3iVector(mano_faces)
        hand_mesh.compute_vertex_normals()

        T_final = np.eye(4)
        T_final[:3, :3] = R_co
        T_final[:3, 3] = t

        # clear and add mesh
        mesh_copy = copy.deepcopy(base_mesh)
        mesh_copy.transform(T_final)
        T_flip = np.asarray([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
            [0., 0., 0., 1.],
        ])
        #  OpenCV/HO3D camera coordinate system into the OpenGL coordinate system
        mesh_copy.transform(T_flip)

        hand_mesh_copy = copy.deepcopy(hand_mesh)
        hand_mesh_copy.transform(T_flip)

        obj_renderer.scene.add_geometry("obj", mesh_copy, mat)
        hand_renderer.scene.add_geometry("hand", hand_mesh_copy, mat)

        fx, fy = camMat[0, 0], camMat[1, 1]
        cx, cy = camMat[0, 2], camMat[1, 2]
        intrinsic = PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
        obj_renderer.setup_camera(intrinsic, np.eye(4))
        hand_renderer.setup_camera(intrinsic, np.eye(4))

        # color_o3d = renderer.render_to_image()  # default color pass
        # color = np.asarray(color_o3d)  # to numpy H×W×3 uint8
        # cv2.imwrite(os.path.join(out_dir, f"color_{idx:06d}.png"), color)

        obj_renderer.scene.clear_geometry()
        obj_renderer.scene.add_geometry("object", mesh_copy, mat)
        obj_renderer.setup_camera(intrinsic, np.eye(4))
        obj_depth = np.asarray(obj_renderer.render_to_depth_image(True))
        obj_mask = np.isfinite(obj_depth)

        # Render hand mask
        hand_renderer.scene.clear_geometry()
        hand_renderer.scene.add_geometry("hand", hand_mesh_copy, mat)
        hand_renderer.setup_camera(intrinsic, np.eye(4))
        hand_depth = np.asarray(hand_renderer.render_to_depth_image(True))
        hand_mask = np.isfinite(hand_depth)

        # Combine masks with depth ordering
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        # Object pixels where no hand occlusion
        obj_pixels = obj_mask & (~hand_mask | (obj_depth < hand_depth))
        combined_mask[obj_pixels] = 1

        # Hand pixels (closer to camera)
        hand_pixels = hand_mask & (~obj_mask | (hand_depth <= obj_depth))
        combined_mask[hand_pixels] = 2

        # Convert to color
        segmentation_color = np.zeros((h, w, 3), dtype=np.uint8)
        segmentation_color[combined_mask == 1] = [0, 255, 0]  # Green for object
        segmentation_color[combined_mask == 2] = [0, 0, 255]  # Blue for hand

        seg_file = os.path.join(out_dir, f"{idx:06d}.png")
        cv2.imwrite(seg_file, segmentation_color)


def annotation_list_to_dict(eval_root: Path, annotation_path: Path, annotation_dict_path: Path):

    with open(annotation_path, 'r') as f:
        annotation_list = json.load(f)

    total_num_files = 0
    annotation_dict = defaultdict(dict)

    for seq_folder in tqdm(sorted(eval_root.iterdir())):
        rgb_folder = seq_folder / 'meta'
        seq_name = seq_folder.name

        for meta_name in tqdm(sorted(rgb_folder.iterdir())):
            meta_id = meta_name.stem

            data = np.load(str(meta_name), allow_pickle=True)
            if type(data['objRot']) is not np.ndarray:
                print(f'Ignoring {seq_name}/{meta_id}')
                continue

            if total_num_files >= len(annotation_list):
                print(f'Ignoring {seq_name}/{meta_id}, out of bounds: {total_num_files}/{len(annotation_list)}')
                continue
            annotation_dict[seq_name][meta_id] = annotation_list[total_num_files]

            total_num_files += 1

    print(f'Processed {total_num_files}/{len(annotation_list)} annotations')
    with open(annotation_dict_path, 'w') as f:
        json.dump(annotation_dict, f, indent=2)


def main():

    ho3d_data_root = Path('/mnt/personal/jelint19/data/HO3D/')
    eval_root = ho3d_data_root / 'evaluation'
    mesh_root = ho3d_data_root / 'models'

    eval_xyz_path = ho3d_data_root / 'evaluation_xyz.json'
    eval_verts_path = ho3d_data_root / 'evaluation_verts.json'
    eval_xyz_dict_path = ho3d_data_root / 'evaluation_xyz_dict.json'
    eval_verts_dict_path = ho3d_data_root / 'evaluation_verts_dict.json'
    evaluation_mapping = ho3d_data_root / 'evaluation.txt'
    # eval_xyz_path = Path('~/Downloads/evaluation_xyz.json').expanduser()
    # eval_verts_path = Path('~/Downloads/evaluation_verts.json').expanduser()
    # eval_xyz_dict_path = Path('~/Downloads/evaluation_xyz_dict.json').expanduser()
    # eval_verts_dict_path = Path('~/Downloads/evaluation_verts_dict.json').expanduser()

    if not eval_xyz_dict_path.exists():
        print('Converting evaluation_xyz to dict.')
        annotation_list_to_dict(eval_root, eval_xyz_path, eval_xyz_dict_path)

    if not eval_verts_dict_path.exists():
        print('Converting evaluation_xyz to dict.')
        annotation_list_to_dict(eval_root, eval_verts_path, eval_verts_dict_path)

    with open(eval_xyz_dict_path, 'r') as f:
        evaluation_xyz_dict = json.load(f)

    with open(eval_verts_path, 'r') as f:
        evaluation_verts = json.load(f)

    for seq in tqdm(sorted(os.listdir(eval_root)), desc="Sequences"):
        seq_path = eval_root / seq
        if not os.path.isdir(seq_path):
            continue
        print(f"Processing sequence: {seq}")
        render_sequence(seq_path, mesh_root, evaluation_verts, evaluation_mapping)


if __name__ == '__main__':
    main()

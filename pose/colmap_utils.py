import shutil
from pathlib import Path
from typing import Dict

import pycolmap
import torch
from kornia.geometry import Se3, Quaternion


def get_image_Se3_world2cam(image: pycolmap.Image, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = image.cam_from_world
    image_t_cam = torch.tensor(image_world2cam.translation).to(device).to(torch.float)
    image_q_cam_xyzw = torch.tensor(image_world2cam.rotation.quat[[3, 0, 1, 2]]).to(device).to(torch.float)
    Se3_image_world2cam = Se3(Quaternion(image_q_cam_xyzw), image_t_cam)

    return Se3_image_world2cam


def world2cam_from_reconstruction(reconstruction: pycolmap.Reconstruction) -> Dict[int, Se3]:
    poses = {}
    for image_id, image in reconstruction.images.items():
        Se3_world2cam = get_image_Se3_world2cam(image, 'cpu')
        poses[image_id] = Se3_world2cam
    return poses


def merge_two_databases(colmap_db1_path: Path, colmap_db2_path: Path, merged_db_path: Path, db2_imgs_prefix="db2_") \
        -> Dict[str, str]:
    # Ensure output directory exists

    print("Analyzing databases...")

    db1 = pycolmap.Database(str(colmap_db1_path))

    tmp_db2_path = merged_db_path.parent / 'tmp_db2.db'
    shutil.copy(colmap_db2_path, tmp_db2_path)
    tmp_db2 = pycolmap.Database(str(tmp_db2_path))

    db2_rename_dict = {}
    for image in tmp_db2.read_all_images():

        old_name = image.name
        new_name = db2_imgs_prefix + image.name
        image.name = new_name
        tmp_db2.update_image(image)

        db2_rename_dict[old_name] = new_name

    merged_db = pycolmap.Database(str(merged_db_path))
    pycolmap.Database.merge(db1, tmp_db2, merged_db)

    tmp_db2_path.unlink()

    return db2_rename_dict

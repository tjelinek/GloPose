import shutil
from pathlib import Path
from typing import Dict, Tuple

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


def merge_two_databases(colmap_db1_path: Path, colmap_db2_path: Path, merged_db_path: Path, db1_imgs_prefix="db1_",
                        db2_imgs_prefix="db2_") \
        -> Tuple[Dict[str, str], Dict[str, str]]:

    db1 = pycolmap.Database(str(colmap_db1_path))

    tmp_db1_path = merged_db_path.parent / 'tmp_db1.db'
    tmp_db2_path = merged_db_path.parent / 'tmp_db2.db'

    shutil.copy(colmap_db1_path, tmp_db1_path)
    shutil.copy(colmap_db2_path, tmp_db2_path)

    tmp_db1 = pycolmap.Database(str(tmp_db1_path))
    tmp_db2 = pycolmap.Database(str(tmp_db2_path))

    def rename_db_imgs(tmp_db: pycolmap.Database, db_imgs_prefix: str):
        db_rename_dict = {}
        for image in tmp_db.read_all_images():

            old_name = image.name
            new_name = db_imgs_prefix + image.name
            image.name = new_name
            tmp_db.update_image(image)

            db_rename_dict[old_name] = new_name

        return db_rename_dict

    db1_rename_dict = rename_db_imgs(tmp_db1, db1_imgs_prefix)
    db2_rename_dict = rename_db_imgs(tmp_db2, db2_imgs_prefix)

    merged_db = pycolmap.Database(str(merged_db_path))
    pycolmap.Database.merge(tmp_db1, tmp_db2, merged_db)

    tmp_db1_path.unlink()
    tmp_db2_path.unlink()

    db1.close()
    merged_db.close()

    return db1_rename_dict, db2_rename_dict


def merge_colmap_reconstructions(rec1: pycolmap.Reconstruction, rec2: pycolmap.Reconstruction)\
        -> pycolmap.Reconstruction:
    max_camera_id = max(rec1.cameras.keys()) if rec1.cameras else 0

    # Find the max image ID in rec1 to avoid conflicts
    max_image_id = max(rec1.images.keys()) if rec1.images else 0

    # Map old camera IDs to new camera IDs
    camera_id_mapping = {}

    # Map old image IDs to new image IDs
    image_id_mapping = {}

    # Add all cameras from rec2, assigning new IDs
    for old_camera_id, camera in rec2.cameras.items():
        max_camera_id += 1
        camera.camera_id = max_camera_id
        rec1.add_camera(camera)
        camera_id_mapping[old_camera_id] = max_camera_id

    # Add all images from rec2, creating new images with updated camera and image IDs
    for old_image_id, image in rec2.images.items():
        max_image_id += 1

        # Create clean points2D without 3D point associations
        clean_points2D = pycolmap.ListPoint2D()
        for point2D in image.points2D:
            # Create new Point2D without the 3D point association
            clean_point2D = pycolmap.Point2D(xy=point2D.xy)
            clean_points2D.append(clean_point2D)

        # Create new image with cleaned points2D
        new_image = pycolmap.Image(
            name=image.name,
            points2D=clean_points2D,
            cam_from_world=image.cam_from_world,
            camera_id=camera_id_mapping[image.camera_id],
            id=max_image_id
        )
        rec1.add_image(new_image)
        image_id_mapping[old_image_id] = max_image_id

    # Add all 3D points from rec2, updating track image IDs
    for point3D in rec2.points3D.values():
        # Create new track with updated image IDs
        new_track = pycolmap.Track()
        for track_element in point3D.track.elements:
            new_track.add_element(
                image_id_mapping[track_element.image_id],
                track_element.point2D_idx
            )

        rec1.add_point3D(point3D.xyz, new_track, point3D.color)

    return rec1

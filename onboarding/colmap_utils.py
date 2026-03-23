import shutil
from pathlib import Path
from typing import Dict, Tuple

import pycolmap
import torch
from kornia.geometry import Se3, Quaternion

# pycolmap 4.0 introduces Rig→Frame→Image hierarchy; 3.x has Image.set_cam_from_world()
PYCOLMAP_MAJOR = int(pycolmap.__version__.split('.')[0])
PYCOLMAP4 = PYCOLMAP_MAJOR >= 4


def make_point2d_list(points: list | None = None):
    """Create a Point2D list container, compatible with both pycolmap 3.x and 4.x."""
    cls = pycolmap.Point2DList if hasattr(pycolmap, 'Point2DList') else pycolmap.ListPoint2D
    return cls(points) if points is not None else cls()


def add_posed_image_to_reconstruction(
    rec: pycolmap.Reconstruction,
    image_id: int,
    camera_id: int,
    name: str,
    cam_from_world: pycolmap.Rigid3d,
    points2D=None,
) -> pycolmap.Image:
    """Add an image with its pose to a reconstruction.

    Handles both pycolmap 3.x (Image.set_cam_from_world) and
    4.x (Rig→Frame→Image chain) transparently.
    """
    if points2D is not None:
        img = pycolmap.Image(name=name, points2D=points2D, camera_id=camera_id, image_id=image_id)
    else:
        img = pycolmap.Image(image_id=image_id, camera_id=camera_id, name=name)

    if PYCOLMAP4:
        cam = rec.cameras[camera_id]

        # One rig per camera (reused across images sharing the same camera)
        rig_id = camera_id
        if rig_id not in rec.rigs:
            rig = pycolmap.Rig(rig_id=rig_id)
            rig.add_ref_sensor(cam.sensor_id)
            rec.add_rig(rig)

        # Create frame, link to image, add to reconstruction, then set pose
        frame_id = image_id  # 1:1 mapping between frames and images
        img.frame_id = frame_id
        frame = pycolmap.Frame(frame_id=frame_id, rig_id=rig_id)
        frame.add_data_id(img.data_id)
        rec.add_frame(frame)
        rec.frames[frame_id].set_cam_from_world(camera_id, cam_from_world)

        rec.add_image(img)
        rec.register_frame(frame_id)
    else:
        img.set_cam_from_world(camera_id, cam_from_world)
        rec.add_image(img)

    return img


def create_database_cache(database: pycolmap.Database):
    """Create a DatabaseCache, compatible with both pycolmap 3.x and 4.x."""
    if PYCOLMAP4:
        cache_opts = pycolmap.DatabaseCacheOptions()
        cache_opts.min_num_matches = 0
        return pycolmap.DatabaseCache.create(database, cache_opts)
    else:
        return pycolmap.DatabaseCache().create(database, 0, False, set())


def get_image_Se3_world2cam(image: pycolmap.Image, device: str) -> Se3:
    image_world2cam: pycolmap.Rigid3d = image.cam_from_world()
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
    db1 = pycolmap.Database.open(str(colmap_db1_path))

    tmp_db1_path = merged_db_path.parent / 'tmp_db1.db'
    tmp_db2_path = merged_db_path.parent / 'tmp_db2.db'

    shutil.copy(colmap_db1_path, tmp_db1_path)
    shutil.copy(colmap_db2_path, tmp_db2_path)

    tmp_db1 = pycolmap.Database.open(str(tmp_db1_path))
    tmp_db2 = pycolmap.Database.open(str(tmp_db2_path))

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

    merged_db = pycolmap.Database.open(str(merged_db_path))
    pycolmap.Database.merge(tmp_db1, tmp_db2, merged_db)

    tmp_db1_path.unlink()
    tmp_db2_path.unlink()

    db1.close()
    merged_db.close()

    return db1_rename_dict, db2_rename_dict


def merge_colmap_reconstructions(
        rec1: pycolmap.Reconstruction, rec2: pycolmap.Reconstruction,
        rec1_id_map: Dict[int, int], rec2_id_map: Dict[int, int],
        rec1_name_map: Dict[str, str], rec2_name_map: Dict[str, str],
) -> pycolmap.Reconstruction:
    """Merge two COLMAP reconstructions, remapping image IDs and names to match a merged DB.

    Args:
        rec1, rec2: Source reconstructions.
        rec1_id_map, rec2_id_map: old_colmap_image_id → new_merged_image_id for each reconstruction.
        rec1_name_map, rec2_name_map: old_image_name → new_prefixed_image_name for each reconstruction.
    """
    merged = pycolmap.Reconstruction()
    next_camera_id = 0

    def _add_reconstruction(rec, id_map, name_map):
        nonlocal next_camera_id
        camera_id_remap = {}

        for old_cam_id, camera in rec.cameras.items():
            new_cam_id = next_camera_id
            next_camera_id += 1
            camera.camera_id = new_cam_id
            merged.add_camera(camera)
            camera_id_remap[old_cam_id] = new_cam_id

        for old_image_id, image in rec.images.items():
            new_image_id = id_map[old_image_id]
            new_name = name_map[image.name]
            new_camera_id = camera_id_remap[image.camera_id]

            clean_points2D = make_point2d_list()
            for point2D in image.points2D:
                clean_points2D.append(pycolmap.Point2D(xy=point2D.xy))

            add_posed_image_to_reconstruction(
                merged, new_image_id, new_camera_id, new_name,
                image.cam_from_world(), points2D=clean_points2D)

        for point3D in rec.points3D.values():
            new_track = pycolmap.Track()
            for elem in point3D.track.elements:
                new_track.add_element(id_map[elem.image_id], elem.point2D_idx)
            merged.add_point3D(point3D.xyz, new_track, point3D.color)

    _add_reconstruction(rec1, rec1_id_map, rec1_name_map)
    _add_reconstruction(rec2, rec2_id_map, rec2_name_map)

    return merged


def colmap_K_params_vec(camera_K, camera_type=pycolmap.CameraModelId.PINHOLE):
    if camera_type == pycolmap.CameraModelId.PINHOLE:
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])
        params_vec = [f_x, f_y, c_x, c_y]
    elif camera_type == pycolmap.CameraModelId.SIMPLE_PINHOLE:
        f_x = float(camera_K[0, 0])
        f_y = float(camera_K[1, 1])
        c_x = float(camera_K[0, 2])
        c_y = float(camera_K[1, 2])
        params_vec = [(f_x + f_y) / 2., c_x, c_y]
    else:
        raise ValueError(f'Unknown camera model {camera_type}')

    return params_vec

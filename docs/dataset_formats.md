# Dataset Formats & Comparison Methods

Internal reference for all dataset formats used in GloPose and the external methods used for comparison.

---

## Table of Contents

1. [BOP Standard Format](#1-bop-standard-format)
2. [HANDAL](#2-handal)
3. [HO3D](#3-ho3d)
4. [NAVI](#4-navi)
5. [BEHAVE](#5-behave)
6. [BOP Classic (T-LESS, LM-O, IC-BIN)](#6-bop-classic-t-less-lm-o-ic-bin)
7. [HOPE](#7-hope)
8. [Google Scanned Objects](#8-google-scanned-objects)
9. [TUM RGB-D](#9-tum-rgb-d)
10. [Synthetic Objects](#10-synthetic-objects)
11. [CNOS Detection Pipeline](#11-cnos-detection-pipeline)
12. [Mast3r / Dust3r](#12-mast3r--dust3r)
13. [VGGT](#13-vggt)
14. [MapAnything](#14-mapanything)

---

## 1. BOP Standard Format

Datasets following the [BOP benchmark](https://bop.felk.cvut.cz/datasets/) convention share a common
folder layout and annotation schema. HANDAL, HOPE, T-LESS, LM-O, and IC-BIN all use this format
(with dataset-specific extensions).

### 1.1 Folder Layout

```
<bop_root>/<dataset>/
    <split>/                            # train, train_primesense, train_pbr, test, val, ...
        <scene_id>/                     # 6-digit zero-padded: 000001, 000002, ...
            rgb/
                000000.png              # or .jpg; 6-digit zero-padded frame ID
                000001.png
                ...
            depth/                      # 16-bit PNG depth maps
                000000.png
                ...
            mask/                       # full projected object masks (optional)
                000000_000000.png       # {frame_id}_{object_instance_index}.png
            mask_visib/                 # visible-part masks
                000000_000000.png
            scene_gt.json               # GT object poses per frame
            scene_camera.json           # camera intrinsics (+ optional extrinsics) per frame
            scene_gt_info.json          # bboxes, visibility fractions
    models/
        obj_000001.ply                  # one PLY mesh per object, units: mm
        ...
        models_info.json                # diameters, bounding box dimensions
    test_targets_bop19.json             # or val_targets_bop24.json, test_targets_bop24.json
```

For HOPE/HANDAL, additional onboarding directories:

```
    onboarding_static/
        <obj_id>_up/                    # turntable, camera above
        <obj_id>_down/                  # turntable, camera below
    onboarding_dynamic/
        <obj_id>/                       # hand-held, free motion
```

### 1.2 `scene_gt.json` -- Object Poses

```json
{
  "0": [
    {
      "obj_id": 1,
      "cam_R_m2c": [r00, r01, r02, r10, r11, r12, r20, r21, r22],
      "cam_t_m2c": [tx, ty, tz]
    }
  ]
}
```

| Field | Format | Description |
|-------|--------|-------------|
| Key | String frame ID (`"0"`, `"1"`, ...) | Matches image filename stem |
| `obj_id` | int | Object identifier (dataset-local) |
| `cam_R_m2c` | 9 floats, row-major 3x3 | Rotation: **model-to-camera** |
| `cam_t_m2c` | 3 floats | Translation: **model-to-camera**, in **mm** |

Multiple objects per frame are stored as a list. GloPose loads this in
`utils/bop_challenge.py:read_obj2cam_Se3_from_gt()`.

### 1.3 `scene_camera.json` -- Camera Parameters

```json
{
  "0": {
    "cam_K": [fx, 0, cx, 0, fy, cy, 0, 0, 1],
    "depth_scale": 0.1,
    "cam_R_w2c": [r00, ..., r22],
    "cam_t_w2c": [tx, ty, tz],
    "width": 640,
    "height": 480
  }
}
```

| Field | Format | Description |
|-------|--------|-------------|
| `cam_K` | 9 floats, row-major 3x3 | Pinhole intrinsic matrix |
| `depth_scale` | float | Multiplier: raw depth pixel value x depth_scale = mm |
| `cam_R_w2c` | 9 floats (optional) | World-to-camera rotation (onboarding sequences only) |
| `cam_t_w2c` | 3 floats (optional) | World-to-camera translation in mm (onboarding sequences only) |
| `width`, `height` | int (optional) | Image dimensions |

`width`/`height` may be absent (GloPose falls back to 0). `cam_R_w2c`/`cam_t_w2c` are only present
in onboarding sequences, not in standard BOP test/val splits.

GloPose loads intrinsics in `utils/bop_challenge.py:get_pinhole_params()` and world-to-camera
extrinsics in `read_gt_Se3_world2cam()`.

### 1.4 `scene_gt_info.json` -- Visibility & Bounding Boxes

```json
{
  "0": [
    {
      "bbox_obj": [x, y, w, h],
      "bbox_visib": [x, y, w, h],
      "px_count_all": 1234,
      "px_count_valid": 1200,
      "px_count_visib": 1100,
      "visib_fract": 0.89
    }
  ]
}
```

GloPose does **not** read this file directly. It is used by the CNOS dataloader
(`repositories/cnos/src/dataloader/bop_pbr.py`).

### 1.5 Test Targets Files

```json
[
  {"im_id": 0, "scene_id": 1, "obj_id": 5, "inst_count": 1},
  ...
]
```

`obj_id` and `inst_count` are optional. GloPose groups by `(im_id, scene_id)` in
`utils/bop_challenge.py:group_test_targets_by_image()`.

### 1.6 Object Models

PLY meshes at `models/obj_NNNNNN.ply`:
- Binary little-endian PLY with vertices (float xyz + uchar rgba) and triangle faces
- Units: **millimeters**
- `models_info.json` provides per-object `diameter`, `min_x/y/z`, `size_x/y/z` (all in mm)

### 1.7 Depth Maps

- 16-bit unsigned int PNG
- Raw value x `depth_scale` (from `scene_camera.json`) = depth in **mm**
- GloPose applies an additional `depth_scale_to_meter` conversion (see per-dataset sections)

### 1.8 Image Naming

- RGB: `{frame_id:06d}.png` or `.jpg` (GloPose checks both; see `pose_estimator.py:312`)
- Masks: `{frame_id:06d}_{object_instance_index:06d}.png`
- Depth: `{frame_id:06d}.png`
- Frame IDs may be non-contiguous (especially in HANDAL static sequences)

### 1.9 Coordinate System Conventions

- **Object coordinates**: defined by PLY model, origin at model center, units in mm
- **Camera coordinates**: standard CV convention (x-right, y-down, z-forward)
- **`cam_R_m2c`, `cam_t_m2c`**: transform points from object frame to camera frame
- **`cam_R_w2c`, `cam_t_w2c`**: transform points from world frame to camera frame

GloPose conversion chain (in `utils/bop_challenge.py`):
1. `read_obj2cam_Se3_from_gt()` -> `Se3(R_m2c, t_m2c)` (object-to-camera)
2. `extract_gt_Se3_cam2obj()` -> invert + scale -> camera-to-object
3. Run scripts invert again -> `gt_Se3_world2cam` (since "world" = object frame in onboarding)

---

## 2. HANDAL

40 hand-held objects across 17 categories. Available in both native and BOP format.

### 2.1 Native Format

```
HANDAL/
    handal_dataset_<category>/          # 17 categories (mugs, hammers, spatulas, ...)
        models/
            models_info.json
            obj_000001.ply              # per-object PLY, units: mm
            ...
        models_parts/                   # HANDAL-specific
            obj_000001_handle.ply
            obj_000001_not.ply
        train/
            <sequence_id>/              # e.g., 001001 (first 3 digits = obj_id)
                rgb/                    # {frame_id:06d}.jpg
                mask/                   # {frame_id:06d}_{obj_idx:06d}.png
                mask_visib/
                mask_parts/             # HANDAL-specific: {fid}_{oid}_handle.png
                scene_gt.json
                scene_camera.json
                scene_gt_info.json
        test/
            ...
        dynamic/                        # present in ~10 of 17 categories
            002999_train/               # obj_id 002, "999" = dynamic marker
            ...
```

**No `depth/` directory** in native HANDAL.

### 2.2 BOP Format

```
bop/handal/
    onboarding_static/
        obj_NNNNNN_up/
        obj_NNNNNN_down/
    onboarding_dynamic/
        obj_NNNNNN/
    val/
        <scene_id>/
    test/
        <scene_id>/
    models/
        obj_000001.ply ... obj_000040.ply
    val_targets_bop24.json
    test_targets_bop24.json
```

### 2.3 Sequence Naming

| Format | Example | Meaning |
|--------|---------|---------|
| Native | `handal_dataset_mugs@001001` | Category `mugs`, obj 001, sequence 001 |
| BOP onboarding | `obj_000001_up` | Object 1, static upper turntable |
| BOP onboarding | `obj_000001_dynamic` | Object 1, hand-held dynamic |
| BOP val | `000001_000005` | Scene 1, object 5 |

### 2.4 Key Parameters

| Property | Static sequences | Dynamic sequences |
|----------|-----------------|-------------------|
| Resolution | 1920x1440 | 640x480 |
| Typical fx, fy | ~1590, ~1589 | ~567, ~567 |
| Frame IDs | Non-contiguous (0, 8, 15, 22, ...) | Consecutive (0, 1, 2, ...) |
| Frames/sequence | ~124-133 | ~400-522 |
| `depth_scale_to_meter` | 0.001 | 0.001 |
| `image_downsample` | 0.5 | 0.5 |
| `similarity_transformation` | `'kabsch'` | `'depths'` |

### 2.5 Object ID Scope

Object IDs are **not globally unique** across native categories (each category starts at 1).
In BOP format, objects are globally numbered `obj_000001` through `obj_000040`.

### 2.6 GloPose Loading

- **Native**: `run_HANDAL.py` -- splits on `@`, sets `cam_scale=1.0`, `image_downsample=0.5`
- **BOP onboarding**: `run_bop_HANDAL_onboarding.py` -> `set_config_for_bop_onboarding()` ->
  `run_on_bop_sequences()`
- **BOP val**: `run_bop_HANDAL.py` -> `get_bop_val_sequences()` -> `run_on_bop_sequences()`

---

## 3. HO3D

Hand-Object 3D dataset v3. 9 YCB objects manipulated by hands, captured with multi-camera Kinect rig.

### 3.1 Folder Layout

```
HO3D/
    train/                              # 55 sequences
        <sequence_name>/                # e.g., ABF10, MC1, SM2
            rgb/                        # {frame_id:04d}.jpg (4-digit, 640x480)
            depth/                      # {frame_id:04d}.png (16-bit, encoded)
            meta/                       # {frame_id:04d}.pkl and .npz
            seg/                        # {frame_id:04d}.png (320x240, half-res)
    evaluation/                         # 13 sequences
        <sequence_name>/
            rgb/ depth/ meta/
            segmentation_rendered/      # GloPose-generated (no seg/ in eval split)
    models/                             # YCB object meshes
        003_cracker_box/
            textured.obj                # full mesh (~51MB)
            textured_simple.obj         # simplified mesh (~1.6MB)
            texture_map.png
            points.xyz                  # point cloud
        ...
    calibration/                        # multi-camera extrinsics (v3)
```

### 3.2 Annotation Format (`.pkl` meta files)

Each frame has a pickle dict with:

| Field | Shape | Description |
|-------|-------|-------------|
| `camMat` | `(3, 3)` float64 | Camera intrinsic matrix |
| `objRot` | `(3, 1)` or `(3,)` float32 | Rodrigues axis-angle rotation (obj-to-cam) |
| `objTrans` | `(3,)` float32 | Translation in **meters** (obj-to-cam) |
| `objName` | str | YCB object name, e.g., `'021_bleach_cleanser'` |
| `objLabel` | int | YCB numeric label |
| `objCorners3D` | `(8, 3)` float64 | 3D bbox in current camera frame (meters) |
| `objCorners3DRest` | `(8, 3)` float64 | 3D bbox in object rest pose (meters) |
| `handPose` | `(48,)` float32 | MANO hand pose (train only) |
| `handTrans` | `(3,)` float32 | Hand translation (train only) |
| `handBeta` | `(10,)` float32 | MANO shape params (train only) |
| `handJoints3D` | `(21, 3)` float64 | 3D hand joints (train); `(3,)` (eval) |

Evaluation split omits `handPose`, `handTrans`, `handBeta` and related contact fields.
`objRot` shape is inconsistent across sequences -- GloPose uses `.squeeze()`.

### 3.3 Key Differences from BOP

| Aspect | HO3D | BOP standard |
|--------|------|--------------|
| **Annotations** | Per-frame `.pkl` files | `scene_gt.json` + `scene_camera.json` |
| **Rotation format** | Rodrigues axis-angle (3D) | 3x3 rotation matrix (flat 9 elements) |
| **Translation units** | **Meters** | **Millimeters** |
| **Image naming** | 4-digit zero-padded `.jpg` | 6-digit zero-padded `.png`/`.jpg` |
| **Image resolution** | 640x480 | Varies |
| **Model units** | **Meters** (YCB) | **Millimeters** |
| **Depth encoding** | 2-channel: `(ch2 + ch1*256) * 0.00012498664727900177` | 16-bit uint * depth_scale |

### 3.4 GloPose Loading

`run_HO3D.py`:
- Iterates `.pkl` files in `meta/`, loads `camMat`, `objRot`, `objTrans`
- Converts `objRot` to quaternion via `Quaternion.from_axis_angle()`
- **Multiplies translations by 1000** (meters -> mm, to match GloPose's internal mm convention)
- `skip_indices *= 10` (sequences are ~1000-1700 frames)
- Segmentation from `seg/` folder (channel 1 = green), falls back to `segmentation_rendered/`

### 3.5 Objects and Sequences

9 YCB objects appear: `003_cracker_box`, `004_sugar_box`, `006_mustard_bottle`,
`010_potted_meat_can`, `011_banana`, `019_pitcher_base`, `021_bleach_cleanser`, `025_mug`,
`035_power_drill`, `037_scissors`.

Sequence naming: `{subject_prefix}{camera_id}` (e.g., `ABF10` = subject ABF, camera 0).
55 train sequences, 13 evaluation sequences.

---

## 4. NAVI

Novel View synthesis and Appearance capture for object Instance recognition. 36 household objects,
video sequences with per-frame poses from COLMAP.

### 4.1 Folder Layout

```
NAVI/navi_v1.5/
    <object_name>/                      # 36 objects
        3d_scan/
            <object_name>.obj           # GT mesh (OBJ, large ~50-200MB)
            <object_name>.mtl
            <object_name>.jpg           # texture
            <object_name>.glb           # GLB format
        video-NN-<camera>-<video_id>/   # 136 video sequences total
            annotations.json
            images/                     # frame_NNNNN.jpg (5-digit, non-contiguous)
            masks/                      # frame_NNNNN.png (binary, palette mode)
            depth/                      # frame_NNNNN.png (uint16, mm)
            video.mp4
        multiview-NN-<camera>/          # 324 multiview sequences (not used by GloPose)
        wild_set/                       # 35 wild sets (not used by GloPose)
    custom_splits/
```

### 4.2 `annotations.json`

JSON array, one entry per frame:

```json
{
  "object_id": "3d_dollhouse_sink",
  "camera": {
    "q": [w, x, y, z],
    "t": [tx, ty, tz],
    "focal_length": 2654.89,
    "camera_model": "canon_t4i"
  },
  "filename": "frame_00000.jpg",
  "image_size": [1080, 1920],
  "split": "train",
  "occluded": false
}
```

| Field | Description |
|-------|-------------|
| `camera.q` | Quaternion `[w, x, y, z]` (Hamilton), **world-to-camera** rotation |
| `camera.t` | Translation `[tx, ty, tz]`, **world-to-camera**, in **mm** |
| `camera.focal_length` | Single focal length in pixels (fx = fy) |
| `image_size` | `[height, width]` |
| `occluded` | Per-frame occlusion flag (~7.6% of frames) |

Principal point is assumed at image center: `cx = width/2`, `cy = height/2`.

### 4.3 Key Parameters

| Property | Value |
|----------|-------|
| Resolution | 1080x1920 (portrait) or 1920x1080 (landscape) |
| Focal length range | 1577-3434 px (varies by camera + zoom) |
| Depth format | uint16 PNG, values in mm, zero = no depth |
| Model format | OBJ + texture, scan-quality |
| Translation units | mm |
| Frames per sequence | ~40-200 |
| Frame naming | `frame_NNNNN.jpg` (non-contiguous, sampled at ~15 frame intervals) |

### 4.4 GloPose Loading

`run_NAVI.py`:
- Sequence format: `<object>@video-NN-<camera>-<video_id>`
- Only `video-*` sequences are discovered (not multiview or wild_set)
- Quaternion loaded directly into `kornia.geometry.Quaternion`
- Frame dicts reindexed to contiguous 0-based indices
- No `depth_scale_to_meter` conversion (already in mm)

---

## 5. BEHAVE

Human-object interaction dataset. 18 object categories, video-based, multi-camera Kinect capture.

### 5.1 Folder Layout

```
BEHAVE/
    train/                              # 835 sequences
        hash_codes.txt
        <hash_id>.mp4                   # RGB video (2048x1536 @ 30fps)
        <hash_id>_mask_obj.mp4          # object mask video
        <hash_id>_mask_hum.mp4          # human mask video
        <hash_id>_gt.pkl                # GT poses
        <hash_id>_metadata.pkl          # sequence metadata
    val/                                # 21 sequences (with GT)
    test/                               # 85 sequences (no GT)
```

Hash IDs are 26-character alphanumeric strings.

### 5.2 Annotation Format (`_gt.pkl`)

| Field | Shape | Description |
|-------|-------|-------------|
| `obj_rot` | `(N, 3, 3)` float64 | **Camera-to-object** rotation matrices |
| `obj_trans` | `(N, 3)` float64 | **Camera-to-object** translation in **meters** |
| `smplh_poses` | `(N, 156)` float32 | SMPL-H body pose (not used by GloPose) |
| `smplh_betas` | `(N, 10)` float32 | SMPL-H shape (not used) |
| `smplh_trans` | `(N, 3)` float64 | SMPL-H translation (not used) |
| `gender` | str | `"male"` / `"female"` |
| `meta` | tuple | `(dataset, gender, obj_name)` |

`_metadata.pkl` contains: `gender`, `obj_name`, `dataset`.

### 5.3 Key Differences

| Aspect | BEHAVE | BOP/NAVI |
|--------|--------|----------|
| **Data format** | MP4 videos (not individual images) | Individual image files |
| **Pose semantics** | Camera-to-object (inverted for GloPose) | Object-to-camera or world-to-camera |
| **Rotation format** | 3x3 matrix | Flat 9 elements (BOP) or quaternion (NAVI) |
| **Translation units** | Meters | Millimeters |
| **Camera intrinsics** | **Not provided** (COLMAP estimates) | Per-frame JSON |
| **3D models** | **Not provided** in GloPose data dir | PLY (BOP) or OBJ (NAVI) |
| **Depth** | **Not provided** | Available (BOP, NAVI) |

### 5.4 GloPose Loading

`run_BEHAVE.py`:
- Only first sequence processed (has `exit()` after one sequence)
- Frame access via OpenCV `VideoCapture` + `get_nth_video_frame()`
- `skip_indices *= 10` (30fps video, long sequences)
- `Se3_cam2obj` inverted to get `world2cam`
- No intrinsics passed -> COLMAP estimates them
- 18 objects: backpack, boxlarge, boxlong, boxmedium, boxsmall, boxtiny, chairblack, chairwood,
  monitor, plasticcontainer, stool, suitcase, tablesmall, tablesquare, toolbox, trashbin,
  yogaball, yogamat

---

## 6. BOP Classic (T-LESS, LM-O, IC-BIN)

These follow the standard BOP format (section 1) with dataset-specific quirks.

### 6.1 T-LESS (Texture-Less Objects)

```
bop/tless/
    train_primesense/                   # real sensor data for onboarding
    test_primesense/                    # test split
    models/
        obj_000001.ply ... obj_000030.ply   # 30 texture-less industrial parts
```

| Property | Value |
|----------|-------|
| Objects | 30 (texture-less, many with symmetries) |
| Splits | `train_primesense` (onboarding), `test_primesense` (eval) |
| Targets year | `bop19` |
| Templates per object | 491 (fewer than 642, due to symmetries) |
| `depth_scale_to_meter` | 0.001 |
| `skip_indices` | multiplied by 4 |
| Inference downsampling | 1.0 (none) |

### 6.2 LM-O (Linemod Occlusion)

```
bop/lmo/
    train/                              # onboarding
    test/                               # evaluation
    models/
        obj_000001.ply, obj_000005.ply, obj_000006.ply,
        obj_000008.ply ... obj_000012.ply    # 8 objects (non-contiguous IDs)
```

| Property | Value |
|----------|-------|
| Objects | 8 (IDs: 1, 5, 6, 8, 9, 10, 11, 12) |
| Splits | `train`, `test` |
| Targets year | `bop19` |
| Templates per object | 642 |
| `depth_scale_to_meter` | 0.001 |

### 6.3 IC-BIN (Bin-Picking)

```
bop/icbin/
    train/
    test/
    models/
        obj_000001.ply, obj_000002.ply  # 2 objects only
```

| Property | Value |
|----------|-------|
| Objects | 2 |
| Splits | `train`, `test` |
| Targets year | `bop19` |
| Heavy clutter/occlusion | Yes (bin-picking scenario) |

### 6.4 GloPose Loading

`run_BOP_classic_onboarding.py`:
- Sequence code: `{dataset}@{split}@{scene_name}` (e.g., `tless@train_primesense@000001`)
- `depth_scale_to_meter = 0.001`, `skip_indices *= 4`
- Loads via `get_bop_images_and_segmentations()`, `read_gt_Se3_cam2obj_transformations()`,
  `read_pinhole_params()`

---

## 7. HOPE

28 household objects, BOP format with onboarding splits.

```
bop/hope/
    onboarding_static/
        obj_NNNNNN_up/
        obj_NNNNNN_down/
    onboarding_dynamic/
        obj_NNNNNN/
    test/
    val/
    models/
        obj_000001.ply ... obj_000028.ply
    test_targets_bop24.json
```

| Property | Value |
|----------|-------|
| Objects | 28 |
| Targets year | `bop24` |
| `image_downsample` | 0.5 |
| `depth_scale_to_meter` | 0.001 |
| Inference downsampling | 0.25 |
| Onboarding types | Static (up/down/both) + dynamic |

GloPose loading: `run_HOPE.py` -> `set_config_for_bop_onboarding()` -> `run_on_bop_sequences()`.
Same static/dynamic config switching as HANDAL.

---

## 8. Google Scanned Objects

1000+ high-quality 3D scans. Used for **synthetic rendering** only (no real images).

### 8.1 Folder Layout

```
GoogleScannedObjects/
    models/
        <object_name>/                  # e.g., Squirrel, SCHOOL_BUS
            meshes/
                model.obj               # OBJ mesh
                model.mtl
            materials/textures/
                texture.png
            metadata.pbtxt
            model.config
```

### 8.2 GloPose Loading

`run_GoogleScannedObjects.py`:
- Synthetic pipeline: loads mesh via Kaolin, renders images from random viewpoints on a sphere
- Camera at `(0, -5.0, 0)`, up vector `(0, 0, 1)` (Z-up convention)
- Rotation generator: `scenarios.random_walk_on_a_sphere`
- Not a real-image dataset -- used for controlled ablation

---

## 9. TUM RGB-D

SLAM benchmark. Real indoor sequences with ground truth trajectories from motion capture.

### 9.1 Folder Layout

```
tum_rgbd/
    rgbd_dataset_freiburg1_desk/
        rgb/
            <timestamp>.png             # e.g., 1305031452.791720.png
        depth/
            <timestamp>.png
        rgb.txt                         # timestamp  filename
        depth.txt                       # timestamp  filename
        groundtruth.txt                 # timestamp  tx ty tz qx qy qz qw
```

### 9.2 Ground Truth Format

`groundtruth.txt`: space-delimited, `#` comments.

```
# timestamp tx ty tz qx qy qz qw
1305031449.7996 1.2334 -0.0113 1.6941 0.7907 0.4393 -0.1770 -0.3879
```

Quaternion order: `qx, qy, qz, qw` in file, reordered to `qw, qx, qy, qz` for kornia.
Translation in meters. World-to-camera. No object models (SLAM benchmark, not object pose).

### 9.3 GloPose Loading

`run_TUM_RGBD.py`:
- `segmentation_provider = 'whites'` (no object masks, uses white regions)
- Processes only one sequence (has `exit()` at line 90)
- First frame used as reference for "object" frame

---

## 10. Synthetic Objects

Local mesh prototypes for testing.

```
prototypes/
    sphere.obj + tex.png
    textured-cube/textured-cube.obj
    8-colored-sphere/8-colored-sphere.obj
```

`run_SyntheticObjects.py` renders images using Kaolin from predefined rotation sequences
(`generate_rotations_y`, etc.), then runs onboarding on the synthetic frames.

---

## 11. CNOS Detection Pipeline

Original CNOS (CAD-based Novel Object Segmentation) by Van Nguyen et al.
Repository: [github.com/nv-nguyen/cnos](https://github.com/nv-nguyen/cnos).
Our fork with modifications lives at `repositories/cnos/`.

### 11.1 Overview

CNOS is a **zero-shot** object detection/segmentation pipeline that requires no retraining for
novel objects. Given CAD models (or onboarding images) of target objects, it detects them in
novel query images using a three-stage approach:

```
Stage 1: Template rendering     (offline, once per object)
Stage 2: Proposal generation    (per query image)
Stage 3: Descriptor matching    (per query image)
```

Entry point: `run_inference.py` (Hydra-configured PyTorch Lightning `trainer.test()`).
The `CNOS` class (`src/model/detector.py`) implements the Lightning module.

### 11.2 Stage 1: Template Rendering (Offline)

Templates are pre-rendered views of each object from uniformly distributed viewpoints on the
upper hemisphere.

**Viewpoint generation** (`src/poses/create_template_poses.py`):
- Blender icosphere subdivision for uniform viewpoint sampling
- Three detail levels: level 0 (coarse), level 1 (medium), level 2 (dense)
- Each viewpoint converted to a 4x4 camera-to-world matrix via `look_at` toward origin
- Saved as `.npy` files in `src/poses/predefined_poses/`

**Rendering** (`src/poses/pyrender.py`):
- Pyrender with OpenGL conventions; camera at identity, object rotated to each viewpoint
- Default intrinsics: `fx=572.41, fy=573.57, cx=325.26, cy=242.05` (480x640)
- Mesh loaded via trimesh, re-centered at bbox centroid
- Output: RGBA PNG per viewpoint, `{idx:06d}.png`
- Objects in mm auto-scaled to meters (diameter > 100 check)

**Rendering types** (from config):
- `pyrender`: standard pyrender (default for CAD-based)
- `pbr`: BlenderProc physically-based rendering (higher quality)
- `onboarding_static` / `onboarding_dynamic`: model-free, uses actual captured images

**Descriptor pre-computation** (`set_reference_objects` in `detector.py`):
- Each template: alpha-channel mask -> zero background -> crop to bbox -> resize to 224x224
  (via `CropResizePad`: resize longest side, pad shorter side) -> ImageNet normalize
- DINOv2 CLS-token extraction -> cached to `descriptors.pth`
- Final shape: `[N_objects, N_templates_per_object, descriptor_dim]`

### 11.3 Stage 2: Proposal Generation (SAM)

For each query image, a class-agnostic segmentation model generates mask proposals:

**SAM** (`src/model/sam.py`):
- Image optionally resized to `segmentor_width_size=640` before mask generation
- `SamAutomaticMaskGenerator` with:
  - `stability_score_thresh=0.97` (only high-confidence masks)
  - `box_nms_thresh=0.7` (SAM's internal proposal NMS)
  - Multi-crop support with `crop_nms_thresh=0.7`
- Masks interpolated back to original resolution

**SAM2** (`src/model/sam2.py`):
- `SAM2AutomaticMaskGenerator` wrapper, same interface

**FastSAM** (`src/model/fastsam.py`):
- YOLO-based segmentation: `conf=0.25`, `iou=0.9`, `max_det=200`

**Post-filtering** (in `Detections.remove_very_small_detections`):
- Remove proposals with box area < `min_box_size^2` = 0.05^2 (0.25% of image area)
- Remove proposals with mask area < `min_mask_size` = 3e-4 (0.03% of image area)

### 11.4 Stage 3: Descriptor Matching

**Query descriptor extraction** (`src/model/dinov2.py`, class `CustomDINOv2`):

Model: `dinov2_vitl14` (ViT-Large, patch size 14, 1024-dim CLS token) via
`torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")`.
Variants: `vits14` (384-d), `vitb14` (768-d), `vitl14` (1024-d), `vitg14` (1536-d).

Per-proposal processing:
1. **Mask out background**: `masked_rgb = rgb * mask` (pixels outside mask become black/zero)
2. **Crop** to bounding box
3. **Resize** to 224x224 via `CropResizePad` (resize longest side, pad to square)
4. **Normalize**: ImageNet `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`
5. **Forward pass** (chunked, `chunk_size=16`): `model.forward_features()` -> `x_norm_clstoken`

Background masking is critical: it ensures the descriptor captures only object appearance.

**Similarity computation** (`find_matched_proposals` in `detector.py`, `PairwiseSimilarity` in
`loss.py`):
- **Cosine similarity** only: `F.cosine_similarity(queries, references, dim=-1)`, clamped to [0, 1]
- Result: `[N_proposals, N_objects, N_templates]`

**Aggregation across templates** (per proposal, per object):
- `mean`: average across all templates
- `median`: median template score
- `max`: best-matching template
- **`avg_5`** (default): average of top-5 most similar templates

**Object assignment & filtering**:
- Each proposal assigned to object with highest aggregated score (`torch.max`)
- Proposals below `confidence_thresh=0.15` discarded
- Top-100 by score kept (`max_num_instances=100`, BOP challenge limit)

### 11.5 NMS

Per-object-class NMS (`Detections.apply_nms_per_object_id` in `utils.py`):
- For each unique `object_id`, run `torchvision.ops.nms` on that class's boxes
- IoU threshold: **0.25**
- Overlapping detections of **different** objects are preserved
- Overlapping detections of the **same** object are suppressed

### 11.6 Output Format

Per-image NPZ:
```python
{
    "scene_id": int,
    "image_id": int,
    "category_id": object_ids + 1,  # 1-indexed (LMO uses special mapping)
    "score": float[],
    "bbox": [x, y, w, h][],         # COCO format
    "time": float,
    "segmentation": masks[],         # optional
}
```

Final JSON (BOP23 format, converted from NPZ via multiprocessing):
```json
[
  {
    "scene_id": 1,
    "image_id": 42,
    "category_id": 5,
    "bbox": [100, 200, 50, 60],
    "score": 0.87,
    "time": 0.34,
    "segmentation": {"counts": "...", "size": [480, 640]}
  }
]
```

Directly submittable to BOP evaluation server (Tasks 5 & 6).

### 11.7 Default Parameters

| Parameter | Default | Description |
|---|---|---|
| Segmentor | SAM | Class-agnostic proposal generator |
| `segmentor_width_size` | 640 | Input width for SAM |
| Descriptor model | `dinov2_vitl14` | 1024-d CLS token |
| Proposal crop size | 224 | Input size for DINOv2 |
| `rendering_type` | `pbr` | Template rendering method |
| `level_templates` | 0 | Viewpoint density (coarse) |
| `aggregation_function` | `avg_5` | Top-5 average |
| `confidence_thresh` | 0.15 | Min cosine similarity |
| `max_num_instances` | 100 | Max detections per image |
| `nms_thresh` | 0.25 | Per-class NMS IoU threshold |
| `min_box_size` | 0.05 | Min box side / image side |
| `min_mask_size` | 3e-4 | Min mask area / image area |
| SAM `stability_score_thresh` | 0.97 | Mask confidence gate |
| SAM `box_nms_thresh` | 0.7 | SAM-internal NMS |
| `chunk_size` | 16 | DINOv2 batch size |

### 11.8 GloPose Modifications vs Original

GloPose extends the original CNOS in several ways (implemented in our fork at `repositories/cnos/`
and in `pose/pose_estimator.py`, `condensate_templates.py`):

| Feature | Original CNOS | GloPose Extension |
|---------|--------------|-------------------|
| Similarity metric | Cosine only | + CSLS (cross-domain similarity local scaling) |
| OOD filtering | Single `confidence_thresh` | + Lowe ratio test, cosine quantiles, Mahalanobis distance |
| Descriptor post-processing | None | PCA whitening + L2 re-normalization |
| Patch-level matching | Not used | Average patch-descriptor cosine similarity filter |
| Template condensation | Not present | Hart's CNN (original + symmetric), imblearn CNN |
| Template sources | Pre-rendered from CAD | + ViewGraph keyframes, condensed templates |
| Additional NMS | Per-class box NMS only | + Masks-inside-masks suppression (80% containment) |
| Descriptor output | CLS token only | CLS + patch tokens returned together |
| Template storage | Flat `[N_obj, N_templ, D]` tensor | `Dict[int, Tensor]` (variable templates per object) |

### 11.9 `Detections` Type

Defined in `src/model/utils.py`:
- `boxes`: `(N, 4)` tensor in xyxy format
- `masks`: list of binary masks or RLE-encoded masks
- `scores`: `(N,)` confidence scores
- `object_ids`: `(N,)` matched object IDs

### 11.10 BOP Dataset Support

Core BOP datasets: LM, LMO, T-LESS, ITODD, HB, HOPE, YCBV, RUAPC, ICBIN, ICMI, TUDL, TYOL.
BOP 2024 additions: HOT3D, HANDAL, HOPEv2.

Split conventions: most use `test`; HB and T-LESS use `test_primesense`.
Target files: `test_targets_bop19.json` (classic) or `test_targets_bop24.json` (2024).

---

## 12. Mast3r / Dust3r

Located at `repositories/mast3r/` (Dust3r is a git submodule at `repositories/mast3r/dust3r/`).

**Status**: Repository cloned, **no integration** into GloPose code. Adapter planned (CLAUDE.md P3.4).

### 12.1 What It Does

Mast3r (Matching And Stereo 3D Reconstruction) is a ViT-based model that predicts dense 3D point
maps and matching descriptors from image pairs. Multiple pairs are combined via global alignment
to produce a full multi-view reconstruction.

### 12.2 Two Reconstruction Modes

**Mode A: Sparse Global Alignment** (primary, `sparse_global_alignment()`):
1. Create image pairs (`make_pairs()` with scene graph: complete/swin/logwin/retrieval)
2. For each pair, run Mast3r forward pass -> per-pixel 3D points + descriptors + confidence
3. Extract keypoints and correspondences
4. Build minimum spanning tree from pairwise scores
5. Coarse optimization (lr=0.07, 300 iters): camera poses + focals via 3D matching loss
6. Fine refinement (lr=0.01, 300 iters): 2D reprojection error
7. Output: `SparseGA` scene object

**Mode B: GLOMAP/COLMAP Integration** (`colmap/mapping.py`):
1. Run Mast3r matching on pairs -> 2D-2D correspondences
2. Export to COLMAP database
3. Run GLOMAP or pycolmap mapper
4. Output: standard COLMAP reconstruction

### 12.3 API

**Model loading:**
```python
from mast3r.model import AsymmetricMASt3R
model = AsymmetricMASt3R.from_pretrained("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
```

**Input:**
- List of image file paths
- Images preprocessed internally to `[B, C, H, W]` normalized to [-1, 1]

**Output (SparseGA):**
```python
scene.cam2w        # (N, 4, 4) camera-to-world transforms
scene.intrinsics   # list of (3, 3) intrinsic matrices
scene.depthmaps    # per-image depth maps
scene.pts3d        # per-image 3D point arrays
scene.imgs         # RGB images as [H, W, 3] numpy in [0, 1]
scene.get_focals() # focal lengths
```

### 12.4 Key Files for Integration

- `mast3r/cloud_opt/sparse_ga.py` -- `sparse_global_alignment()` main entry point
- `mast3r/image_pairs.py` -- `make_pairs()` for generating image pair lists
- `dust3r/inference.py` -- `inference()` for pairwise forward pass
- `dust3r/utils/image.py` -- `load_images()` for preprocessing

---

## 13. VGGT

Located at `repositories/vggt/`.

**Status**: Repository cloned, **no integration** into GloPose code. Adapter planned (CLAUDE.md P3.4).

### 13.1 What It Does

VGGT (Visual Geometry Grounded Transformer, 1B params) is a feed-forward model that jointly predicts
camera poses, depth maps, 3D points, and point tracks from a set of input images in a single
forward pass. No iterative optimization.

### 13.2 Pipeline

1. Load images, center-pad to square, resize to 1024x1024 (load resolution)
2. Model internally processes at 518x518
3. DINOv2 backbone -> alternating frame attention + global attention blocks
4. Camera head: `pose_enc [B, S, 9]` = `[translation(3), quaternion(4), FoV(2)]`
5. Depth head: depth maps `[B, S, H, W, 1]` + confidence
6. Convert pose encoding to extrinsic `[S, 3, 4]` and intrinsic `[S, 3, 3]`
7. Unproject depth to world points
8. Optional bundle adjustment via VGGSfM tracker + pycolmap

### 13.3 API

**Model loading:**
```python
from vggt.models.vggt import VGGT
model = VGGT()
model.load_state_dict(torch.hub.load_state_dict_from_url(
    "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"))
```

**Image preprocessing:**
```python
from vggt.utils.load_fn import load_and_preprocess_images_square
images, original_coords = load_and_preprocess_images_square(image_paths, target_size=1024)
# images: [N, 3, H, W] tensor in [0, 1]
```

**Forward pass output (predictions dict):**

| Key | Shape | Description |
|-----|-------|-------------|
| `pose_enc` | `[B, S, 9]` | Camera pose encoding |
| `depth` | `[B, S, H, W, 1]` | Depth maps |
| `depth_conf` | `[B, S, H, W]` | Depth confidence |
| `world_points` | `[B, S, H, W, 3]` | 3D world coordinates per pixel |
| `world_points_conf` | `[B, S, H, W]` | World point confidence |

**Pose conversion:**
```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
# extrinsics: [B, S, 3, 4] camera-from-world (OpenCV convention)
# intrinsics: [B, S, 3, 3]
```

### 13.4 Key Files for Integration

- `vggt/models/vggt.py` -- `VGGT` model class
- `vggt/utils/load_fn.py` -- `load_and_preprocess_images_square()`
- `vggt/utils/pose_enc.py` -- `pose_encoding_to_extri_intri()`
- `vggt/utils/geometry.py` -- `unproject_depth_map_to_point_map()`
- `demo_colmap.py` -- full pipeline reference (images -> COLMAP reconstruction)

---

## 14. MapAnything

**Status**: **Not present** in the repository. Not cloned, no submodule, no code references
outside of CLAUDE.md TODO items. Needs to be added as a submodule or installed before any
integration work.

---

## Quick Reference: Translation Units & Pose Conventions

| Dataset | Translation units | Pose semantics | Rotation format | Model units |
|---------|------------------|----------------|-----------------|-------------|
| BOP (all) | mm | obj-to-cam (`cam_R_m2c`) | 3x3 flat | mm |
| HANDAL | mm | obj-to-cam | 3x3 flat | mm |
| HO3D | **meters** (x1000 in GloPose) | obj-to-cam | Rodrigues 3D | **meters** |
| NAVI | mm | world-to-cam | quaternion wxyz | - |
| BEHAVE | **meters** | cam-to-obj (inverted) | 3x3 matrix | - |
| TUM-RGBD | **meters** | world-to-cam | quaternion xyzw | - |
| HOPE | mm | obj-to-cam | 3x3 flat | mm |

## Quick Reference: GloPose `depth_scale_to_meter`

| Dataset | Value | Meaning |
|---------|-------|---------|
| HANDAL | 0.001 | raw mm -> meters |
| HOPE | 0.001 | raw mm -> meters |
| T-LESS, LM-O, IC-BIN | 0.001 | raw mm -> meters |
| HO3D | special | 2-channel decode * 0.00012498664727900177 |

## Quick Reference: Image Resolution & Naming

| Dataset | Resolution | Naming | Extension |
|---------|-----------|--------|-----------|
| HANDAL static | 1920x1440 | `{:06d}` | .jpg |
| HANDAL dynamic | 640x480 | `{:06d}` | .jpg |
| HO3D | 640x480 | `{:04d}` | .jpg |
| NAVI | 1080x1920 | `frame_{:05d}` | .jpg |
| BEHAVE | 2048x1536 | (video frames) | .mp4 |
| BOP classic | varies | `{:06d}` | .png |
| HOPE | varies | `{:06d}` | .png/.jpg |
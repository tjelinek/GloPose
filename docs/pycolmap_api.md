# PyCOLMAP API Documentation

Source: https://colmap.github.io/pycolmap/pycolmap.html (COLMAP 3.14.0.dev0)

## Core Classes and Enumerations

### Device
Specifies computational backend with members: `auto`, `cpu`, `cuda`. Constructors accept either integer values or string names.

### SensorType
Defines sensor categories: `INVALID`, `CAMERA`, `IMU`. Used to identify sensor types in the system.

### sensor_t and data_t
Fundamental identifier types. `sensor_t` combines a `SensorType` and numeric ID. `data_t` associates sensor identifiers with data IDs, supporting dictionary conversion via `todict()` and `mergedict()`.

## Geometric Utilities

### Rotation3d
Represents 3D rotations using quaternions in [x,y,z,w] format. Supports multiple initialization methods:
- Quaternion vectors
- 3x3 rotation matrices
- Axis-angle representations

Key methods include `angle()`, `angle_to()`, `inverse()`, `matrix()`, and `normalize()`.

### Rigid3d
Encodes rigid transformations combining rotation and translation. Supports:
- Direct construction from rotation and translation
- 3x4 matrix initialization
- Static `interpolate()` method for pose interpolation
- Methods: `inverse()`, `matrix()`, `adjoint()`, `tgt_origin_in_src()`

### Sim3d
Extends rigid transformations with uniform scaling. Properties include `scale`, `rotation`, and `translation`. Method `transform_camera_world()` applies similarity transforms to camera poses.

### AlignedBox3d
Axis-aligned bounding box defined by minimum and maximum points. Methods include `contains_point()`, `contains_bbox()`, and `diagonal()`.

## Pose and Coordinate Systems

### PosePrior
Represents pose constraints with optional position covariance. Supports WGS84 and Cartesian coordinate systems. Methods: `has_position()`, `has_gravity()`, `has_position_cov()`.

### GPSTransform
Converts between coordinate systems (ECEF, ellipsoid, ENU, UTM). Supports GRS80 and WGS84 ellipsoids.

## Camera Models

### CameraModelId
Enumeration of supported models: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV, FISHEYE variants, FOV, THIN_PRISM_FISHEYE, DIVISION, and others.

### Camera
Complete camera representation including:
- Intrinsic parameters and distortion coefficients
- Sensor dimensions (width, height)
- Sensor identifier
- Model-specific calibration

Key methods:
- `create_from_model_id()`, `create_from_model_name()` static constructors
- `img_from_cam()` projects 3D points to image plane
- `cam_from_img()` unprojects 2D points to camera frame
- `calibration_matrix()` computes K matrix
- `rescale()` adjusts for resolution changes
- `verify_params()`, `has_bogus_params()` validate parameters
- `focal_length_idxs()`, `principal_point_idxs()`, `extra_params_idxs()` identify parameter types

## Point and Track Data

### Point2D
2D image point with optional 3D correspondence: `xy` coordinates and `point3D_id`. Method `has_point3D()` checks linkage.

### Track and TrackElement
Tracks link 2D observations across images to 3D points. TrackElement has `image_id` and `point2D_idx`.

### Point3D
3D world point with `xyz` position, `color`, `normal`, `error`, and `track`. Methods for accessing track elements.

### Point3DMap
Dictionary-like container for Point3D objects indexed by ID.

## Image

### Image
Represents a registered image with:
- `image_id`, `camera_id`, `name`
- `cam_from_world` (Rigid3d pose)
- `points2D` (Point2DList of observations)
- `has_pose` flag

## RANSAC

### RANSACOptions
- `max_error` (4.0 pixels default)
- `confidence` (0.9999 default)
- `max_num_trials` (100000 default)
- `min_num_trials` (1000 default)
- `min_inlier_ratio` (0.01 default)

## Two-View Geometry

### TwoViewGeometryOptions
Controls two-view estimation including RANSAC settings, pose solver selection, and geometric verification thresholds.

Functions:
- `estimate_calibrated_two_view_geometry()`
- `estimate_two_view_geometry()`
- `estimate_two_view_geometry_pose()`

## Bundle Adjustment

### BundleAdjustmentOptions
High-level interface combining config, gauge, and backend settings.

### bundle_adjustment()
Top-level BA function: `pycolmap.bundle_adjustment(reconstruction, ba_options)`

### BundleAdjustmentConfig
Specifies which parameters to optimize (cameras, poses, points, rig calibrations, etc.).

### CeresBundleAdjustmentOptions
Fine-grained Ceres solver configuration including function and gradient tolerance, max iterations, minimizer type, and loss function selection.

## Reconstruction

### Reconstruction
Model container with:
- `images` dict (image_id -> Image)
- `cameras` dict (camera_id -> Camera)
- `points3D` dict (point3d_id -> Point3D)

Key methods:
- `find_image_with_name(name)` -> Image or None
- `delete_point3D(point3d_id)` removes a 3D point
- `write(path)` / `read(path)` serialization
- `summary()` text summary
- `transform(sim3d)` applies similarity transform
- `point3D(id)` access individual point

## Incremental Reconstruction

### IncrementalPipelineOptions
Controls sequential SfM with thresholds for triangulation, reprojection, and track constraints.
- `triangulation.ignore_two_view_tracks`
- `ba_local_num_images`
- `ba_global_images_freq`
- `triangulation.max_transitivity`
- `init_image_id1`, `init_image_id2`

### incremental_mapping()
`pycolmap.incremental_mapping(database_path, image_path, output_path, options=opts)` -> list of Reconstructions

## Feature Matching

### match_exhaustive()
`pycolmap.match_exhaustive(database_path, verification_options=opts)`

## Database

### Database
Persistent storage. Key methods:
- `write_camera()`, `write_image()`, `write_keypoints()`, `write_matches()`
- `read_all_images()`, `read_keypoints()`
- `close()`

## Pose Estimation

### estimate_absolute_pose()
PnP solver for 2D-3D correspondences.

### estimate_and_refine_absolute_pose()
Combined estimation and refinement.

## Alignment

- `estimate_rigid3d()`, `estimate_rigid3d_robust()`
- `estimate_sim3d()`, `estimate_sim3d_robust()`
- `align_reconstructions_via_reprojections()`
- `align_reconstructions_via_proj_centers()`

## Multi-View Stereo

### patch_match_stereo()
Computes depth maps from images.

### stereo_fusion()
Merges depth maps into point clouds.

### poisson_meshing() / delaunay_meshing()
Surface reconstruction from point clouds.

## Utility Functions

- `image_pair_to_pair_id()` / `pair_id_to_image_pair()`
- `set_random_seed()`
- `extract_features()`
- `triangulate_points()`
- `undistort_camera()`, `undistort_image()`, `undistort_images()`
- `infer_camera_from_image()`

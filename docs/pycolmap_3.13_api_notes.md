# pycolmap 4.0 API Changes (vs 3.x)

## Rig → Frame → Image hierarchy (NEW in 4.0)
Poses are stored on Frames, not Images. Every Image must belong to a Frame,
and every Frame must belong to a Rig. In GloPose, use the helper:
```python
from onboarding.colmap_utils import add_posed_image_to_reconstruction
add_posed_image_to_reconstruction(rec, image_id, camera_id, name, cam_from_world, points2D=None)
```
This creates the Rig (one per camera, reused), Frame, and Image, sets the pose,
and calls `register_frame()`.

## Database
- `Database.open(path)` (static method) — unchanged
- `Database()` creates abstract base — cannot write

## Image
- Constructor: `Image(name=, points2D=, camera_id=, image_id=)` — NO `cam_from_world` kwarg
- `image.cam_from_world()` → Rigid3d (method, not property)
- Setting pose: go through `Frame.set_cam_from_world(camera_id, rigid3d)` — the frame must be
  added to a Reconstruction first (needs `rig_ptr_`)
- `image.frame_id` — must be set before `add_image()`, must point to existing Frame
- No `registered` attribute — use `rec.register_frame(frame_id)` instead
- `image.has_pose` (bool, read-only)
- `image.points2D` (Point2DList, read-only)
- `image.data_id` → `data_t` (used for Frame.add_data_id)

## Camera
- Constructor kwargs still work: `Camera(camera_id=, model=, width=, height=, params=)`
- `camera.sensor_id` → `sensor_t` (used for Rig.add_ref_sensor)

## Rig (NEW)
- `Rig(rig_id=)` — create rig
- `rig.add_ref_sensor(camera.sensor_id)` — set reference sensor
- `rec.add_rig(rig)` — add to reconstruction

## Frame (NEW)
- `Frame(frame_id=, rig_id=)` — create frame
- `frame.add_data_id(image.data_id)` — link frame to image
- `frame.set_cam_from_world(camera_id, rigid3d)` — set pose (must be done AFTER `rec.add_frame()`)
- `rec.add_frame(frame)` — add to reconstruction
- `rec.register_frame(frame_id)` — register so images persist on write/read

## Rigid3d
- Constructor: `Rigid3d(rotation, translation)` or `Rigid3d(matrix)` or `Rigid3d()`
- `.rotation` (Rotation3d property)
- `.translation` (ndarray property)
- `.inverse()` → Rigid3d
- `.matrix()` → ndarray[3,4] (always a method, never a property)

## Sim3d
- Constructor: `Sim3d(scale, rotation, translation)` or `Sim3d(matrix)` or `Sim3d()`
- `.scale`, `.rotation`, `.translation` (properties)
- `.transform_camera_world(cam_from_world: Rigid3d)` → Rigid3d
- `.inverse()` → Sim3d

## Point2DList (renamed)
- `pycolmap.ListPoint2D` removed → use `pycolmap.Point2DList`

## DatabaseCache
- Old: `DatabaseCache().create(database, min_num_matches, ignore_watermarks, image_names_set)`
- New: `DatabaseCache.create(database, DatabaseCacheOptions())` (static method)

## IncrementalPipelineOptions
- `ba_local_num_images` moved to `opts.mapper.ba_local_num_images`
- `ba_global_images_freq` renamed to `opts.ba_global_frames_freq`
- `init_image_id1`, `init_image_id2` still at top level
- `triangulation` sub-options unchanged

## Reconstruction
- `.find_image_with_name(name)` → Image | None
- `.images`, `.cameras`, `.points3D` (dict-like maps)
- `.frames` (FrameMap, NEW), `.rigs` (RigMap, NEW)
- `.transform(sim3d)`, `.write(path)` (directory must exist)
- `.add_camera()`, `.add_image()`, `.add_point3D()`, `.add_frame()`, `.add_rig()`
- `.register_frame(frame_id)` — NEW, required for images to persist on write/read
- `.num_frames()`, `.num_rigs()` — NEW

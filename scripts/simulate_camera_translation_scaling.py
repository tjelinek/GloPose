import kaolin
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from kornia.geometry import Se3, Quaternion

from auxiliary_scripts.math_utils import Se3_epipolar_cam_from_Se3_obj
from utils import homogenize_3x4_transformation_matrix

camera_trans = torch.tensor([[3.14, -5.0, -2.81]]).cpu().to(torch.float32)
camera_up = torch.tensor([[0, 0, 1]]).cpu().to(torch.float32)
obj_center = torch.zeros((1, 3,)).cpu().to(torch.float32)

T_world_to_cam_4x3 = kaolin.render.camera.generate_transformation_matrix(camera_position=camera_trans,
                                                                         camera_up_direction=camera_up,
                                                                         look_at=obj_center)
T_world_to_cam_4x4 = homogenize_3x4_transformation_matrix(T_world_to_cam_4x3.permute(0, 2, 1))
Se3_obj_to_cam = Se3.from_matrix(T_world_to_cam_4x4)

#############################################################

rr.init(f'simulating_rotations')
rr.save(f'rerun_simulating_rotations.rrd')

blueprint = rrb.Blueprint(
    rrb.Spatial3DView(origin='pose', name='3D Space'),
)
rr.send_blueprint(blueprint)

###############################################################

translation = torch.tensor([[1., 2., 3.]]).cpu()
rotation = Quaternion.random(batch_size=1).cpu()
Se3_obj1_to_obj2_gt = Se3(rotation, translation)

position_cam1 = Se3_obj_to_cam.inverse().t.squeeze()

Se3_cam1_to_cam2_scaled = Se3_epipolar_cam_from_Se3_obj(Se3_obj1_to_obj2_gt, Se3_obj_to_cam)
Se3_obj1_to_cam2_scaled = Se3_cam1_to_cam2_scaled * Se3_obj_to_cam
Se3_obj1_to_obj2_scaled = Se3_obj_to_cam.inverse() * Se3_obj1_to_cam2_scaled
Se3_obj2_to_obj1_scaled = Se3_obj1_to_obj2_scaled.inverse()
Se3_obj1_to_obj1_scaled = Se3_obj2_to_obj1_scaled * Se3_obj1_to_obj2_scaled

position_cam2_scaled = Se3_obj1_to_cam2_scaled.inverse().t.squeeze()
position_obj2_scaled = Se3_obj1_to_obj2_scaled.inverse().t.squeeze()
position_obj1_scaled = Se3_obj1_to_obj1_scaled.inverse().t.squeeze()

colors_unscaled = (np.asarray([[0, 255, 0]] * 4) * np.array([1., 0.75, 0.5, 0.25])[:, np.newaxis]).astype(np.uint8)
colors_scaled = (np.asarray([[0, 0, 255]] * 4) * np.array([1., 0.75, 0.5, 0.25])[:, np.newaxis]).astype(np.uint8)
strips_radii = np.asarray([0.1] * 4)

for factor in torch.linspace(0, 2, 100).cpu():
    Se3_cam1_to_cam2_unscaled = Se3(Se3_cam1_to_cam2_scaled.quaternion, Se3_cam1_to_cam2_scaled.t * factor)
    Se3_obj1_to_cam2_unscaled = Se3_cam1_to_cam2_unscaled * Se3_obj_to_cam
    Se3_obj1_to_obj2_unscaled = Se3_obj_to_cam.inverse() * Se3_obj1_to_cam2_unscaled
    Se3_obj2_to_obj1_unscaled = Se3_obj1_to_obj2_unscaled.inverse()
    Se3_obj1_to_obj1_unscaled = Se3_obj2_to_obj1_unscaled * Se3_obj1_to_obj2_unscaled

    position_cam2_unscaled = Se3_obj1_to_cam2_unscaled.inverse().t.squeeze()
    position_obj2_unscaled = Se3_obj1_to_obj2_unscaled.inverse().t.squeeze()
    position_obj1_unscaled = Se3_obj1_to_obj1_unscaled.inverse().t.squeeze()

    rr.set_time_sequence('scale_factor', int(factor * 100))

    line_strip_unscaled = np.stack([obj_center[0], position_cam1.numpy(force=True),
                                    position_cam2_unscaled.numpy(force=True),
                                    position_obj2_unscaled.numpy(force=True),
                                    position_obj1_unscaled.numpy(force=True)])

    line_strip_scaled = np.stack([obj_center[0], position_cam1.numpy(force=True),
                                  position_cam2_scaled.numpy(force=True),
                                  position_obj2_scaled.numpy(force=True),
                                  position_obj1_scaled.numpy(force=True)])

    line_strip_unscaled = np.stack([line_strip_unscaled[:-1], line_strip_unscaled[1:]], axis=1)
    line_strip_scaled = np.stack([line_strip_scaled[:-1], line_strip_scaled[1:]], axis=1)

    rr.log('pose/unscaled_rotation',
           rr.LineStrips3D(strips=line_strip_unscaled,
                           colors=colors_unscaled,
                           radii=strips_radii))

    # rr.log('pose/unscaled_rotation_cam1_to_obj',
    #        rr.LineStrips3D(strips=[position_cam2_unscaled, position_cam1],
    #                        colors=[[255, 255, 0]],
    #                        radii=[[0.1]]))

    rr.log('pose/scaled_rotation',
           rr.LineStrips3D(strips=line_strip_scaled,
                           colors=colors_scaled,
                           radii=strips_radii))

    rr.log(
        f'pose/camera1',
        rr.Transform3D(translation=Se3_obj_to_cam.inverse().t.squeeze().numpy(force=True),
                       rotation=rr.Quaternion(xyzw=Se3_obj_to_cam.inverse().quaternion.q.squeeze().numpy(force=True)),
                       from_parent=None),
    )
    rr.log(
        f'pose/camera1',
        rr.Pinhole(
            resolution=[1, 1],
            focal_length=[1., 1.],
            camera_xyz=rr.ViewCoordinates.RUB,
        ),
    )

    rr.log(f'pose/camera2',
           rr.Transform3D(translation=Se3_obj1_to_cam2_unscaled.inverse().t.squeeze().numpy(force=True),
                          rotation=rr.Quaternion(
                              xyzw=Se3_obj1_to_cam2_unscaled.inverse().quaternion.q.squeeze().numpy(force=True)),
                          from_parent=None),
           )
    rr.log(
        f'pose/camera2',
        rr.Pinhole(
            resolution=[1, 1],
            focal_length=[1., 1.],
            camera_xyz=rr.ViewCoordinates.RUB,
        ),
    )

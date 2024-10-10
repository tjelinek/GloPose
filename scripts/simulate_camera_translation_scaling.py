import kaolin
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from kornia.geometry import Se3, Quaternion

from utils import homogenize_3x4_transformation_matrix

camera_trans = torch.tensor([[3.14, -5.0, -2.81]]).cuda().to(torch.float32)
camera_up = torch.tensor([[0, 0, 1]]).cuda().to(torch.float32)
obj_center = torch.zeros((1, 3, )).cuda().to(torch.float32)

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

translation = torch.tensor([[1., 2., 3.]]).cuda()
rotation = Quaternion.random(batch_size=1).cuda()

position_cam1 = Se3_obj_to_cam.inverse().t.squeeze()

Se3_cam1_to_cam2_scaled = Se3(rotation, translation)
Se3_obj1_to_cam2_scaled = Se3_cam1_to_cam2_scaled * Se3_obj_to_cam
Se3_obj1_to_obj2_scaled = Se3_obj_to_cam * Se3_obj1_to_cam2_scaled
position_cam2_scaled = Se3_obj1_to_cam2_scaled.inverse().t.squeeze()
position_obj2_scaled = Se3_obj1_to_obj2_scaled.inverse().t.squeeze()

colors_unscaled = [[0., 255., 0.]] * 3
colors_scaled = [[0., 0., 255.]] * 3
strips_radii = [0.1] * 3

for factor in torch.linspace(0, 2, 100).cuda():

    Se3_cam1_to_cam2_unscaled = Se3(rotation, translation * factor)
    Se3_obj1_to_cam2_unscaled = Se3_cam1_to_cam2_unscaled * Se3_obj_to_cam
    Se3_obj1_to_obj2_unscaled = Se3_obj_to_cam * Se3_obj1_to_cam2_unscaled

    position_cam2_unscaled = Se3_obj1_to_cam2_unscaled.inverse().t.squeeze()
    position_obj2_unscaled = Se3_obj1_to_obj2_unscaled.inverse().t.squeeze()

    rr.set_time_sequence('scale_factor', int(factor * 100))

    breakpoint()
    line_strip_unscaled = np.stack([obj_center[0], position_cam1, position_cam2_unscaled, position_obj2_unscaled])
    line_strip_scaled = np.stack([obj_center[0], position_cam1, position_cam2_scaled, position_obj2_scaled])

    rr.log('pose/unscaled_rotation',
           rr.LineStrips3D(strips=line_strip_unscaled,  # gt_t_cam
                           colors=colors_unscaled,
                           radii=strips_radii))

    rr.log('pose/unscaled_rotation',
           rr.LineStrips3D(strips=line_strip_scaled,  # pred_t_cam
                           colors=colors_scaled,
                           radii=strips_radii))
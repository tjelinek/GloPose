import kaolin
import rerun as rr
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

translation = torch.tensor([[1., 2., 3.]]).cuda()
rotation = Quaternion.random(batch_size=1).cuda()

position_cam1 = Se3_obj_to_cam.inverse().t.squeeze()

Se3_cam1_to_cam2_scaled = Se3(rotation, translation)
Se3_obj1_to_cam2_scaled = Se3_cam1_to_cam2_scaled * Se3_obj_to_cam
Se3_obj1_to_obj2_scaled = Se3_obj_to_cam * Se3_obj1_to_cam2_scaled
position_cam2_scaled = Se3_obj1_to_cam2_scaled.inverse().t.squeeze()
position_obj2_scaled = Se3_obj1_to_obj2_scaled.inverse().t.squeeze()

for factor in torch.linspace(0, 2, 100).cuda():

    Se3_cam1_to_cam2_unscaled = Se3(rotation, translation * factor)
    Se3_obj1_to_cam2_unscaled = Se3_cam1_to_cam2_unscaled * Se3_obj_to_cam
    Se3_obj1_to_obj2_unscaled = Se3_obj_to_cam * Se3_obj1_to_cam2_unscaled

    position_cam2_unscaled = Se3_obj1_to_cam2_unscaled.inverse().t.squeeze()
    position_obj2_unscaled = Se3_obj1_to_obj2_unscaled.inverse().t.squeeze()

    breakpoint()
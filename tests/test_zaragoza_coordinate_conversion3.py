import math

import torch
from kornia.geometry import (Quaternion, rotation_matrix_to_axis_angle, Se3, So3)



x = torch.Tensor((10., 0., 0.)).cuda()

r1_rad = torch.deg2rad(torch.Tensor([0, 30., 0.])).cuda()
r2_rad = torch.deg2rad(torch.Tensor([0, 0., -30])).cuda()
rot_rad = torch.deg2rad(torch.Tensor([0, 30., -30])).cuda()

print(f'-------------------{rot_rad}')
print(f'{(rot_rad ** 2).sum()}')

breakpoint()

R1 = Quaternion.from_axis_angle(r1_rad).matrix()
R2 = Quaternion.from_axis_angle(r2_rad).matrix()
R23 = Quaternion.from_axis_angle(rot_rad).matrix()
x2 = R2 @ R1 @ x
x23 = R23 @ x
x2_gt = torch.Tensor((5 * math.sqrt(3) / 2, 5, 5 * math.sqrt(3) / 2))
print(x23)
#print((x2 ** 2).sum())
print(x2)

print(x2_gt)

So3.identity() * So3.identity()
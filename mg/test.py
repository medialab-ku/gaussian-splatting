from superpixel import SuperPixelManager
from numpy.linalg import inv
import numpy as np
import cv2
import math
import torch

intr = torch.eye((3), dtype=torch.float32)
inv_intr = torch.eye((3), dtype=torch.float32)
fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

intr[0][0] = fx
intr[0][2] = cx
intr[1][1] = fy
intr[1][2] = cy
intr[2][2] = 1

inv_intr[0][0] = 1 / fx
inv_intr[0][2] = -cx / fx
inv_intr[1][1] = 1 / fy
inv_intr[1][2] = -cy / fy
inv_intr[2][2] = 1

xyz = torch.zeros((3, 1), dtype=torch.float32)
xyz[2] = 5
xyzs = torch.zeros((3, 0), dtype=torch.float32)
xyzs = torch.cat((xyzs, xyz.detach()), dim=1)

xyz = torch.zeros((3, 1), dtype=torch.float32)
xyz[0] = 2
xyz[1] = 4
xyz[2] = 20
xyzs = torch.cat((xyzs, xyz.detach()), dim=1)


print(xyzs)
print(xyzs.shape)

print(intr.shape)

uv = torch.matmul(intr, xyzs)
print(uv.shape)
print(uv)


uv = uv[:, :] / uv[2, :]
print(uv)
print(uv.shape)


recover = torch.matmul(inv_intr, uv)*20
print(recover)



#
# d = query_kf.unsqueeze(dim=2)
# d = d / 5000.0
# xyz = torch.mul(xy_one, d)
# print(xyz)
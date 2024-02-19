from superpixel import SuperPixelManager
from numpy.linalg import inv
import numpy as np
import cv2
import math
import torch
from datetime import datetime

fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

near = 0.1
far = 3.0


u1 = 0
u2 = 640
v1 = 0
v2 = 480

for i in range(10, 0, -1):
    idx_1 = i - 1
    print("idx1", idx_1)
    for j in range(idx_1, 0, -1):
        idx_2 = j - 1
        print(idx_1, idx_2)



near_x1 = (u1 - cx) * near / fx

near_x2 = (u2 - cx) * near / fx

near_y1 = (v1 - cy) * near / fy

near_y2 = (v2 - cy) * near / fy

far_x1 = (u1 - cx) * far / fx

far_x2 = (u2 - cx) * far / fx

far_y1 = (v1 - cy) * far / fy

far_y2 = (v2 - cy) * far / fy

print(near_x1, near_x2, near_y1, near_y2)
print(far_x1, far_x2, far_y1, far_y2)

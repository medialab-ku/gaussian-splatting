from fast_slic import Slic
from superpixel import SuperPixelManager
import numpy as np
import cv2
import math


fx = 535.4
fy = 539.2
cx = 320.1
cy = 247.6

intr[0][0] = fx
intr[0][2] = cx
intr[1][1] = fy
intr[1][2] = cy
intr[2][2] = 1

point = np.array([-1.20930433, -1.06417656,  2.41499996])


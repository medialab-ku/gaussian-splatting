from superpixel import SuperPixelManager
from numpy.linalg import inv
import numpy as np
import cv2
import math
import torch
from datetime import datetime
dc = {}
dc["A"] = 1
dc["B"] = 2
dc["C"] = 3
dc["D"] = 4
dc["E"] = 5


tr = torch.arange(4, dtype=torch.int32, device="cpu").unsqueeze(1).T
tr2 = torch.arange(3).unsqueeze(1).T

print(tr)
print(tr.shape)

print(tr2)
print(tr2.shape)

tr2 = tr2 + tr.shape[1]

print(int(tr2[0][1]))
print(tr2)
print(tr.shape)
print(tr2.shape)

from fast_slic import Slic
from superpixel import SuperPixelManager
import numpy as np
import cv2
sp_manager = SuperPixelManager()
path = "c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/rgb/1341847980.722988.png"
img = cv2.imread(path)
super_pixel_index = sp_manager.ComputeSuperPixel(img)
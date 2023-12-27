
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil

import numpy as np
import cv2
from numpy.linalg import inv


class SuperPixelManager:
    def __init__(self):
        self.iteration_N = 1
        self.region_size = 30
        self.ruler = 100

    def ComputeSuperPixel(self, rgb):
        slic = cv2.ximgproc.createSuperpixelSLIC(rgb, algorithm=102, region_size=self.region_size, ruler=self.ruler)
        slic.iterate(self.iteration_N)

        slic.enforceLabelConnectivity()
        lbls = slic.getLabels()
        num_slic = slic.getNumberOfSuperpixels()

        indices = []
        # sample_idxs = np.random.choice(np.arange(num_slic), size=SAMPLE_SIZE, replace=False)
        for cls_lbl in range(num_slic):
            fst_cls = np.argwhere(lbls == cls_lbl)
            y, x = fst_cls[:, 0], fst_cls[:, 1] # x: 가로, y: 세로
            indices.append((y.mean(), x.mean()))
        lsc_mask = slic.getLabelContourMask()
        cv2.imshow("super", lsc_mask)

        return indices

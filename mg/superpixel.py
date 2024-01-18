
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil
import torch

import numpy as np
import cv2


class SuperPixelManager:
    def __init__(self):
        self.iteration_N = 1
        self.region_size = 20
        self.ruler = 100
        self.device = "cuda"

    def ComputeSuperPixel(self, rgb):
        slic = cv2.ximgproc.createSuperpixelSLIC(rgb, algorithm=102, region_size=self.region_size, ruler=self.ruler)
        slic.iterate(self.iteration_N)

        slic.enforceLabelConnectivity()
        lbls = slic.getLabels()
        num_slic = slic.getNumberOfSuperpixels()

        with torch.no_grad():
            indices = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            # indices = []
            # TODO: 아래의 루프문을 개선해야한다.
            for cls_lbl in range(num_slic):
                fst_cls = np.argwhere(lbls == cls_lbl)
                y, x = fst_cls[:, 0], fst_cls[:, 1] # x: 가로, y: 세로
                indices = torch.cat((indices, torch.tensor(([[int(y.mean())], [int(x.mean())]]),
                                                           dtype=torch.int32, device=self.device)), dim=1)

        #
        return indices

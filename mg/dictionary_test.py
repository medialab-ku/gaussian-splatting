
import cv2
import math
import numpy as np
import torch
from torch import nn


orb_cuda = cv2.cuda_ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                    WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )

orb = cv2.ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                          WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )


bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
dictionary = np.loadtxt("dictionary.txt").astype(np.float32)
bowDiction = cv2.BOWImgDescriptorExtractor(orb_cuda, cv2.BFMatcher(cv2.NORM_HAMMING))
bowDiction.setVocabulary(dictionary)

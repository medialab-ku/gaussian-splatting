
import os

import numpy as np
import cv2

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images
#
path = "C:/lab/research/dataset/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk/rgb"
path2 = "C:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/rgb"
images = load_images_from_folder(path2)
BOW_cuda = cv2.BOWKMeansTrainer(1000)
BOW = cv2.BOWKMeansTrainer(1000)

# sift = cv2.SIFT_create()
orb = cv2.ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                          WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )
orb_cuda = cv2.cuda_ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                          WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )

desc_type = np.uint8
gpuMat = cv2.cuda_GpuMat()
des_from_cuda = np.empty((32, 1000))
for img in images:
    # kp, des = orb.detectAndCompute(img, None)
    # BOW.add(np.float32(des))
    # kp, des = sift.detectAndCompute(img, None)
    # BOW.add(des)

    gpuMat.upload(img)
    kp_cuda, des_cuda = orb_cuda.detectAndComputeAsync(gpuMat, None)
    des_from_cuda = des_cuda.download()
    # print("cuda", des_cuda)
    # des_cuda.download(des_from_cuda)
    # print("cpu", des_from_cuda)

    BOW_cuda.add(np.float32(des_from_cuda))

# dictionary = BOW.cluster().astype(desc_type)
# np.savetxt("dictionary.txt", dictionary)

#
dictionary_cuda = BOW_cuda.cluster().astype(desc_type)
np.savetxt("dictionary_cuda.txt", dictionary_cuda)


# dictionary = np.loadtxt("dictionary.txt").astype(desc_type)
# dictionary_cuda = np.loadtxt("dictionary_cuda.txt").astype(desc_type)
# # dictionary = dictionary.astype(np.float32)
#
# bowDiction = cv2.BOWImgDescriptorExtractor(orb, cv2.BFMatcher(cv2.NORM_HAMMING))
# bowDiction.setVocabulary(dictionary_cuda)

# for img in images:
#     # kp = orb.detect(img, None)
#     # desc1 = bowDiction.compute(img, kp)
#     # print("desc1", desc1)
#
#     gpuMat.upload(img)
#     kp_cuda = orb_cuda.detectAsync(gpuMat, None)
#     desc2 = bowDiction.compute(img, orb_cuda.convert(kp_cuda))
#     print("desc2", desc2)
# np.savetxt("dictionary.txt", dictionary)
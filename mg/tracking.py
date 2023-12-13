
import cv2
import numpy as np
import torch
class Tracker:
    def __init__(self):
        self.device = "cpu"
        self.SetIntrinsics()
        self.SetORBSettings()
        self.KF_gray_list = []
        self.KF_depth_list = []
        self.KF_orb_list = []
    def SetIntrinsics(self):
        fx = 535.4
        fy = 539.2
        cx = 320.1
        cy = 247.6

        self.intr = torch.zeros((3, 3), device=self.device)
        self.intr[0][0] = fx
        self.intr[0][2] = cx
        self.intr[1][1] = fy
        self.intr[1][2] = cy
        self.intr[2][2] = 1

        self.inv_intr = torch.zeros((3, 3), device=self.device)
        self.inv_intr[0][0] = 1 / fx
        self.inv_intr[0][2] = -cx / fx
        self.inv_intr[1][1] = 1 / fy
        self.inv_intr[1][2] = -cy / fy
        self.inv_intr[2][2] = 1

    def SetORBSettings(self):
        self.orb = cv2.ORB_create(
            nfeatures=40000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=20,
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def Track(self, rgb, gray, depth):
        if len(self.KF_gray_list) == 0:
            self.KF_gray_list.append(gray)
            self.KF_depth_list.append(depth)
            kp, des = self.orb.detectAndCompute(gray, None)
            self.KF_orb_list.append((kp, des))
        else:
            current_kp, current_des = self.orb.detectAndCompute(gray, None)
            kf_orb = self.KF_orb_list[len(self.KF_orb_list)-1]
            matches = self.bf.match(current_des, kf_orb[1])
            matches = sorted(matches, key=lambda x: x.distance)
            match_cnt = 100
            for j in matches[:match_cnt]:
                kf_idx = j.trainIdx  # i.trainIdx
                kf_x, kf_y = kf_orb[0][kf_idx].pt
                int_kf_x = int(kf_x)
                int_kf_y = int(kf_y)
                if (int_kf_y == 0 or int_kf_x == 0 or int_kf_y == 479 or int_kf_x == 639):
                    continue
                # point_3d_list.append(xyz_keyframe[int_kf_y][int_kf_x])
                # color_3d_list.append(ref_rgb_list[int_kf_y][int_kf_x])
                # Append kf x, y list
                q_idx = j.queryIdx  # i.trainIdx
                x, y = current_kp[q_idx].pt
                # point_2d_list.append(np.array([x, y]))
            # Append query x,y list
                cv2.circle(gray, (int(x), int(y)), 3, (255, 0, 0), 2)
        self.Play(gray)

    def Play(self, img):
        cv2.imshow("test", img)
        cv2.waitKey(10)

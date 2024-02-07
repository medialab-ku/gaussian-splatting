
from plyfile import PlyData, PlyElement
import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv

from utility import Rot2Quat, QuaternionInfo


class Tracker:
    def __init__(self):
        self.device = "cuda"
        self.SetORBSettings()
        self.GenerateUVTensor()
        self.Initial = True
        self.KF_rgb = None
        self.KF_gray = None
        self.KF_xyz = None
        self.KF_orb = None
        self.intr = np.eye(3, dtype=np.float32)
        self.SetIntrinsics()
        # self.KF_pose = None

    def SetIntrinsics(self):
        fx = 535.4
        fy = 539.2
        cx = 320.1
        cy = 247.6

        self.intr[0][0] = fx
        self.intr[0][2] = cx
        self.intr[1][1] = fy
        self.intr[1][2] = cy
        self.intr[2][2] = 1

        self.inv_intr = torch.zeros((3, 3), device=self.device, dtype=torch.float32)
        self.inv_intr[0][0] = 1 / fx
        self.inv_intr[0][2] = -cx / fx
        self.inv_intr[1][1] = 1 / fy
        self.inv_intr[1][2] = -cy / fy
        self.inv_intr[2][2] = 1

        FoVx = 2*math.atan(640/(2*fx))
        FoVy = 2*math.atan(480/(2*fy))
        self.projection_matrix = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4, dtype = torch.float32)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def SetORBSettings(self):

        self.orb = cv2.ORB_create(
            nfeatures=1000,
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

    def GenerateUVTensor(self):
        width = 640
        height = 480
        u = torch.arange(width, dtype=torch.float32)
        for i in range(height - 1):
            u = torch.vstack((u, torch.arange(width)))

        v = torch.tile(torch.arange(height), (1, 1)).T
        for i in range(width - 1):
            v = torch.hstack((v, torch.tile(torch.arange(height, dtype=torch.float32), (1, 1)).T))

        self.uv = torch.stack((u, v), dim=2).to(self.device)



    def RecoverXYZFromKeyFrame(self, query_kf):
        scale_factor = 5000.0
        ones = torch.ones((self.uv.shape[0], self.uv.shape[1], 1), dtype=torch.float32).to(self.device)
        uv_one = torch.cat((self.uv, ones), dim=2).to(self.device)
        uv_one = torch.unsqueeze(uv_one, dim=2)

        xy_one = torch.tensordot(uv_one, self.inv_intr, dims=([3], [1])).squeeze()

        d = query_kf.unsqueeze(dim=2)
        d = d / scale_factor

        xyz = torch.mul(xy_one, d)
        return xyz

    def CreateKeyframe(self, rgb, gray, depth):
        # Convert Depth to XYZ
        KF_depth_map = torch.from_numpy(np.array(depth, dtype=np.float32)).to(self.device)
        KF_xyz = self.RecoverXYZFromKeyFrame(KF_depth_map).detach().to('cpu').numpy()

        self.KF_rgb = rgb
        self.KF_gray = gray
        self.KF_xyz = KF_xyz
        kp, des = self.orb.detectAndCompute(gray, None)
        self.KF_orb = (kp, des)
        # self.KF_pose = pose

    def Match2D2D(self, kf_orb, gray):
        ## gpu 연산으로 고치기
        match_cnt = 100
        current_kp, current_des = self.orb.detectAndCompute(gray, None)
        matches = self.bf.match(current_des, kf_orb[1])
        matches = sorted(matches, key=lambda x: x.distance)
        query_2d_list = []
        ref_3d_list = []
        ref_color_list = []
        for j in matches[:match_cnt]:
            # Append KF lists
            kf_idx = j.trainIdx  # i.trainIdx
            kf_x, kf_y = kf_orb[0][kf_idx].pt
            int_kf_x = int(kf_x)
            int_kf_y = int(kf_y)
            if (int_kf_y == 0 or int_kf_x == 0 or int_kf_y == 479 or int_kf_x == 639):
                continue
            ref_3d_list.append(self.KF_xyz[int_kf_y][int_kf_x])
            ref_color_list.append(self.KF_rgb[int_kf_y][int_kf_x])

            # Append Query list
            q_idx = j.queryIdx  # i.trainIdx
            x, y = current_kp[q_idx].pt
            query_2d_list.append(np.array([x, y], dtype=np.float32))

        return np.array(query_2d_list), np.array(ref_3d_list), np.array(ref_color_list, dtype=np.float32)

    def Track(self, play_instance):
        sensor = play_instance[1]
        rgb = sensor[0]
        gray = sensor[1]
        depth = sensor[2]

        cv2.imshow("Tracking input", rgb)
        cv2.waitKey(1)

        if not play_instance[0]:  # Abort (System is not awake)
            return

        if self.Initial:
            # initial KF
            self.Initial = False
            self.CreateKeyframe(rgb, gray, depth)
            # 0.0. Status
            # 0.1. First KF
            return [True, True], [rgb, gray, self.KF_xyz]
        else:
            # Perform 2D-2D matching and, get corresponding 3D(xyz) points
            query_2d_list, ref_3d_list, ref_color_list = self.Match2D2D(self.KF_orb, gray)

            # Crop near points (for mapping)
            z_mask_0 = ref_3d_list[:, 2] > 0.2
            ref_3d_list = ref_3d_list[z_mask_0]
            ref_color_list = ref_color_list[z_mask_0]
            query_2d_list = query_2d_list[z_mask_0]

            pnp_ref_3d_list = np.copy(ref_3d_list)
            pnp_query_2d_list = np.copy(query_2d_list)

            # Crop near points (for PNP Solver)
            z_mask_1 = pnp_ref_3d_list[:, 2] > 0.5
            pnp_ref_3d_list = pnp_ref_3d_list[z_mask_1]
            pnp_query_2d_list = pnp_query_2d_list[z_mask_1]
            # Crop far points (for PNP Solver)
            z_mask_2 = pnp_ref_3d_list[:, 2] <= 3
            pnp_ref_3d_list = pnp_ref_3d_list[z_mask_2]
            pnp_query_2d_list = pnp_query_2d_list[z_mask_2]

            # PNP Solver
            print("Track PNP", pnp_query_2d_list.shape, pnp_ref_3d_list.shape )
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_ref_3d_list, pnp_query_2d_list, self.intr,
                                                          distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999,
                                                          reprojectionError=1, iterationsCount=1)
            rot, _ = cv2.Rodrigues(rvec)
            quat = Rot2Quat(rot)
            axis, angle = QuaternionInfo(quat)
            shift = np.linalg.norm(tvec[:3, 0].T)
            # print(f"angle: {angle}, shift: {shift}")
            if 0.5 <= angle < 1.0 or 0.2 <= shift < 0.5:  # Mapping is required
                print(f"Make KF! angle: {angle}, shift: {shift}")
                self.CreateKeyframe(rgb, gray, depth)
                relative_pose = [rot, quat, tvec]

                # render_pkg = render(viewpoint_cam, self.gaussian, pipe, bg)
                return [True, False], [rgb, gray, self.KF_xyz], relative_pose, ref_3d_list, ref_color_list
            else:  # Mapping is not required
                return [False], []


import cv2
import math
import numpy as np
import torch
from torch import nn
class Mapper:


    def __init__(self):
        self.width = 640
        self.height = 480
        self.device = "cuda"
        with torch.no_grad():
            self.intr = torch.zeros((3, 3), device=self.device, dtype=torch.float32)
        self.SetIntrinsics()

        # from images
        self.KF_gray_list = []
        self.KF_rgb_list = []
        self.KF_xyz_list = []   # Converted from Depth map
        self.KF_superpixel_list = []

        with torch.no_grad():
            self.KF_poses = torch.empty((4, 4, 0), dtype=torch.float32, device=self.device)
        self.KF_gray_gpuMat = cv2.cuda_GpuMat()
        self.Current_gray_gpuMat = cv2.cuda_GpuMat()

        # points (2D, 3D)
        self.KF_ref_sp_3d_list = []
        self.KF_ref_sp_color_list = []
        with torch.no_grad():
            self.SP_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            self.pointclouds = torch.empty((6, 0), dtype=torch.float32, device=self.device)
            # super pixel pose
            self.SP_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            self.SP_index_list = torch.empty((0), dtype=torch.int32, device=self.device)

            # orb
        self.orb = cv2.cuda_ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                       WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )
        self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        self.KF_orb_list = []
        self.KF_recon_index_list = []
        self.KF_match_uv_list=[]
        self.iteration = 0

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

    def CreateInitialKeyframe(self, rgb, KF_xyz, KF_orb, keyframe_pose, KF_recon_index):
        self.KF_rgb_list.append(rgb)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_orb_list.append(KF_orb)
        self.KF_recon_index_list.append(KF_recon_index)
        with torch.no_grad():
            self.KF_poses = torch.cat((self.KF_poses, keyframe_pose.unsqueeze(dim=2)), dim=2)
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()


    def CreateKeyframe(self, rgb, gray, KF_xyz, KF_orb, keyframe_pose, KF_recon_index):
        self.KF_rgb_list.append(rgb)
        self.KF_gray_list.append(gray)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_orb_list.append(KF_orb)
        self.KF_recon_index_list.append(KF_recon_index)
        with torch.no_grad():
            self.KF_poses = torch.cat((self.KF_poses, keyframe_pose.unsqueeze(dim=2)), dim=2)

    def ConvertCamXYZ2GlobalXYZ(self, cam_xyz, pose):
        with torch.no_grad():
            ones = torch.full((1, cam_xyz.shape[1]), 1.0, dtype=torch.float32, device=self.device)
            cam_xyz_homo = torch.cat((cam_xyz, ones), dim=0)
            global_xyz = torch.matmul(pose, cam_xyz_homo)
        return global_xyz

    def CheckSuperPixelFrame(self, pose):
        with torch.no_grad():
            trace = torch.matmul(self.SP_pose, torch.inverse(pose))
            val = float(trace[0][0] + trace[1][1] + trace[2][2])
            if val > 1.0:
                val = 1.0
            elif val < -1.0:
                val = -1.0
            angle = math.acos((val-1)/2)

            shift_matrix = self.SP_pose[:3, 3] - pose[:3, 3]
            shift = torch.dot(shift_matrix, shift_matrix)
        if(angle > 0.5 or shift > 0.5):
            return True
        else:
            return False


    def Match2D2D(self, current_orb, current_recon_index, ref_orb, ref_reconindex, ref_rgb, ref_xyz, ref_pose):
        current_kp, current_des = current_orb
        ref_kp, ref_des = ref_orb
        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        ref_kp_cpu = self.orb.convert(ref_kp)
        current_kp_cpu = self.orb.convert(current_kp)

        match_cnt = 0
        match_cnt_threshold = 100
        for j in matches:
            if j.distance < 50:
                match_cnt+=1
            else:
                break
            if match_cnt == match_cnt_threshold:
                break

        pointcloud_index = self.pointclouds.shape[1]
        # ref_3d_list = []
        # ref_color_list = []

        with torch.no_grad():
            cam_xyz = torch.empty((3, 0), dtype=torch.float32, device=self.device)
            rgb = torch.empty((3, 0), dtype=torch.float32, device=self.device)
            ref_match_uv = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            current_match_uv = torch.empty((2, 0), dtype=torch.int32, device=self.device)

        for j in matches[:match_cnt]:
            # index of jth match
            ref_idx = j.trainIdx
            current_idx = j.queryIdx
            ref_u, ref_v = ref_kp_cpu[ref_idx].pt
            ref_u = int(ref_u)
            ref_v = int(ref_v)
            current_u, current_v = current_kp_cpu[current_idx].pt
            current_u = int(current_u)
            current_v = int(current_v)

            # Skip the edge of the images
            if current_u == 0 or current_v == 0 or ref_u == 0 or ref_v == 0 or current_u == self.width-1 \
                    or current_v == self.height-1 or ref_u == self.width-1 or ref_v == self.height-1:
                continue

            if current_recon_index[current_u][current_v] < 0 and ref_reconindex[ref_u][ref_v] < 0:
                # print(f"xyz {ref_xyz[ref_v][ref_u]}")  # shape: (3,)
                # print(f"rgb {ref_rgb[ref_v][ref_u]}")  # shape: (3,)
                if ref_xyz[ref_v][ref_u][2] < 0.1:
                    continue
                # ref_3d_list.append(ref_xyz[ref_v][ref_u])
                # ref_color_list.append(ref_rgb[ref_v][ref_u])

                with torch.no_grad():
                    cam_xyz = torch.cat((cam_xyz, torch.from_numpy(ref_xyz[ref_v][ref_u]).unsqueeze(dim=1).to(self.device)), dim=1)
                    rgb = torch.cat((rgb, torch.from_numpy (ref_rgb[ref_v][ref_u]).unsqueeze(dim=1).to(self.device)), dim=1)
                    current_match_uv = torch.cat((current_match_uv, torch.tensor(([[current_u], [current_v]]), dtype=torch.int32, device=self.device)), dim=1)
                    ref_match_uv = torch.cat((ref_match_uv, torch.tensor(([[ref_u], [ref_v]]), dtype=torch.int32, device=self.device)), dim=1)
                current_recon_index[current_u][current_v] = pointcloud_index
                ref_reconindex[ref_u][ref_v] = pointcloud_index
                pointcloud_index += 1
                continue
            # skip if already reconstructed
            elif current_recon_index[current_u][current_v] >= 0 and ref_reconindex[ref_u][ref_v] >= 0:
                continue

            elif ref_reconindex[ref_u][ref_v] >= 0:  # New match for current, but already recon for ref.
                current_recon_index[current_u][current_v] = ref_reconindex[ref_u][ref_v]
                with torch.no_grad():
                    current_match_uv = torch.cat((current_match_uv,
                                                  torch.tensor(([[current_u], [current_v]]), dtype=torch.int32,
                                                               device=self.device)), dim=1)
                continue
            elif current_recon_index[current_u][current_v] >= 0:  # Already recon in current, but new for ref.
                # Ignore, probably a noise
                continue

        # compute cam_xyz into global_xyz
        # concat pointclouds, (global_xyz, rgb)
        with torch.no_grad():
            global_xyz = self.ConvertCamXYZ2GlobalXYZ(cam_xyz, ref_pose)
            pointcloud = torch.cat((global_xyz[:3, :], rgb), dim=0)
            self.pointclouds = torch.cat((self.pointclouds, pointcloud), dim=1)
        return True, current_match_uv, ref_match_uv

    def ComputeCovisibility(self, current_orb):
        with torch.no_grad():
            current_recon_index = torch.full((self.width, self.height), -1, dtype=torch.int32, device=self.device)
            current_match_uv_total = torch.empty((2, 0), dtype=torch.int32, device=self.device)

        for i in range(len(self.KF_rgb_list)):
            recent_i = -(i+1)  # begin from the most recent frame
            ref_orb = self.KF_orb_list[recent_i]
            ref_recon_index = self.KF_recon_index_list[recent_i]
            ref_rgb = self.KF_rgb_list[recent_i]
            ref_xyz = self.KF_xyz_list[recent_i]
            with torch.no_grad():
                ref_pose = self.KF_poses[:, :, recent_i]
            match_result = self.Match2D2D(current_orb, current_recon_index, ref_orb, ref_recon_index, ref_rgb, ref_xyz, ref_pose)
            if not match_result[0]:
                break
            else:
                with torch.no_grad():
                    current_match_uv = match_result[1]
                    ref_match_uv = match_result[2]
                    current_match_uv_total = torch.cat((current_match_uv_total, current_match_uv), dim=1)
                    self.KF_match_uv_list[recent_i] = torch.cat((self.KF_match_uv_list[recent_i], ref_match_uv), dim=1)

        self.KF_match_uv_list.append(current_match_uv_total)
        # print(f"New {len(self.KF_match_uv_list) -1 }th match uv: {self.KF_match_uv_list[-1].shape}\n")

        return current_recon_index



    def FullBundleAdjustment(self):
        Result_GMapping = False
        Result_First_KF = False
        Result_BA = True

        if len(self.KF_match_uv_list) < 3:
            return [False, False, False], []

        ones = torch.full((1, self.pointclouds.shape[1]), 1.0, dtype=torch.float32, device=self.device)
        pointclouds = nn.Parameter(
            torch.cat((self.pointclouds[:3, :].detach(), ones), dim=0).detach().requires_grad_(True))
        poses = nn.Parameter(self.KF_poses[:3, :, :].detach().requires_grad_(True))
        pose_last = torch.eye((4), dtype=torch.float32,
                              device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
        poses_four = torch.cat((poses, pose_last), dim=0)
        pointclouds_lr = 1e-4
        poses_lr = 1e-4
        l = [
            {'params': [pointclouds], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses], 'lr': poses_lr, "name": "poses"}
        ]
        optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        print("BA Begins")
        for iteration in range(1):
            uv_cnt = 0
            loss_total = 0
            for i in range(0, len(self.KF_match_uv_list)):
                match_uv = self.KF_match_uv_list[i]
                if match_uv.shape[1] == 0:
                    continue
                uv_cnt += match_uv.shape[1]
                recon_index = self.KF_recon_index_list[i]
                pointcloud_indice = recon_index[match_uv[0, :], match_uv[1, :]]
                pointcloud_seen_from_kf = torch.index_select(pointclouds, 1, pointcloud_indice)
                pose_from_kf = poses_four[:, :, i]
                world_to_kf = torch.inverse(pose_from_kf)
                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]
                match_uv_mask = match_uv[:, mask]
                loss_total += torch.sum(torch.norm((match_uv_mask - cam_uv_mask[:2, :]), dim=0)) # 개선해야함
            # loss_total = loss_total/uv_cnt
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.pointclouds[:3, :] = pointclouds[:3, :].detach()
        # poses = poses / poses[3, 3, :]
        self.KF_poses[:3, :, :] = poses[:3, :, :].detach()
        SP_poses = torch.index_select(self.KF_poses, 2, self.SP_index_list).detach()
        self.SP_pose = SP_poses[:, :, -1].detach()
        print("BA ENDS")
        return [Result_GMapping, Result_First_KF, Result_BA], [SP_poses.detach().cpu()]

    def Map(self, tracking_result_instance):
        Result_GMapping = False
        Result_First_KF = False
        Result_BA = False

        if not tracking_result_instance[0]:  # Abort (System is not awake)
            return
        tracking_result = tracking_result_instance[1]
        status = tracking_result[0]  #[Tracking Success, First KF]
        sensor = tracking_result[1]
        rgb_img = sensor[0]
        gray_img = sensor[1]
        KF_xyz = sensor[2]

        self.Current_gray_gpuMat.upload(gray_img)
        current_kp, current_des = self.orb.detectAndComputeAsync(self.Current_gray_gpuMat, None)

        if status[1]:  # First KF
            Result_GMapping = True
            Result_First_KF = False
            Result_BA = False
            with torch.no_grad():

                self.CreateInitialKeyframe(rgb_img, KF_xyz, (current_kp, current_des),
                                           torch.eye(4, dtype=torch.float32, device=self.device),
                                           torch.full((self.width, self.height), -1, dtype=torch.int32,
                                                      device=self.device))
                self.SP_index_list = torch.cat((self.SP_index_list, torch.tensor([0], dtype=torch.int32, device=self.device)), dim=0)
                self.KF_match_uv_list.append(torch.empty((2, 0), dtype=torch.int32, device=self.device))  # Nothing to match
            return [Result_GMapping, Result_First_KF, Result_BA], [rgb_img, KF_xyz], \
                torch.eye(4, dtype=torch.float32).cpu()

        else:  # Not first KF
            Result_GMapping = True
            Result_First_KF = True
            Result_BA = False
            # 이전 키프레임을 기준으로 한 point들을 저장한다.
            # 현재 키프레임과 이전 키프레임 사이에서 생성된 point들인데, origin은 이전 것을 기준으로 함.
            ref_3d_list = tracking_result[3]  # 이전 프레임의, Camera 스페이스 xyz임. / 가우시안에 corner점도 추가할 까 해서 넣은 것
            ref_color_list = tracking_result[4]  # 위에 꺼랑 매칭 되는 RGB임


            # 현 키프레임 이미지와, 새로운 pose를 저장한다.
            relative_pose = tracking_result[2]
            rot = relative_pose[0]
            quat = relative_pose[1]
            tvec = relative_pose[2]

            with torch.no_grad():
                KF_relative_pose = torch.eye(4, dtype=torch.float32, device=self.device)
                KF_relative_pose[:3, :3] = torch.from_numpy(rot)
                KF_relative_pose[:3, 3] = torch.from_numpy(tvec).squeeze()

                Prev_KF_pose= self.KF_poses[:, :, -1]

                KF_current_pose = torch.matmul(Prev_KF_pose, torch.inverse(KF_relative_pose))
                current_recon_index = self.ComputeCovisibility((current_kp, current_des))
                self.CreateKeyframe(rgb_img, gray_img, KF_xyz, (current_kp, current_des), KF_current_pose, current_recon_index)
                if self.CheckSuperPixelFrame(KF_current_pose):
                    self.SP_pose = KF_current_pose
                    self.SP_index_list = torch.cat(
                        (self.SP_index_list, torch.tensor([self.KF_poses.shape[2] - 1], dtype=torch.int32,
                                                          device=self.device)), dim=0)
                    # prev_pose = self.KF_pose_list[-2]
                    # point_list_for_gaussian = self.TMPConvertCamera2World(ref_3d_list, prev_pose) # 이전 프레임의, Camera 스페이스 xyz임. / 가우시안에 corner점도 추가할 까 해서 넣은 것
                    return [Result_GMapping, Result_First_KF, Result_BA], [rgb_img, KF_xyz], KF_current_pose.detach().cpu()
                else:  # Gaussian Mapping is not required
                    Result_GMapping = False
                    Result_First_KF = False
                    Result_BA = False
                    return [Result_GMapping, Result_First_KF, Result_BA], []

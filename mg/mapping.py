
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
            self.intr_np = np.eye(3, dtype=np.float32)
        self.SetIntrinsics()

        # from images
        self.KF_rgb_list = []
        self.KF_xyz_list = []   # Converted from Depth map
        self.KF_superpixel_list = []
        self.KF_bow_list = []
        self.KF_covis_list = []
        self.KF_loop_list = []
        self.KF_essen_list = []


        with torch.no_grad():
            self.KF_poses = torch.empty((4, 4, 0), dtype=torch.float32, device=self.device)
        self.KF_gray_gpuMat = cv2.cuda_GpuMat()
        self.Current_gray_gpuMat = cv2.cuda_GpuMat()

        # points (2D, 3D)
        with torch.no_grad():
            self.SP_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            self.pointclouds_ptr = torch.empty((1,0), dtype=torch.int32, device=self.device)
            self.pointclouds = torch.empty((6, 0), dtype=torch.float32, device=self.device)
            self.pointclouds_cntr = torch.empty((7, 0), dtype=torch.float32, device=self.device)  # xyz, rgb, cntr
            # super pixel pose
            self.SP_pose = torch.eye(4, dtype=torch.float32, device=self.device)
            self.SP_index_list = torch.empty((0), dtype=torch.int32, device=self.device)

        # orb
        self.orb_cuda = cv2.cuda_ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                            WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )

        self.orb = cv2.ORB.create(nfeatures=1000, scaleFactor=1.2, nlevels=8, edgeThreshold=31, firstLevel=0,
                                  WTA_K=2, scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=20, )
        # self.sift = cv2.SIFT_create()


        self.bf = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
        self.dictionary = np.loadtxt("dictionary_cuda.txt").astype(np.uint8)
        self.bowDiction = cv2.BOWImgDescriptorExtractor(self.orb, cv2.BFMatcher(cv2.NORM_HAMMING))  # ORB 여야 작동함. ORB CUDA는 안됌
        self.bowDiction.setVocabulary(self.dictionary)

        self.KF_orb_list = []
        self.KF_recon_index_list = []
        self.KF_match_uv_list=[]
        self.iteration = 0
        self.BA_iteration = 0
        self.LoopDetectionIndex = 0

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

        self.intr_np[0][0] = fx
        self.intr_np[0][2] = cx
        self.intr_np[1][1] = fy
        self.intr_np[1][2] = cy
        self.intr_np[2][2] = 1

    def CreateInitialKeyframe(self, rgb, KF_xyz, KF_orb, keyframe_pose, KF_recon_index):
        self.KF_rgb_list.append(rgb)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_orb_list.append(KF_orb)
        self.KF_recon_index_list.append(KF_recon_index)
        with torch.no_grad():
            self.KF_poses = torch.cat((self.KF_poses, keyframe_pose.unsqueeze(dim=2)), dim=2)
        self.KF_gray_gpuMat = self.Current_gray_gpuMat.clone()
        self.KF_covis_list.append([])
        self.KF_loop_list.append([])
        self.KF_essen_list.append([])


    def CreateKeyframe(self, rgb, KF_xyz, KF_orb, keyframe_pose, KF_recon_index):
        self.KF_rgb_list.append(rgb)
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

    def SimilarityBOW(self, desc1, desc2):
        diff = desc1/np.linalg.norm(desc1) - desc2/np.linalg.norm(desc2)
        result = 1 - 0.5 * np.linalg.norm(diff)
        return result

    def DetectLoop(self, current_idx):
        result = False
        bow_desc = self.KF_bow_list[current_idx]
        loop_neighbor_list = self.KF_covis_list[current_idx]
        bow_score_min = 1.0

        # oldest_loop_frame = None

        for loop_neighbor in loop_neighbor_list:
            bow_neighbor = self.KF_bow_list[loop_neighbor]
            score = self.SimilarityBOW(bow_desc, bow_neighbor)
            if bow_score_min > score:
                bow_score_min = score
        if bow_score_min < 0.67:  # 휴리스틱
            bow_score_min = 0.67
        loop_candidate = []
        for j in range(len(self.KF_bow_list)-1):
            bow_candidate = self.KF_bow_list[j]
            score_new = self.SimilarityBOW(bow_desc, bow_candidate)
            if bow_score_min < score_new and (not(j in loop_neighbor_list)):
                loop_candidate.append(j)
        # for neighbor_index in loop_neighbor_list:
        #     loop_candidate.remove(neighbor_index) if neighbor_index in loop_candidate else None
        if len(loop_candidate) < 3:
            # 3개 미만이면 볼 것도 없다.
            return False

        loop_candidate.sort()
        print(current_idx, bow_score_min, " loop candidate after remove", loop_candidate)
        cntr_serial = 1
        match2d2d_candidate_list = []
        for i in range(1, len(loop_candidate)):
            if loop_candidate[i] - loop_candidate[i-1] == 1:
                cntr_serial+=1
            else:
                if cntr_serial > 2:
                    # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
                    result = True
                    for j in range(cntr_serial):
                        hard_close = None
                        ref_idx = loop_candidate[i - j - 1]
                        if not (ref_idx in match2d2d_candidate_list):
                            match2d2d_candidate_list.append(ref_idx)
                cntr_serial = 0
        if cntr_serial > 2:
            # orb matching을 한 뒤에, essential, covis graph 등을 생성 한다.
            result = True
            for j in range(cntr_serial):
                hard_close = None
                ref_idx = loop_candidate[- j - 1]
                if not (ref_idx in match2d2d_candidate_list):
                    match2d2d_candidate_list.append(ref_idx)

        match2d2d_candidate_list_tmp = match2d2d_candidate_list.copy()
        for match2d2d_candidate in match2d2d_candidate_list:
            loop_list = self.KF_loop_list[match2d2d_candidate]
            for loop_candidate in loop_list:
                if not (loop_candidate in match2d2d_candidate_list_tmp):
                    match2d2d_candidate_list_tmp.append(loop_candidate)

        print ("match2d2d_candidate", match2d2d_candidate_list_tmp)
        print ("current_candidate", self.KF_loop_list[current_idx])
        for match2d2d_candidate in match2d2d_candidate_list_tmp:
            for current_candidate in self.KF_loop_list[current_idx]:
                self.LoopMatch2D2D(current_candidate, match2d2d_candidate, False)

        return result#, oldest_loop_frame

    def LoopMatch2D2D(self, current_idx, ref_idx, hard_close):
        # hard_close = False
        current_kp, current_des = self.KF_orb_list[current_idx]
        ref_kp, ref_des = self.KF_orb_list[ref_idx]

        current_recon_index = self.KF_recon_index_list[current_idx]
        ref_reconindex = self.KF_recon_index_list[ref_idx]

        ref_rgb = self.KF_rgb_list[ref_idx]
        ref_xyz = self.KF_xyz_list[ref_idx]
        ref_pose = self.KF_poses[:, :, ref_idx]

        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        ref_kp_cpu = self.orb_cuda.convert(ref_kp)
        current_kp_cpu = self.orb_cuda.convert(current_kp)

        match_cnt = 0
        covis_threshold = 15
        loop_threshold = 30
        essen_threshold = 100
        for j in matches:
            if j.distance < 40:
                match_cnt += 1
            else:
                break
        if match_cnt < covis_threshold:
            # covisiblity가 없다.
            return
        if match_cnt > covis_threshold:
            if not (ref_idx in self.KF_covis_list[current_idx]):
                self.KF_covis_list[current_idx].append(ref_idx)
            if not (current_idx in self.KF_covis_list[ref_idx]):
                self.KF_covis_list[ref_idx].append(current_idx)

        if match_cnt > loop_threshold:
            if not (ref_idx in self.KF_loop_list[current_idx]):
                self.KF_loop_list[current_idx].append(ref_idx)
            if not (current_idx in self.KF_loop_list[ref_idx]):
                self.KF_loop_list[ref_idx].append(current_idx)

        if match_cnt > essen_threshold:
            if not (ref_idx in self.KF_essen_list[current_idx]):
                self.KF_essen_list[current_idx].append(ref_idx)
            if not (current_idx in self.KF_essen_list[ref_idx]):
                self.KF_essen_list[ref_idx].append(current_idx)

        #################
        pointcloud_index = self.pointclouds_cntr.shape[1]

        cam_xyz_list = []
        cam_rgb_list = []
        ref_match_uv_list = []
        current_match_uv_list = []
        pointcloud_cntr_add_list = []
        pointcloud_cntr_subtract_list = []
        pointcloud_ptr_update = {}

        pnp_ref_3d_list = []
        pnp_query_2d_list = []

        mask = ref_reconindex.ge(0)
        for j in matches[:match_cnt]:
            # index of jth match
            ref_u, ref_v = ref_kp_cpu[j.trainIdx].pt
            ref_u = int(ref_u)
            ref_v = int(ref_v)
            current_u, current_v = current_kp_cpu[j.queryIdx].pt
            current_u = int(current_u)
            current_v = int(current_v)

            # Skip the edge of the images
            if current_u == 0 or current_v == 0 or ref_u == 0 or ref_v == 0 or current_u == self.width-1 \
                    or current_v == self.height-1 or ref_u == self.width-1 or ref_v == self.height-1:
                continue

            if hard_close:
                pnp_ref_3d_list.append(ref_xyz[ref_v][ref_u])
                pnp_query_2d_list.append(np.array([float(current_u), float(current_v)]))

            if current_recon_index[current_u][current_v] < 0 and ref_reconindex[ref_u][ref_v] < 0:
                # 새로운 포인트 클라우드 생성
                if ref_xyz[ref_v][ref_u][2] < 0.1:
                    # 노이즈 일 것이다. Skip
                    continue

                with torch.no_grad():
                    # 위의 Torch 연산을 조금 더 빠른 list로 바꿈
                    cam_xyz_list.append(ref_xyz[ref_v][ref_u])
                    cam_rgb_list.append(ref_rgb[ref_v][ref_u])
                    ref_match_uv_list.append((ref_u, ref_v))
                    current_match_uv_list.append((current_u, current_v))

                current_recon_index[current_u][current_v] = pointcloud_index
                ref_reconindex[ref_u][ref_v] = pointcloud_index
                pointcloud_index += 1
                continue
            elif current_recon_index[current_u][current_v] >= 0 and ref_reconindex[ref_u][ref_v] >= 0:
                # pointcloud_cntr_subtract_list.append(current_recon_index[current_u][current_v])
                # current_recon_index[current_u][current_v] = ref_reconindex[ref_u][ref_v]
                if current_recon_index[current_u][current_v] > ref_reconindex[ref_u][ref_v]:
                    pointcloud_ptr_update[current_recon_index[current_u][current_v]] = ref_reconindex[ref_u][ref_v]
                else:
                    pointcloud_ptr_update[ref_reconindex[ref_u][ref_v]] = current_recon_index[current_u][current_v]

                # pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])

                continue

            elif ref_reconindex[ref_u][ref_v] >= 0:  # New match for current, but already recon for ref.
                # 현재 프레임에서는 처음 생성, 기존에는 생성 된 포인트
                # 기존 pointcloud의 인덱스를 이어받는다.
                current_recon_index[current_u][current_v] = ref_reconindex[ref_u][ref_v]
                # 해당 인덱스 pointcloud의 카운트를 1 증가 시킨다.

                # pointcloud_ptr_update[current_recon_index[current_u][current_v]] = ref_reconindex[ref_u][ref_v]
                pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])
                current_match_uv_list.append((current_u, current_v))
                continue
            elif current_recon_index[current_u][current_v] >= 0:  # Already recon in current, but new for ref.
                ref_reconindex[ref_u][ref_v] = current_recon_index[current_u][current_v]
                # 해당 인덱스 pointcloud의 카운트를 1 증가 시킨다.
                # pointcloud_ptr_update[ref_reconindex[ref_u][ref_v]] = current_recon_index[current_u][current_v]
                pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])
                ref_match_uv_list.append((ref_u, ref_v))
                continue

        if hard_close:
            print("Hard Close!: current: ", current_idx, " / ref: ", ref_idx)
            ref_3d_list = np.array(pnp_ref_3d_list)
            query_2d_list = np.array(pnp_query_2d_list)

            # Crop near points (for PNP Solver)
            z_mask_1 = ref_3d_list[:, 2] > 0.1
            ref_3d_list = ref_3d_list[z_mask_1]
            query_2d_list = query_2d_list[z_mask_1]
            # Crop far points (for PNP Solver)
            z_mask_2 = ref_3d_list[:, 2] <= 3
            ref_3d_list = ref_3d_list[z_mask_2]
            query_2d_list = query_2d_list[z_mask_2]

            # PNP Solver
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(ref_3d_list, query_2d_list, self.intr_np,
                                                          distCoeffs=None, flags=cv2.SOLVEPNP_EPNP,
                                                          confidence=0.9999,
                                                          reprojectionError=1, iterationsCount=1)
            rot, _ = cv2.Rodrigues(rvec)
            with torch.no_grad():
                relative_pose = torch.eye(4, dtype=torch.float32, device=self.device)
                relative_pose[:3, :3] = torch.from_numpy(rot)
                relative_pose[:3, 3] = torch.from_numpy(tvec).squeeze()
                ref_pose = self.KF_poses[:, :, ref_idx]
                current_pose = torch.matmul(ref_pose, torch.inverse(relative_pose))
                self.KF_poses[:, :, current_idx] = current_pose.detach()



        with torch.no_grad():
            if len(cam_xyz_list) > 0:
                cam_xyz_torch = torch.from_numpy(np.array(cam_xyz_list)).T.to(self.device)
                cam_rgb_torch = torch.from_numpy(np.array(cam_rgb_list)).T.to(self.device)
                global_xyz_torch = self.ConvertCamXYZ2GlobalXYZ(cam_xyz_torch, ref_pose).to(self.device)

                pointcloud_cntr = torch.empty((7, cam_xyz_torch.shape[1]), device=self.device)
                pointcloud_cntr[:3, :] = global_xyz_torch[:3, :]
                pointcloud_cntr[3:6, :] = cam_rgb_torch[:, :]
                pointcloud_cntr[6, :] = 2

                pointcloud_ptr = torch.arange(cam_xyz_torch.shape[1], dtype=torch.int32,
                                              device=self.device).unsqueeze(1).T
                pointcloud_ptr += self.pointclouds_cntr.shape[1]

                # self.pointclouds = torch.cat((self.pointclouds, pointcloud_cntr[:6, :]), dim=1)
                self.pointclouds_cntr = torch.cat((self.pointclouds_cntr, pointcloud_cntr), dim=1)
                self.pointclouds_ptr = torch.cat((self.pointclouds_ptr, pointcloud_ptr), dim=1)
            for pc_idx in pointcloud_ptr_update:
                updated_idx = int(pointcloud_ptr_update[pc_idx])
                while not (updated_idx == int(self.pointclouds_ptr[0][updated_idx])):
                    print("updated_idx", updated_idx, self.pointclouds_ptr[0][updated_idx])
                    updated_idx = int(self.pointclouds_ptr[0][updated_idx].detach())

                cntr_update = float(self.pointclouds_cntr[6, pc_idx].detach())
                self.pointclouds_cntr[6, updated_idx] += cntr_update
                self.pointclouds_ptr[0][pc_idx] = updated_idx
            ref_match_uv_list_torch = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            current_match_uv_list_torch = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            if len(current_match_uv_list) > 0:
                current_match_uv_list_torch = torch.from_numpy(np.array(current_match_uv_list)).T.to(self.device)
                self.KF_match_uv_list[current_idx] = torch.cat((self.KF_match_uv_list[current_idx], current_match_uv_list_torch), dim=1)

            if len(ref_match_uv_list) > 0:
                ref_match_uv_list_torch = torch.from_numpy(np.array(ref_match_uv_list)).T.to(self.device)
                self.KF_match_uv_list[ref_idx] = torch.cat((self.KF_match_uv_list[ref_idx], ref_match_uv_list_torch), dim=1)
            # for i in pointcloud_cntr_add_list:
            #     pc_idx = int(self.pointclouds_ptr[0][i])
            #     self.pointclouds_cntr[6, pc_idx] += 1
            # for i in pointcloud_cntr_subtract_list:
            #     pc_idx = int(self.pointclouds_ptr[0][i])
            #     self.pointclouds_cntr[6, pc_idx] -= 1





    def Match2D2D(self, current_kf_idx, current_orb, current_recon_index, ref_kf_idx, ref_orb, ref_reconindex, ref_rgb, ref_xyz, ref_pose):
        current_kp, current_des = current_orb
        ref_kp, ref_des = ref_orb
        matches = self.bf.match(current_des, ref_des)
        matches = sorted(matches, key=lambda x: x.distance)

        covis = False
        essen = False
        loop = False

        ref_kp_cpu = self.orb_cuda.convert(ref_kp)
        current_kp_cpu = self.orb_cuda.convert(current_kp)

        match_cnt = 0
        covis_threshold = 15
        loop_threshold = 30
        essen_threshold = 100
        for j in matches:
            if j.distance < 40:
                match_cnt += 1
            else:
                break
        if match_cnt < covis_threshold:
            # covisiblity가 없다.
            return False, []
        if match_cnt > covis_threshold:
            covis = True

        if match_cnt > loop_threshold:
            loop = True

        if match_cnt > essen_threshold:
            essen = True

        pointcloud_index = self.pointclouds_cntr.shape[1]

        cam_xyz_list = []
        cam_rgb_list = []
        ref_match_uv_list = []
        current_match_uv_list = []
        pointcloud_cntr_add_list = []
        pointcloud_cntr_subtract_list = []
        pointcloud_ptr_update = {}

        mask = ref_reconindex.ge(0)
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
                # 새로운 포인트 클라우드 생성
                if ref_xyz[ref_v][ref_u][2] < 0.1:
                    # 노이즈 일 것이다. Skip
                    continue

                with torch.no_grad():
                    # 위의 Torch 연산을 조금 더 빠른 list로 바꿈
                    cam_xyz_list.append(ref_xyz[ref_v][ref_u])
                    cam_rgb_list.append(ref_rgb[ref_v][ref_u])
                    ref_match_uv_list.append((ref_u, ref_v))
                    current_match_uv_list.append((current_u, current_v))

                current_recon_index[current_u][current_v] = pointcloud_index
                ref_reconindex[ref_u][ref_v] = pointcloud_index
                pointcloud_index += 1
                continue
            elif current_recon_index[current_u][current_v] >= 0 and ref_reconindex[ref_u][ref_v] >= 0:
                # 이전 프레임에서 이미 관측되었고, 현재 프레임 에서도 이미 관측 되었다면, 노이즈 일 확률이 높다.
                # TODO: 제대로 하려면, 해당 pointcloud를 projection해서, 에러가 적은 것을 써야함.

                # pointcloud_cntr_subtract_list.append(current_recon_index[current_u][current_v])
                # current_recon_index[current_u][current_v] = ref_reconindex[ref_u][ref_v]

                if current_recon_index[current_u][current_v] > ref_reconindex[ref_u][ref_v]:
                    pointcloud_ptr_update[current_recon_index[current_u][current_v]] = ref_reconindex[ref_u][ref_v]
                else:
                    pointcloud_ptr_update[ref_reconindex[ref_u][ref_v]] = current_recon_index[current_u][current_v]
                # pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])
                continue

            elif ref_reconindex[ref_u][ref_v] >= 0:  # New match for current, but already recon for ref.
                # 현재 프레임에서는 처음 생성, 기존에는 생성 된 포인트
                # 기존 pointcloud의 인덱스를 이어받는다.
                current_recon_index[current_u][current_v] = ref_reconindex[ref_u][ref_v]
                # 해당 인덱스 pointcloud의 카운트를 1 증가 시킨다.

                # pointcloud_ptr_update[current_recon_index[current_u][current_v]] = ref_reconindex[ref_u][ref_v]
                pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])
                current_match_uv_list.append((current_u, current_v))
                continue
            elif current_recon_index[current_u][current_v] >= 0:  # Already recon in current, but new for ref.
                ref_reconindex[ref_u][ref_v] = current_recon_index[current_u][current_v]
                # 해당 인덱스 pointcloud의 카운트를 1 증가 시킨다.
                # pointcloud_ptr_update[ref_reconindex[ref_u][ref_v]] = current_recon_index[current_u][current_v]
                pointcloud_cntr_add_list.append(ref_reconindex[ref_u][ref_v])
                ref_match_uv_list.append((ref_u, ref_v))
                continue

        mask = ref_reconindex.ge(0)
        # compute cam_xyz into global_xyz
        # concat pointclouds, (global_xyz, rgb)
        with torch.no_grad():
            if len(cam_xyz_list) > 0:
                cam_xyz_torch = torch.from_numpy(np.array(cam_xyz_list)).T.to(self.device)
                cam_rgb_torch = torch.from_numpy(np.array(cam_rgb_list)).T.to(self.device)
                global_xyz_torch = self.ConvertCamXYZ2GlobalXYZ(cam_xyz_torch, ref_pose).to(self.device)

                pointcloud_cntr = torch.empty((7, cam_xyz_torch.shape[1]), device=self.device)
                pointcloud_cntr[:3, :] = global_xyz_torch[:3, :]
                pointcloud_cntr[3:6, :] = cam_rgb_torch[:, :]
                pointcloud_cntr[6, :] = 2

                pointcloud_ptr = torch.arange(cam_xyz_torch.shape[1],
                                              dtype=torch.int32, device=self.device).unsqueeze(1).T
                pointcloud_ptr += self.pointclouds_cntr.shape[1]

                # self.pointclouds = torch.cat((self.pointclouds, pointcloud_cntr[:6, :]), dim=1)
                self.pointclouds_cntr = torch.cat((self.pointclouds_cntr, pointcloud_cntr), dim=1)
                self.pointclouds_ptr = torch.cat((self.pointclouds_ptr, pointcloud_ptr), dim=1)
            for pc_idx in pointcloud_ptr_update:
                updated_idx = int(pointcloud_ptr_update[pc_idx])

                while not (updated_idx == int(self.pointclouds_ptr[0][updated_idx])):
                    print("updated_idx", updated_idx, self.pointclouds_ptr[0][updated_idx])
                    updated_idx = int(self.pointclouds_ptr[0][updated_idx].detach())
                cntr_update = float(self.pointclouds_cntr[6, pc_idx].detach())
                self.pointclouds_cntr[6, updated_idx] += cntr_update
                self.pointclouds_ptr[0][pc_idx] = updated_idx


            ref_match_uv_list_torch = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            current_match_uv_list_torch = torch.empty((2, 0), dtype=torch.int32, device=self.device)
            if len(ref_match_uv_list) > 0:
                ref_match_uv_list_torch = torch.from_numpy(np.array(ref_match_uv_list)).T.to(self.device)
            if len(current_match_uv_list) > 0:
                current_match_uv_list_torch = torch.from_numpy(np.array(current_match_uv_list)).T.to(self.device)
        # for i in pointcloud_cntr_add_list:
        #     pc_idx = int(self.pointclouds_ptr[0][i])
        #     self.pointclouds_cntr[6, pc_idx] += 1
        # for i in pointcloud_cntr_subtract_list:
        #     pc_idx = int(self.pointclouds_ptr[0][i])
        #     self.pointclouds_cntr[6, pc_idx] -= 1
        return True, current_match_uv_list_torch, ref_match_uv_list_torch, covis, loop, essen

    def ComputeCovisibility(self, current_orb):
        with torch.no_grad():
            current_recon_index = torch.full((self.width, self.height), -1, dtype=torch.int32, device=self.device)
            current_match_uv_total = torch.empty((2, 0), dtype=torch.int32, device=self.device)

        # 가장 최근 KF하고만 Covisibility를 검사한다. (스피드 이슈)

        loop_end = 5
        if self.KF_poses.shape[2] < 5:
            loop_end = self.KF_poses.shape[2]

        covis_list = []
        loop_list = []
        assen_list = []
        current_idx = len(self.KF_covis_list)
        for i in range(1, (1+loop_end)):
            ref_idx = current_idx-i

            ref_orb = self.KF_orb_list[ref_idx]
            ref_recon_index = self.KF_recon_index_list[ref_idx]
            ref_rgb = self.KF_rgb_list[ref_idx]
            ref_xyz = self.KF_xyz_list[ref_idx]
            with torch.no_grad():
                ref_pose = self.KF_poses[:, :, ref_idx]
            match_result = self.Match2D2D(current_idx, current_orb, current_recon_index, ref_idx, ref_orb,
                                          ref_recon_index, ref_rgb, ref_xyz, ref_pose)
            if not match_result[0]:
                break
            if match_result[3]:
                covis_list.append(ref_idx)
                if not (current_idx in self.KF_covis_list[ref_idx]):
                    self.KF_covis_list[ref_idx].append(current_idx)
            if match_result[4]:
                loop_list.append(ref_idx)
                if not (current_idx in self.KF_loop_list[ref_idx]):
                    self.KF_loop_list[ref_idx].append(current_idx)
            if match_result[5]:
                assen_list.append(ref_idx)
                if not (current_idx in self.KF_essen_list[ref_idx]):
                    self.KF_essen_list[ref_idx].append(current_idx)
            with torch.no_grad():
                current_match_uv = match_result[1]
                ref_match_uv = match_result[2]
                current_match_uv_total = torch.cat((current_match_uv_total, current_match_uv), dim=1)
                self.KF_match_uv_list[ref_idx] = torch.cat((self.KF_match_uv_list[ref_idx], ref_match_uv), dim=1)

        self.KF_match_uv_list.append(current_match_uv_total)
        # print(f"New {len(self.KF_match_uv_list) -1 }th match uv: {self.KF_match_uv_list[-1].shape}\n")
        self.KF_covis_list.append(covis_list)
        self.KF_loop_list.append(loop_list)
        self.KF_essen_list.append(assen_list)
        return current_recon_index

    def LoopBundleAdjustment(self, iteration_num):
        #TODO 해당 프레임의 covis만 수정하되
        #TODO 가장 old 프레임은 고정

        Result_GMapping = False
        Result_First_KF = False
        Result_BA = True

        if len(self.KF_match_uv_list) < 5:
            return [False, False, False], [], [], []


        current_idx = len(self.KF_bow_list) -1
        loop_neighbor_list = self.KF_covis_list[-1].copy()
        loop_neighbor_list.append(current_idx)

        with torch.no_grad():
            pointclouds = self.pointclouds_cntr.detach()
            ones = torch.ones((1, pointclouds.shape[1]), dtype=torch.float32, device=self.device)

            BA_poses = torch.empty((3, 4, 0), dtype=torch.float32, device=self.device)
            print("current_idx",current_idx, "loop_neighbor_list", loop_neighbor_list)
            for neighbor_idx in loop_neighbor_list:
                neighbor_pose = self.KF_poses[:3, :, neighbor_idx].detach().unsqueeze(dim=2)
                BA_poses = torch.cat((BA_poses, neighbor_pose), dim=2)

        pointclouds_param = nn.Parameter(torch.cat((pointclouds[:3, :].detach(), ones.detach()), dim=0).requires_grad_(True))
        poses_train = nn.Parameter(BA_poses[:, :, :-1].detach().requires_grad_(True))



        # poses = nn.Parameter(self.KF_poses[:3, :, :].detach().requires_grad_(True))
        # pose_last = torch.eye((4), dtype=torch.float32,
        #                       device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
        # poses_four = torch.cat((poses, pose_last), dim=0)
        pointclouds_lr = 1e-6
        poses_lr = 1e-5
        l = [
            {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses_train], 'lr': poses_lr, "name": "poses"}
        ]
        optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)
        print("LOOP BA BEGINS")
        hard_close_candidate = []
        for iteration in range(iteration_num):
            uv_cnt = 0
            loss_total = 0.0
            poses = torch.cat((poses_train, BA_poses[:, :, -1].detach().unsqueeze(dim=2)), dim=2)
            pose_last = torch.eye((4), dtype=torch.float32,
                                  device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
            poses_four = torch.cat((poses, pose_last), dim=0)
            for i in range(len(loop_neighbor_list)):
                kf_idx = loop_neighbor_list[i]
                match_uv = self.KF_match_uv_list[kf_idx]

                recon_index = self.KF_recon_index_list[kf_idx].detach()
                # print("recon_index", recon_index)
                pointclouds_ptr_indice = recon_index[match_uv[0, :], match_uv[1, :]]
                pointcloud_indice = torch.index_select(self.pointclouds_ptr, 1, pointclouds_ptr_indice).squeeze()
                pointcloud_seen_from_kf = torch.index_select(pointclouds_param, 1, pointcloud_indice)

                pointclouds_cntr = torch.index_select(pointclouds[6, :].unsqueeze(dim=0), 1, pointcloud_indice)
                cntr_mask = pointclouds_cntr[0, :] > 3

                pointcloud_seen_from_kf_ctr = pointcloud_seen_from_kf[:, cntr_mask]
                match_uv_ctr = match_uv[:, cntr_mask]

                pose_from_kf = poses_four[:, :, i]
                world_to_kf = torch.inverse(pose_from_kf)

                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf_ctr)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]  # projection 한 uv
                match_uv_mask = match_uv_ctr[:, mask]  # 원본 uv
                loss = torch.norm((match_uv_mask - cam_uv_mask[:2, :]), dim=0)
                sorted_loss, _ = loss.sort(dim=0)
                inlier_num_min = int(loss.shape[0] * 0.0)
                inlier_num_max = int(loss.shape[0] * 1.0)
                loss_kf = torch.sum(sorted_loss[inlier_num_min:inlier_num_max])
                loss_total += loss_kf
                uv_cnt += (inlier_num_max - inlier_num_min)
                # if iteration == iteration_num-1 and (loss_kf/(inlier_num_max - inlier_num_min) > 50):
                #     hard_close_candidate.append(kf_idx)
                # loss_mask = loss[loss.lt(10.0)]
                # loss_total += torch.sum(loss_mask)
                # uv_cnt += int(loss_mask.shape[0])
            loss_total = loss_total/uv_cnt
            print("LOCAL BA losee:", loss_total)
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        self.pointclouds_cntr[:3, :] = pointclouds_param[:3, :].detach()

        for i in range (0, len(loop_neighbor_list)-1):
            kf_idx = loop_neighbor_list[i]
            pose_update = poses_train[:, :, i]
            self.KF_poses[:3, :, kf_idx] = pose_update.detach()
        # for hard_close_kf in hard_close_candidate:
        #     loop_neighbor_list = self.KF_loop_list[hard_close_kf].copy()
        #     if len(loop_neighbor_list) > 0:
        #         loop_neighbor_list.sort()
        #         self.LoopMatch2D2D(hard_close_kf, loop_neighbor_list[0], True)
        SP_poses = torch.index_select(self.KF_poses, 2, self.SP_index_list)
        self.SP_pose = SP_poses[:, :, -1].detach()
        print("LOOP BA ENDS")
        return [Result_GMapping, Result_First_KF, Result_BA], [], [], SP_poses.detach().cpu()
    def FullBundleAdjustment(self, iteration_num):
        Result_GMapping = False
        Result_First_KF = False
        Result_BA = True

        if len(self.KF_match_uv_list) < 5:
            return [False, False, False], [], [], []
        # else:
        #     self.BA_iteration += 1
        # if self.BA_iteration > 10:
        #     self.BA_iteration = 0
        # else:
        #     return [False, False, False], [], [], []

        pointclouds = self.pointclouds_cntr.detach()

        ones = torch.ones((1, pointclouds.shape[1]), dtype=torch.float32, device=self.device)
        pointclouds_param = nn.Parameter(
            torch.cat((pointclouds[:3, :].detach(), ones), dim=0).requires_grad_(True))
        poses_train = nn.Parameter(self.KF_poses[:3, :, 1:].detach().requires_grad_(True))

        pointclouds_lr = 1e-6
        poses_lr = 1e-5
        l = [
            {'params': [pointclouds_param], 'lr': pointclouds_lr, "name": "pointclouds"},
            {'params': [poses_train], 'lr': poses_lr, "name": "poses"}
        ]
        print("FULL BA BEGINS")
        optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-8)
        for iteration in range(iteration_num):
            uv_cnt = 0
            loss_total = 0
            poses = torch.cat((self.KF_poses[:3, :, 0].detach().unsqueeze(dim=2), poses_train), dim=2)
            pose_last = torch.eye((4), dtype=torch.float32,
                                  device=self.device)[3:4, :].unsqueeze(dim=2).repeat(1, 1, poses.shape[2])
            poses_four = torch.cat((poses, pose_last), dim=0)

            for i in range(0, len(self.KF_match_uv_list)):
                match_uv = self.KF_match_uv_list[i]
                if match_uv.shape[1] == 0:
                    continue
                recon_index = self.KF_recon_index_list[i].detach()
                # print("recon_index", recon_index)
                pointclouds_ptr_indice = recon_index[match_uv[0, :], match_uv[1, :]]
                pointcloud_indice = torch.index_select(self.pointclouds_ptr, 1, pointclouds_ptr_indice).squeeze()
                pointcloud_seen_from_kf = torch.index_select(pointclouds_param, 1, pointcloud_indice)

                pointclouds_cntr = torch.index_select(pointclouds[6, :].unsqueeze(dim=0), 1, pointcloud_indice)
                cntr_mask = pointclouds_cntr[0, :] > 3
                pointcloud_seen_from_kf_ctr = pointcloud_seen_from_kf[:, cntr_mask]
                match_uv_ctr = match_uv[:, cntr_mask]

                pose_from_kf = poses_four[:, :, i]
                world_to_kf = torch.inverse(pose_from_kf)

                cam_xyz = torch.matmul(world_to_kf, pointcloud_seen_from_kf_ctr)[:3, :]
                cam_uv = torch.matmul(self.intr, cam_xyz)
                mask = cam_uv[2, :].ne(0)
                cam_uv_mask = cam_uv[:, mask]
                cam_uv_mask = cam_uv_mask / cam_uv_mask[2, :]  # projection 한 uv
                match_uv_mask = match_uv_ctr[:, mask]  # 원본 uv
                loss = torch.norm((match_uv_mask - cam_uv_mask[:2, :]), dim=0)
                sorted_loss, _ = loss.sort(dim=0)
                inlier_num_min = int(sorted_loss.shape[0] * 0.0)
                inlier_num_max = int(sorted_loss.shape[0] * 0.9)
                loss_kf = torch.sum(sorted_loss[inlier_num_min:inlier_num_max])
                # print(i, inlier_num_max - inlier_num_min, loss_kf, "KF loss: ", loss_kf/(inlier_num_max - inlier_num_min))
                loss_total += loss_kf
                uv_cnt += (inlier_num_max - inlier_num_min)

                # loss_total += torch.sum(loss)
                # uv_cnt += int(cam_uv_mask.shape[1])
                # loss_mask = loss[loss.lt(10.0)]
                # loss_total += torch.sum(loss_mask)
                # uv_cnt += int(loss_mask.shape[0])
            loss_total = loss_total/uv_cnt
            print("FULL BA losee:", loss_total)
            loss_total.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        self.pointclouds_cntr[:3, :] = pointclouds_param[:3, :].detach()
        self.KF_poses[:3, :, 1:] = poses_train[:3, :, :].detach()
        SP_poses = torch.index_select(self.KF_poses, 2, self.SP_index_list)
        self.SP_pose = SP_poses[:, :, -1].detach()
        print("FULL BA ENDS")
        return [Result_GMapping, Result_First_KF, Result_BA], [], [], SP_poses.detach().cpu()

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
        current_kp, current_des = self.orb_cuda.detectAndComputeAsync(self.Current_gray_gpuMat, None)
        bow_desc = self.bowDiction.compute(gray_img, self.orb_cuda.convert(current_kp))
        self.KF_bow_list.append(bow_desc)

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


            # 현 키프레임 이미지와, 새로운 pose를 저장 한다.
            relative_pose = tracking_result[2]
            rot = relative_pose[0]
            quat = relative_pose[1]
            tvec = relative_pose[2]

            with torch.no_grad():
                KF_relative_pose = torch.eye(4, dtype=torch.float32, device=self.device)
                KF_relative_pose[:3, :3] = torch.from_numpy(rot)
                KF_relative_pose[:3, 3] = torch.from_numpy(tvec).squeeze()

                Prev_KF_pose= self.KF_poses[:, :, -1].detach()

                KF_current_pose = torch.matmul(Prev_KF_pose, torch.inverse(KF_relative_pose))
                current_recon_index = self.ComputeCovisibility((current_kp, current_des))
                self.CreateKeyframe(rgb_img, KF_xyz, (current_kp, current_des), KF_current_pose, current_recon_index)
            loop_detected = self.DetectLoop(len(self.KF_bow_list)-1)
            BA_result = None
            if loop_detected:
                current_covis_list = self.KF_loop_list[len(self.KF_bow_list)-1].copy()
                current_covis_list.sort()
                self.LoopMatch2D2D(len(self.KF_bow_list)-1, current_covis_list[0], True)
                print("loop_detected")
                self.LoopBundleAdjustment(100)[3]
                BA_val = self.FullBundleAdjustment(10)
                if BA_val[0][2]:
                    Result_BA = True
                    BA_result = BA_val[3]
                KF_current_pose = self.KF_poses[:, :, -1].detach()
                ## TODO: Local BA
            with torch.no_grad():
                if self.CheckSuperPixelFrame(KF_current_pose):
                    self.SP_pose = KF_current_pose.detach()
                    self.SP_index_list = torch.cat(
                        (self.SP_index_list, torch.tensor([self.KF_poses.shape[2] - 1], dtype=torch.int32,
                                                          device=self.device)), dim=0)
                    # prev_pose = self.KF_pose_list[-2]
                    # point_list_for_gaussian = self.TMPConvertCamera2World(ref_3d_list, prev_pose) # 이전 프레임의, Camera 스페이스 xyz임. / 가우시안에 corner점도 추가할 까 해서 넣은 것
                    return [Result_GMapping, Result_First_KF, Result_BA], [rgb_img, KF_xyz], KF_current_pose.detach().cpu(), BA_result
                else:  # Gaussian Mapping is not required
                    Result_GMapping = False
                    Result_First_KF = False
                    return [Result_GMapping, Result_First_KF, Result_BA], [], [], BA_result

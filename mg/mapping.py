
from plyfile import PlyData, PlyElement
import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv
from superpixel import SuperPixelManager
import collections
from scene.cameras import Camera
from arguments import PipelineParams
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
class Mapper:

    def Rot2Quat(self, rot):
        trace = rot[0][0] + rot[1][1] + rot[2][2]
        quaternion = np.empty((1, 4), dtype=np.float32)  # xyz|w
        if trace > 0.0:
            s = math.sqrt(trace + 1.0)
            quaternion[0][3] = s * 0.5
            s = 0.5 / s
            quaternion[0][0] = s * (rot[2][1] - rot[1][2])
            quaternion[0][1] = s * (rot[0][2] - rot[2][0])
            quaternion[0][2] = s * (rot[1][0] - rot[0][1])
        else:
            i = (2 if rot[1][1] < rot[2][2] else 1) if rot[0][0] < rot[1][1] else (2 if rot[0][0] < rot[2][2] else 0)
            j = (i + 1) % 3
            k = (i + 2) % 3
            s = math.sqrt(rot[i][i] - rot[j][j] - rot[k][k] + 1.0)
            quaternion[0][i] = s * 0.5
            s = 0.5 / s
            quaternion[0][3] = s * (rot[k][j] - rot[j][k])
            quaternion[0][j] = s * (rot[j][i] + rot[i][j])
            quaternion[0][k] = s * (rot[k][i] + rot[i][k])
        return quaternion

    def QuaternionInfo(self, quaternion):
        axis = np.array((quaternion[0][0], quaternion[0][1], quaternion[0][2]), dtype = np.float32)
        axis = axis / np.linalg.norm(axis)
        angle = math.acos(quaternion[0][3]) * 2.0
        return axis, angle

    def InfoBTWQuats(self, query, ref):
        # query x ref-1
        quaternion = np.zeros((1, 4), dtype=np.float32)  # xyz|w

        query_w = query[0][3]
        ref_w = ref[0][3]

        inv_ref = -ref[0]
        inv_ref[3] = ref_w

        inv_ref_vec = inv_ref[:3]
        query_vec = query[0][:3]

        print("inv")
        print(inv_ref)
        print(query[0])

        cross = np.cross(query_vec, inv_ref_vec)

        vec = cross + ref_w * query_vec + query_w * inv_ref_vec
        dotpro = np.dot(query[0], inv_ref)

        quaternion[0][3] = ref_w * query_w - dotpro
        quaternion[0][:3] = vec

        quaternion[0] = quaternion[0] / np.linalg.norm(quaternion[0])
        return self.QuaternionInfo(quaternion)

    def __init__(self):
        self.device = "cuda"

        # from images
        self.KF_gray_list = []
        self.KF_rgb_list = []
        self.KF_xyz_list = []   # Converted from Depth map
        self.KF_pose_list = []  #
        self.KF_superpixel_list = []

        # points (2D, 3D)
        self.KF_ref_3d_list = []
        self.KF_ref_color_list = []
        self.KF_ref_2d_list = []
        self.KF_query_2d_list = []
        self.KF_ref_sp_3d_list = []
        self.KF_ref_sp_color_list = []

        # super pixel pose
        self.SP_pose = None

        self.iteration = 0
    def CreateKeyframe(self, rgb, gray, KF_xyz, keyframe_pose):
        self.KF_rgb_list.append(rgb)
        self.KF_gray_list.append(gray)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_pose_list.append(keyframe_pose)

        # super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
        # self.KF_superpixel_list.append(super_pixel_index)
        # sp_3d_list = []
        # sp_color_list = []
        # for sp_index in super_pixel_index:
        #     int_sp_y = int(sp_index[0])
        #     int_sp_x = int(sp_index[1])
        #     if (int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
        #         continue
        #     sp_3d_list.append(KF_xyz[int_sp_y][int_sp_x])
        #     sp_color_list.append(rgb[int_sp_y][int_sp_x])
        # sp_3d_list = np.array(sp_3d_list)
        # sp_color_list = np.array(sp_color_list)
        #
        # point_list_for_gaussian = self.TMPConvertCamera2World(sp_3d_list, keyframe_pose)
        # self.gaussian.AddGaussian(point_list_for_gaussian, sp_color_list)
        # self.KF_ref_sp_3d_list.append(sp_3d_list)
        # self.KF_ref_sp_color_list.append(sp_color_list)
        # # print(f"sp color: {sp_color_list.shape}")
        # # print(f"sp ga: {point_list_for_gaussian.shape}")
        # self.gaussian.InitializeOptimizer()

    def StorPly(self, xyz_array, color_array, ply_path):
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                 ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        normals = np.zeros_like(xyz_array)
        color_list = color_array

        # print(f'xyz: {xyz_array.shape}')
        # print(f'normal: {normals.shape}')
        # print(f'rgb: {color_array.shape}')

        elements = np.empty(xyz_array.shape[0], dtype=dtype)
        attributes = np.concatenate((xyz_array, normals, color_list), axis=1)
        elements[:] = list(map(tuple, attributes))
        #
        vertex_element = PlyElement.describe(elements, 'vertex')

        ply_data = PlyData([vertex_element])
        print(ply_data['vertex'])
        print('------')
        print(ply_data.elements[0])
        # pcd = fetchPly(ply_path)
        ply_data.write(ply_path)


    def TMPBuildPointCloudAfterDone(self):
        path = "c:/lab/research/dataset/ply_test/"
        print( f'Pose: {len(self.KF_pose_list)} 3D: {len(self.KF_ref_3d_list)} Color: {len(self.KF_ref_color_list)}')
        pointcloud_xyz = np.empty((0, 3))
        pointcloud_rgb = np.empty((0, 3))
        for i in range (len(self.KF_ref_3d_list)):
            pose = self.KF_pose_list[i]
            point_3d_list = self.KF_ref_3d_list[i]
            color_3d_list = self.KF_ref_color_list[i]

            ones_list = np.expand_dims(np.ones(point_3d_list.shape[0]), axis=1)
            point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
            point_3d_list = np.matmul(point_3d_list, pose.T)  # Transpose!
            point_3d_list = point_3d_list[:, :] / point_3d_list[:, [-1]]

            pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
            pointcloud_rgb = np.concatenate((pointcloud_rgb, color_3d_list), axis=0)

        ply_path = f'{path}points3D.ply'
        self.StorPly(pointcloud_xyz, pointcloud_rgb, ply_path)
    def TMPConvertCamera2World(self, point_3d_list, pose):
        pointcloud_xyz = np.empty((0, 3), dtype=np.float32)
        ones_list = np.expand_dims(np.ones(point_3d_list.shape[0], dtype=np.float32), axis=1)
        point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
        point_3d_list = np.matmul(point_3d_list, pose.T)  # Transpose!
        point_3d_list = point_3d_list[:, :] / point_3d_list[:, [-1]]

        pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
        return pointcloud_xyz

    def Map(self, tracking_result_instance):
        if not tracking_result_instance[0]:
            return ([False])
        rgb = tracking_result_instance[2]
        gray = tracking_result_instance[3]
        KF_xyz = tracking_result_instance[4]
        keyframe_pose = tracking_result_instance[5]
        if not tracking_result_instance[1]: #First KF
            # First Keyframe
            self.CreateKeyframe(rgb, gray, KF_xyz, keyframe_pose)
            self.SP_pose = keyframe_pose
            return ([True, False, rgb, gray, KF_xyz, keyframe_pose])
        else:
            # 이전 키프레임을 기준으로 한 point들을 저장한다.
            # 현재 키프레임과 이전 키프레임 사이에서 생성된 point들인데, origin은 이전 것을 기준으로 함.
            ref_2d_list = tracking_result_instance[6]
            ref_3d_list = tracking_result_instance[7]
            ref_color_list = tracking_result_instance[8]
            query_2d_list = tracking_result_instance[9]

            self.KF_ref_2d_list.append(ref_2d_list)
            self.KF_ref_color_list.append(ref_color_list)
            self.KF_query_2d_list.append(query_2d_list)
            self.KF_ref_3d_list.append(ref_3d_list)

            # 현 키프레임 이미지와, 새로운 pose를 저장한다.
            self.CreateKeyframe(rgb, gray, KF_xyz, keyframe_pose)
            if self.CheckSuperPixelFrame(keyframe_pose):
                self.SP_pose = keyframe_pose
                return ([True, True, rgb, gray, KF_xyz, keyframe_pose])
            else:
                return ([False])

    def CheckSuperPixelFrame(self, pose):
        trace = np.dot(self.SP_pose, inv(pose))
        val = trace[0][0] + trace[1][1] + trace[2][2]
        angle = math.acos((val-1)/2)

        ref_t = self.SP_pose[:3, 3]
        query_t = pose[:3, 3]

        shift = np.linalg.norm((ref_t - query_t).T)
        if(angle > 0.5 or shift > 0.5):
            return True
        else:
            return False
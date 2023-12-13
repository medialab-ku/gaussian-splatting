from plyfile import PlyData, PlyElement
import numpy as np
import math
import torch
import torch.nn.functional as f
from scipy.spatial.transform import Rotation
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import time
from datetime import datetime

from numpy.linalg import inv
from associate import AssociateRun

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import shutil

import cv2


def BuildProject(project_name):
    path = 'c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/'
    if os.path.isdir(f'{path}{project_name}'):
        shutil.rmtree(f'{path}{project_name}')
    os.mkdir(f'{path}{project_name}') # Create project directory

    os.mkdir(f'{path}{project_name}/images')
    os.mkdir(f'{path}{project_name}/sparse')
    os.mkdir(f'{path}{project_name}/sparse/0')
    os.mkdir(f'{path}{project_name}/superpixel')

def SaveResultData(rgb_d_pair, pointcloud_xyz, pointcloud_rgb, result_file_index_list, result_quaternion_list,
               result_tvec_list, project_name):
    path = f'c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/{project_name}'
    list_len = len(result_file_index_list)

    # images.txt
    ## First frame
    rgb_path = 'c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household'
    src = f'{rgb_path}/{rgb_d_pair[0][0]}'
    dst = f'{path}/images/{rgb_d_pair[0][0][4:]}'
    shutil.copyfile(src, dst)

    images_txt = f'{1} {1.0} {0.0} {0.0} {0.0} {0.0} {0.0} {0.0} {1} {rgb_d_pair[0][0][4:]}\n\n'
    ## Rest frames
    for i in range (list_len):
        qx = result_quaternion_list[i][0][0]
        qy = result_quaternion_list[i][0][1]
        qz = result_quaternion_list[i][0][2]
        qw = result_quaternion_list[i][0][3]
        x = result_tvec_list[i][0][0]
        y = result_tvec_list[i][1][0]
        z = result_tvec_list[i][2][0]
        file_index = result_file_index_list[i]
        images_txt += f'{i+2} {qw} {qx} {qy} {qz} {x} {y} {z} {i+2} {rgb_d_pair[file_index][0][4:]}\n\n'

        src = f'{rgb_path}/{rgb_d_pair[file_index][0]}'
        dst = f'{path}/images/{rgb_d_pair[file_index][0][4:]}'
        shutil.copyfile(src, dst)
    images_txt_path = f'{path}/sparse/0/images.txt'
    f = open(images_txt_path, "w")
    f.write(images_txt)
    f.close()

    # Camera.txt
    intr = CreateIntrinsics()
    fx = intr[0][0]
    cx = intr[0][2]
    fy = intr[1][1]
    cy = intr[1][2]
    w = 640
    h = 480
    cameras_txt = ""
    for i in range (list_len+1):
        index = i+1
        cameras_txt += f'{index} PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n'
    cameras_txt_path = f'{path}/sparse/0/cameras.txt'
    f = open(cameras_txt_path, "w")
    f.write(cameras_txt)
    f.close()

    ply_path = f'{path}/sparse/0/points3D.ply'
    StorPly(pointcloud_xyz, pointcloud_rgb, ply_path)


def Rot2Quat(rot):
    trace = rot[0][0] + rot[1][1] + rot[2][2]
    quaternion = np.empty((1,4)) # xyz|w
    if trace > 0.0:
        s = math.sqrt(trace + 1.0)
        quaternion[0][3] = s * 0.5
        s = 0.5/s
        quaternion[0][0] = s * (rot[2][1] - rot[1][2])
        quaternion[0][1] = s * (rot[0][2] - rot[2][0])
        quaternion[0][2] = s * (rot[1][0] - rot[0][1])
    else:
        i = (2 if rot[1][1] < rot[2][2] else 1) if rot[0][0] < rot[1][1] else (2 if rot[0][0] < rot[2][2] else 0)
        j = (i + 1) % 3
        k = (i + 2) % 3
        s = math.sqrt(rot[i][i] - rot[j][j] - rot[k][k] + 1.0)
        quaternion[0][i] = s * 0.5
        s = 0.5/s
        quaternion[0][3] = s * (rot[k][j] - rot[j][k])
        quaternion[0][j] = s * (rot[j][i] + rot[i][j])
        quaternion[0][k] = s * (rot[k][i] + rot[i][k])
    return quaternion

def QuaternionInfo(quaternion):
    axis = np.array((quaternion[0][0],quaternion[0][1],quaternion[0][2]))
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(quaternion[0][3]) * 2.0
    return axis, angle
def GenerateUVTensor():
    width = 640
    height = 480
    u = torch.arange(width)
    for i in range (height-1):
        u = torch.vstack((u, torch.arange(width)))

    v = torch.tile(torch.arange(height), (1, 1)).T
    for i in range (width-1):
        v = torch.hstack((v, torch.tile(torch.arange(height), (1, 1)).T))

    uv = torch.stack((u, v), dim = 2)

    return uv

def CreateIntrinsics():
    fx = 535.4
    fy = 539.2
    cx = 320.1
    cy = 247.6

    intr = np.zeros((3, 3))
    intr[0][0] = fx
    intr[0][2] = cx
    intr[1][1] = fy
    intr[1][2] = cy
    intr[2][2] = 1

    return intr

def CreateInvIntrinsics():
    fx = 535.4
    fy = 539.2
    cx = 320.1
    cy = 247.6

    inv_intr = torch.zeros(3, 3)
    inv_intr[0][0] = 1/fx
    inv_intr[0][2] = -cx/fx
    inv_intr[1][1] = 1/fy
    inv_intr[1][2] = -cy/fy
    inv_intr[2][2] = 1
    return inv_intr

def RotationFromQuat(quat, device):
    axis = f.normalize(quat[0:3], dim = 0)
    x, y, z, q3 = torch.unbind(quat)
    q0, q1, q2 = torch.unbind(axis)
    extr_r = torch.stack(
        (
            2 * (q0 * q0 + q1 * q1) - 1,
            2 * (q1 * q2 - q0 * q3),
            2 * (q1 * q3 + q0 * q2),
            2 * (q1 * q2 + q0 * q3),
            2 * (q0 * q0 + q2 * q2) - 1,
            2 * (q2 * q3 - q0 * q1),
            2 * (q1 * q3 - q0 * q2),
            2 * (q2 * q3 + q0 * q1),
            2 * (q0 * q0 + q3 * q3) - 1
        ),
        -1,
    ).reshape(3, 3)
    extr_r.to(device)
    return extr_r


def StorPly(xyz_array, color_array, ply_path):
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
    # ply_path = f'C:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/polygon/orb/ply_{name}_{num}.ply'
    vertex_element = PlyElement.describe(elements, 'vertex')

    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

def ReadRGBDImgPair(rgb_d_pair, cnt):
    path = "c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/"
    count = 0
    img_pair=[]
    rgb_list=[]
    gray_list=[]
    d_list=[]
    for a in rgb_d_pair:
        rgb = cv2.imread(f'{path}{a[0]}')
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        gray = cv2.imread(f'{path}{a[0]}', cv2.IMREAD_GRAYSCALE)
        d = cv2.imread(f'{path}{a[1]}', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        count += 1
        img_pair.append((rgb,d))
        rgb_list.append(rgb)
        gray_list.append(gray)
        d_list.append(d)
        if(count >= cnt):
            break
    return img_pair, rgb_list, gray_list, d_list

def ComputeORBFromList(rgb_img_list, orb):
    orb_list = []
    for img in rgb_img_list:
        kp, des = orb.detectAndCompute(img, None)
        orb_list.append((kp, des))
    return orb_list


def RecoverXYZFromKeyFrame(query_kf, uv, inv_intr, device):

    scale_factor = 5000.0
    ones = torch.ones((uv.shape[0], uv.shape[1], 1)).to(device)
    uv_one = torch.cat((uv, ones), dim=2)
    uv_one = torch.unsqueeze(uv_one, dim=2)

    xy_one = torch.tensordot(uv_one, inv_intr, dims=([3], [1])).squeeze()

    d = query_kf.unsqueeze(dim=2)
    d = d / scale_factor

    xyz = torch.mul(xy_one, d)
    # xyz = torch.flatten(xyz, start_dim=0, end_dim=1).to(device)
    return xyz


def GetSuperPixelIndex(img, save, project_name, REGION_SIZE, RULER):
    N = 30

    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=102, region_size=REGION_SIZE, ruler=RULER)
    slic.iterate(N)

    slic.enforceLabelConnectivity()
    lbls = slic.getLabels()
    num_slic = slic.getNumberOfSuperpixels()

    indices = []
    # sample_idxs = np.random.choice(np.arange(num_slic), size=SAMPLE_SIZE, replace=False)
    for cls_lbl in range(num_slic):
        fst_cls = np.argwhere(lbls == cls_lbl)
        y, x = fst_cls[:, 0], fst_cls[:, 1] # x: 가로, y: 세로
        indices.append((y.mean(), x.mean()))

    if save:
        path = 'c:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/'
        sp_path = f'{path}{project_name}/superpixel'

        lsc_mask = slic.getLabelContourMask()
        cv2.imwrite(f'{sp_path}/mask.png', lsc_mask)
        # cv2.imshow('mask', lsc_mask)

        color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        color_img[:] = (0, 0, 255)
        mask_inv = cv2.bitwise_not(lsc_mask)
        result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
        result_fg = cv2.bitwise_and(color_img, color_img, mask=lsc_mask)
        result = cv2.add(result_bg, result_fg)
        cv2.imwrite(f'{sp_path}/result.png', result)
        # cv2.imshow('ColorCodedWindow', result)
        # cv2.waitKey(0)

    return indices



def ORBMatchORBList(SEED_TYPE, bf, rgb_list, orb_list, d_list, match_cnt, uv, intr, inv_intr, device,
                    project_name, REGION_SIZE, RULER):
    '''
        orb_list[n][0] = kp
        orb_list[n][1] = des
    '''
    # keyframe_interval = 30
    kf_interval = 0
    depth_keyframe = torch.from_numpy(np.array(d_list[0], dtype=np.float16)).to(device)
    xyz_keyframe = np.array(RecoverXYZFromKeyFrame(depth_keyframe, uv, inv_intr, device))
    keyframe_pose = np.eye(4)

    ref_orb_list = orb_list[0]
    ref_rgb_list = rgb_list[0]
    ref_superpixel_list = []
    if SEED_TYPE > 0:
        ref_superpixel_list = GetSuperPixelIndex(ref_rgb_list, True, project_name, REGION_SIZE, RULER)

    pointcloud_xyz = np.empty((0,3))
    pointcloud_rgb = np.empty((0,3))
    result_file_index_list = []
    result_quaternion_list = []
    result_tvec_list = []

    for i in range(0, len(orb_list)-1, 1):
        kf_interval+=1
        # print(f'frame: {i+1}')
        matches = bf.match(orb_list[i+1][1], ref_orb_list[1])
        matches = sorted(matches, key=lambda x: x.distance)
        point_2d_list = []
        point_3d_list = []
        color_3d_list = []
        for j in matches[:match_cnt]:
            kf_idx = j.trainIdx  # i.trainIdx
            kf_x, kf_y = ref_orb_list[0][kf_idx].pt
            int_kf_x = int(kf_x)
            int_kf_y = int(kf_y)
            if(int_kf_y == 0 or int_kf_x == 0 or int_kf_y == 479 or int_kf_x == 639):
                continue
            # xyz = xyz_keyframe[int(kf_y)][int(kf_x)]
            # if (xyz[2] > 0.5 and xyz[2] <=3):
            point_3d_list.append(xyz_keyframe[int_kf_y][int_kf_x])
            color_3d_list.append(ref_rgb_list[int_kf_y][int_kf_x])
            # Append kf x, y list
            q_idx = j.queryIdx  # i.trainIdx
            x, y = orb_list[i+1][0][q_idx].pt
            point_2d_list.append(np.array([x, y]))
            # Append query x,y list
            # cv2.circle(rgb_list[i+1], (int(x), int(y)), 3, (255, 0, 0), 2)
        point_3d_list = np.array(point_3d_list)
        color_3d_list = np.array(color_3d_list)
        point_2d_list = np.array(point_2d_list)

        z_mask_0 = point_3d_list[:, 2] > 0.2
        point_3d_list = point_3d_list[z_mask_0]
        color_3d_list = color_3d_list[z_mask_0]
        point_2d_list = point_2d_list[z_mask_0]

        pnp_3d_list = np.copy(point_3d_list)
        pnp_2d_list = np.copy(point_2d_list)

        # print(point_3d_list.shape)
        z_mask_1 = pnp_3d_list[:, 2] > 0.5
        pnp_3d_list = pnp_3d_list[z_mask_1]
        pnp_2d_list = pnp_2d_list[z_mask_1]
        z_mask_2 = pnp_3d_list[:, 2] <= 3
        pnp_3d_list = pnp_3d_list[z_mask_2]
        pnp_2d_list = pnp_2d_list[z_mask_2]

        ret, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_3d_list, pnp_2d_list, intr,
                                                      distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999,
                                                      reprojectionError=1, iterationsCount=1000)
        rot, _ = cv2.Rodrigues(rvec)

        ##For log
        quat = Rot2Quat(rot)
        axis, angle =QuaternionInfo(quat)
        shift = np.linalg.norm(tvec[:3, 0].T)
        # or kf_interval < 5
        if(kf_interval<10):
            continue
        if(angle >= 0.2 or shift >= 0.5 or angle == 0.0 or shift == 0.0):
            print(f'REFINE {i+1} {angle} {shift}')
            ret, rvec, tvec, inliers = cv2.solvePnPRansac(pnp_3d_list, pnp_2d_list, intr,
                                                          distCoeffs=None, flags=cv2.SOLVEPNP_EPNP, confidence=0.9999,
                                                          reprojectionError=1, iterationsCount=10000)
            rot, _ = cv2.Rodrigues(rvec)
            quat = Rot2Quat(rot)
            axis, angle = QuaternionInfo(quat)
            shift = np.linalg.norm(tvec[:3, 0].T)
        if (angle >= 0.2 or shift >= 0.5 or angle == 0.0 or shift == 0.0):
            print(f'BREAK {i+1} {angle} {shift}')
            continue
        else:
            print(f'{i+1} {angle} {shift}')
            kf_interval = 0

            # Store 3D pointcloud
            if SEED_TYPE == 0: # orb only
                test = 1
                # ones_list = np.expand_dims(np.ones(point_3d_list.shape[0]), axis=1)
                # point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
                # point_3d_list = np.matmul(point_3d_list, keyframe_pose.T) #Transpose!
                # point_3d_list = point_3d_list[:, :]/point_3d_list[:, [-1]]
                #
                # pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
                # # print(pointcloud_xyz.shape)
                # pointcloud_rgb = np.concatenate((pointcloud_rgb, color_3d_list), axis=0)
                # # print(pointcloud_rgb.shape)
            elif SEED_TYPE != 0:
                ref_superpixel_list = GetSuperPixelIndex(ref_rgb_list, False, project_name, REGION_SIZE, RULER)
                superpixel_point_3d_list = []
                superpixel_color_3d_list = []
                for sp_index in ref_superpixel_list:
                    int_sp_y = int(sp_index[0])
                    int_sp_x = int(sp_index[1])
                    if(int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
                        continue
                    superpixel_point_3d_list.append(xyz_keyframe[int_sp_y][int_sp_x])
                    superpixel_color_3d_list.append(ref_rgb_list[int_sp_y][int_sp_x])
                if SEED_TYPE == 1: # super pixel only
                    point_3d_list = np.array(superpixel_point_3d_list)
                    color_3d_list = np.array(superpixel_color_3d_list)
                elif SEED_TYPE == 2: # hybrid
                    point_3d_list = np.concatenate((point_3d_list, np.array(superpixel_point_3d_list)), axis = 0)
                    color_3d_list = np.concatenate((color_3d_list, np.array(superpixel_color_3d_list)), axis = 0)

            #####################################################################################
            ones_list = np.expand_dims(np.ones(point_3d_list.shape[0]), axis=1)
            point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
            point_3d_list = np.matmul(point_3d_list, keyframe_pose.T)  # Transpose!
            point_3d_list = point_3d_list[:, :] / point_3d_list[:, [-1]]

            pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
            pointcloud_rgb = np.concatenate((pointcloud_rgb, color_3d_list), axis=0)

            keyframe_pose_new = np.eye(4)
            keyframe_pose_new[:3, :3] = rot
            keyframe_pose_new[:3, 3:4] = tvec
            keyframe_pose = np.dot(keyframe_pose, inv(keyframe_pose_new))

            inv_keyframe_pose = inv(keyframe_pose)
            rot_store = inv_keyframe_pose[:3, :3]
            tvec_store = inv_keyframe_pose[:3, 3:4]
            store_quat = Rot2Quat(rot_store)

            result_file_index_list.append(i+1)
            result_quaternion_list.append(store_quat)
            result_tvec_list.append(tvec_store)

            # Set as Keyframe
            depth_keyframe = torch.from_numpy(np.array(d_list[i + 1], dtype=np.float16)).to(device)
            xyz_keyframe = np.array(RecoverXYZFromKeyFrame(depth_keyframe, uv, inv_intr, device))
            ref_orb_list = orb_list[i + 1]
            ref_rgb_list = rgb_list[i + 1]


        # cv2.imshow('target', rgb_list[i+1])
        # cv2.waitKey(1000)

    # StorPly(pointcloud_xyz, pointcloud_rgb, match_cnt, "kf_interval_10")
    return(pointcloud_xyz, pointcloud_rgb, result_file_index_list, result_quaternion_list, result_tvec_list)



project_name="orb"
SEED_TYPE = 0 # 0: Orb only, 1: Super pixel only, 2: Hybrid
REGION_SIZE = 0
RULER = 0

rgb_cnt = 550
match_cnt = 100
BuildProject(f'{project_name}_{REGION_SIZE}_{RULER}')
device = 'cpu'
uv = GenerateUVTensor().to(device)
inv_intr = CreateInvIntrinsics()
intr = CreateIntrinsics()

np_quat = np.array([1.0, 0.0, 0.0, 0.0])
extr_quat = torch.tensor(np_quat, dtype = torch.float32, requires_grad=True, device=device)
extr_t = torch.zeros(1, 3, requires_grad=True, device=device)

rgb_d_pair = AssociateRun()
rgbd_pair, rgb_list, gray_list, d_list = ReadRGBDImgPair(rgb_d_pair, rgb_cnt)

orb = cv2.ORB_create(
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
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
orb_list = ComputeORBFromList(gray_list, orb)
pointcloud_xyz, pointcloud_rgb, result_file_index_list, result_quaternion_list, result_tvec_list = \
    ORBMatchORBList(SEED_TYPE, bf, rgb_list, orb_list, d_list, match_cnt, uv, intr, inv_intr, device,
                    f'{project_name}_{REGION_SIZE}_{RULER}', REGION_SIZE, RULER)

SaveResultData(rgb_d_pair, pointcloud_xyz, pointcloud_rgb, result_file_index_list, result_quaternion_list,
               result_tvec_list, f'{project_name}_{REGION_SIZE}_{RULER}')


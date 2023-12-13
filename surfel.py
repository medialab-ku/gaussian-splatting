from plyfile import PlyData, PlyElement
from pathlib import Path
import shutil

import numpy as np
import OpenEXR as exr
import Imath

# import open3d as o3d
from associate import AssociateRun

from scene.colmap_loader import read_extrinsics_binary, read_intrinsics_binary
from scene.dataset_readers import readColmapCameras
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2


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


def read_img_file(path):
    return cv2.imread(path)

def ComputeSuperPixel(path):
    img =  cv2.imread(path)
    REGION_SIZE = 200
    RULER = 100
    N = 50
    SAMPLE_SIZE = 5
    INTENSITY_TH = 100

    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=102, region_size=REGION_SIZE, ruler=RULER)
    slic.iterate(N)


    slic.enforceLabelConnectivity()
    lbls = slic.getLabels()
    num_slic = slic.getNumberOfSuperpixels()


    lsc_mask = slic.getLabelContourMask()
    cv2.imshow('mask', lsc_mask)

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color_img[:] = (0, 0, 255)
    mask_inv = cv2.bitwise_not(lsc_mask)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=lsc_mask)
    result = cv2.add(result_bg, result_fg)
    cv2.imshow('ColorCodedWindow', result)
    cv2.waitKey(0)

    # sample_idxs = np.random.choice(np.arange(num_slic), size=SAMPLE_SIZE, replace=False)
    for cls_lbl in range(num_slic):
        fst_cls = np.argwhere(lbls == cls_lbl)
        x, y = fst_cls[:, 0], fst_cls[:, 1]
        c = (x.mean(), y.mean())
        # print(f'class {cls_lbl} centroid coordinates: ({c[0]:.1f}, {c[1]:.1f})')



def Get3DPointOfSuperPixel(img, depth, indicies):
    for index in indicies:
        u = index[0]
        v = index[1]
        z = depth[v, u]
        c = img [v, u]

def read_depth_exr_file(path):
    d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(d)
    # for val in d:
    #     print(val)
    cv2.imshow("d", d)
    cv2.waitKey(0)

def Compute3DPoint(u, v, z):
    fx = 507.124847412109
    fy = 507.124847412109
    cx = 317.184417724609
    cy = 178.029098510742
    x = z * (u - cx)/fx
    y = z * (v - cy)/fy

    return x, y, z

def GetSuperPixelIndex(img):
    REGION_SIZE = 200
    RULER = 100
    N = 50
    SAMPLE_SIZE = 5
    INTENSITY_TH = 100

    slic = cv2.ximgproc.createSuperpixelSLIC(img, algorithm=102, region_size=REGION_SIZE, ruler=RULER)
    slic.iterate(N)

    slic.enforceLabelConnectivity()
    lbls = slic.getLabels()
    num_slic = slic.getNumberOfSuperpixels()

    lsc_mask = slic.getLabelContourMask()
    cv2.imshow('mask', lsc_mask)

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    color_img[:] = (0, 0, 255)
    mask_inv = cv2.bitwise_not(lsc_mask)
    result_bg = cv2.bitwise_and(img, img, mask=mask_inv)
    result_fg = cv2.bitwise_and(color_img, color_img, mask=lsc_mask)
    result = cv2.add(result_bg, result_fg)
    cv2.imshow('ColorCodedWindow', result)
    cv2.waitKey(1)

    indices = []
    # sample_idxs = np.random.choice(np.arange(num_slic), size=SAMPLE_SIZE, replace=False)
    for cls_lbl in range(num_slic):
        fst_cls = np.argwhere(lbls == cls_lbl)
        y, x = fst_cls[:, 0], fst_cls[:, 1] # x: 가로, y: 세로
        indices.append((y.mean(), x.mean()))
    return indices


def Create3DPointFromSuperPixel(img, depth, confidence, scale_img_to_depth, R, T):
    # Get indices from super pixels
    indices = GetSuperPixelIndex(img)
    # print(f'super pixel number: {len(indices)}')
    xyz_list = []
    color_list = []
    for index in indices:
        # single_point = np.empty([],dtype=dtype)
        u = int(index[1]) # 가로
        v = int(index[0]) # 세로

        if u >= img.shape[1]:
            u = img.shape[1] - 1

        if v >= img.shape[0]:
            v = img.shape[0] - 1


        # Check confidence
        du = int(scale_img_to_depth * u)
        dv = int(scale_img_to_depth * v)
        # if confidence[dv, du] == 0:
        #     continue

        # Get color
        color = np.array(img[v, u])

        # Compute 3D coordinate
        z = depth[dv, du] * 1.7

        # Apply camera pose to the Computed 3D point
        # Compute3DPoint(u, v, z)
        # point3d = R.dot(np.array(Compute3DPoint(u, v, z))) + T
        point3d = (R).dot(np.array(Compute3DPoint(u, v, z)) - T)
        xyz_list.append(point3d)
        color_list.append(color)

    return xyz_list, color_list

def GetCameraPose():

    ex_bin_path = f'C:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/col_sp_10_50/sparse/0/images.bin'
    in_bin_path = f'C:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/col_sp_10_50/sparse/0/cameras.bin'

    cam_extrinsics = read_extrinsics_binary(ex_bin_path)
    cam_intrinsics = read_intrinsics_binary(in_bin_path)
    print(cam_intrinsics)

    img_path = f'C:/lab/research/dataset/arkit_diet/arkit_diet_rgb_colmap2/images'
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=img_path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    xyz_list=np.array([])
    color_list=np.array([])
    pointcloud =np.array([])
    first = True
    counter = 0
    for key in cam_infos:
        counter += 1
        print(f'frame: {counter}')
        R = key.R
        T = key.T
        print(R)
        print(T)
        name = key.image_name
        # print(f'R: {key.R} / T: {key.T} / name: {key.image_name}')

        img_path = f'C:/lab/research/dataset/arkit_diet/arkit_diet_rgb/{name}.jpg'
        depth_path = f'C:/lab/research/dataset/arkit_diet/arkit_diet_d/{name}.exr'
        confidence_path = f'C:/lab/research/dataset/arkit_diet/arkit_diet_confidence/{name}.exr'
        img = cv2.imread(img_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        depth2= cv2.imread(depth_path)
        confidence = cv2.imread(confidence_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        cv2.imshow("d", depth2*10)

        scale_img_to_depth = depth.shape[0] / img.shape[0]
        xyz, color = Create3DPointFromSuperPixel(img, depth, confidence, scale_img_to_depth, R, T)
        if first:
            xyz_list = np.array(xyz)
            color_list = np.array(color)
            first = False
        else:
            xyz_list = np.concatenate((xyz_list, xyz), axis = 0)
            color_list = np.concatenate((color_list, color), axis = 0)
        # if counter == 30:
        #     break

    return xyz_list, color_list

def StorePly(xyz_list, color_list):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    # xyz = xyz_list
    # rgb = color_list
    normals = np.zeros_like(xyz_list)

    elements = np.empty(xyz_list.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz_list, normals, color_list), axis=1)
    elements[:] = list(map(tuple, attributes))


    ply_path = f'C:/lab/research/dataset/arkit_diet/arkit_diet_rgb_colmap2/sparse/0/ply_customed.ply'
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)




    # for key in cam_infos:
    #     print(f'R: {key.R} / T: {key.T} / name: {key.image_name}')
        # rgb_map = cv2.imread(f'C:/lab/research/dataset/arkit_diet/arkit_diet_rgb/{key.image_name}.jpg')
        # depth_map = cv2.imread(f'C:/lab/research/dataset/arkit_diet/arkit_diet_d/{key.image_name}.exr')
        # rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_map, depth_map, convert_rgb_to_intensity=False)

    # 이 밑에 테스트코드 계속 작업
    # RGB이미지와 Depth이미지를 읽어서 rgbd를 만든다. -> 추후 슈퍼픽셀로 변환
    # 해당 이미지에 intrinsic을 넣어서 pcd를 만든다
    # 그 pcd에 extrinsic을 가미해서, world 좌표계로 끌고 온다.
    # 그것을 array에 넣는다.
    # 최종적으로 파일로 뽑은 뒤 meshlab에서 확인한다. (기존 ply와 함께 로드해서 좌표계가 맞는지 본다)

#
# rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity = False)
# pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic)
#
# # flip the orientation, so it looks upright, not upside-down
# pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])



xyz_list, color_list = point_cloud =GetCameraPose()

StorePly(xyz_list, color_list)
# print(pointcloud)

# for i in range (2201):
#     index = f'{i}'
#     if i < 10:
#         index = f'000{i}'
#     elif i < 100:
#         index = f'00{i}'
#     elif i < 1000:
#         index = f'0{i}'
#     else:
#         index = f'{i}'
#     #
#     # target_index = f'{i}'
#     # if 211-i < 10:
#     #     target_index = f'00{211-i}'
#     # elif 211-i < 100:
#     #     target_index = f'0{211-i}'
#     # else:
#     #     target_index = f'{211-i}'
#
#
#     src_path = f'C:/lab/research/dataset/arkit1/camera/{i}.jpg'
#     target_path = f'C:/lab/research/dataset/arkit_diet/arkit_full/input/{index}.jpg'
#     shutil.copyfile(src_path, target_path)

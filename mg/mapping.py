
from plyfile import PlyData, PlyElement
import cv2
import math
import numpy as np
import torch
from numpy.linalg import inv
from scene import Scene, GaussianModel
from superpixel import SuperPixelManager
import collections
from scene.cameras import Camera
from arguments import PipelineParams
from gaussian_renderer import mg_render
from argparse import ArgumentParser
from utils.loss_utils import l1_loss, ssim
class Mapper:
    def __init__(self):
        self.device = "cuda"
        self.projection_matrix = None
        self.SetProjectionMatrix()
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))

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

        # Super pixel
        self.sp_manager = SuperPixelManager()
        # Gaussians
        self.gaussian = GaussianModel(3, self.device)
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        # self.GaussianList

    def SetProjectionMatrix(self):
        fx = 535.4
        fy = 539.2

        FoVx = 2 * math.atan(640 / (2 * fx))
        FoVy = 2 * math.atan(480 / (2 * fy))
        self.FoVx = np.float32(FoVx)
        self.FoVy = np.float32(FoVy)

        self.projection_matrix = self.getProjectionMatrix(znear=0.01, zfar=100, fovX=FoVx, fovY=FoVy).transpose(0, 1).type(torch.FloatTensor).to(self.device)

    def getProjectionMatrix(self, znear, zfar, fovX, fovY):
        tanHalfFovY = math.tan((fovY / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = torch.zeros(4, 4, dtype=torch.float32, device=self.device)

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = z_sign
        P[2, 2] = z_sign * zfar / (zfar - znear)
        P[2, 3] = -(zfar * znear) / (zfar - znear)
        return P

    def CreateKeyframe(self, rgb, gray, KF_xyz, keyframe_pose):
        self.KF_rgb_list.append(rgb)
        self.KF_gray_list.append(gray)
        self.KF_xyz_list.append(KF_xyz)
        self.KF_pose_list.append(keyframe_pose)
        super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
        self.KF_superpixel_list.append(super_pixel_index)
        sp_3d_list = []
        sp_color_list = []
        for sp_index in super_pixel_index:
            int_sp_y = int(sp_index[0])
            int_sp_x = int(sp_index[1])
            if (int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
                continue
            sp_3d_list.append(KF_xyz[int_sp_y][int_sp_x])
            sp_color_list.append(rgb[int_sp_y][int_sp_x])
        sp_3d_list = np.array(sp_3d_list)
        sp_color_list = np.array(sp_color_list)

        point_list_for_gaussian = self.TMPConvertCamera2World(sp_3d_list, keyframe_pose)
        self.gaussian.AddGaussian(point_list_for_gaussian, sp_color_list)
        self.KF_ref_sp_3d_list.append(sp_3d_list)
        self.KF_ref_sp_color_list.append(sp_color_list)

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

        # plydata = PlyData.read(ply_path)
        # vertices = ply_data['vertex']
        # positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
        # colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        # normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
        # print('------v')
        # print(vertices)
        # print('------p')
        # print(positions)
        # print('------c')
        # print(colors)
        # print('------n')
        # print(normals)


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
            return
        rgb = tracking_result_instance[2]
        gray = tracking_result_instance[3]
        KF_xyz = tracking_result_instance[4]
        keyframe_pose = tracking_result_instance[5]
        if not tracking_result_instance[1]:
            # First Keyframe
            self.CreateKeyframe(rgb, gray, KF_xyz, keyframe_pose)
            return

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


            # ref_3d_list 카메라 스페이스다. 월드로 바꿔서 넣어야 함
            point_list_for_gaussian = self.TMPConvertCamera2World(ref_3d_list, self.KF_pose_list[-1])
            self.gaussian.AddGaussian(point_list_for_gaussian, ref_color_list)
            # 현 키프레임 이미지와, 새로운 pose를 저장한다.
            self.CreateKeyframe(rgb, gray, KF_xyz, keyframe_pose)

            self.RenderGaussian("viz", np.eye(4))

            return


    def RenderGaussian(self, title, pose):
        pose = torch.from_numpy(pose)
        world_view_transform = inv(pose)
        world_view_transform.astype(np.float32)
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center).to(self.device)
        world_view_transform = torch.from_numpy(world_view_transform)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix.type(torch.DoubleTensor)).unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device)
        world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        camera_center = camera_center.type(torch.FloatTensor).to(self.device)

        render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, world_view_transform, full_proj_transform, camera_center,
                               self.gaussian, self.pipe, self.background, 1.0)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        render_pkg["visibility_filter"], render_pkg["radii"]

        img_rgb = image.permute(1, 2, 0).detach().to("cpu").numpy()
        cv2.imshow(f"gaussian_{title}", img_rgb)
        cv2.waitKey(1)

        return render_pkg
    def TMPGAUSSIANRENDER(self):
        # Camera에서 아래 것이 만족해야한다.
        # 아래의 네가지가 gaussian_renderer로 넘어가야함.
        bg_color = [1, 1, 1]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # for i in range (len(self.KF_ref_3d_list)):
        #     print(f'Pose: {self.KF_pose_list[i]}')
        #     world_view_transform = inv(self.KF_pose_list[i])
        #     world_view_transform.astype(np.float32)
        #     camera_center = inv(world_view_transform)[3, :3]
        #     camera_center = torch.from_numpy(camera_center).to(self.device)
        #     world_view_transform = torch.from_numpy(world_view_transform)
        #
        #     full_proj_transform = (world_view_transform.unsqueeze(0).bmm((self.projection_matrix.type(torch.DoubleTensor)).unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(self.device)
        #     world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        #     camera_center = camera_center.type(torch.FloatTensor).to(self.device)
        #     # print(f'FoVX: {self.FoVx.dtype}')
        #     # print(f'FoVy: {self.FoVy.dtype}')
        #     # print(f'world_view_transform: {world_view_transform.dtype}')
        #     # print(f'full_proj_transform: {full_proj_transform.dtype}')
        #     # print(f'camera_center: {camera_center.dtype}')
        #     # print(f'background: {background.dtype}')
        #
        #     render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, world_view_transform, full_proj_transform, camera_center, self.gaussian, self.pipe, background, 1.0)
        #     image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        #     img_rgb = image.permute(1, 2, 0).detach().to("cpu").numpy()
        #     cv2.imshow("gaussian", img_rgb)
        #     cv2.waitKey(1)

        pose = torch.from_numpy(np.eye(4))
        world_view_transform = inv(pose)
        world_view_transform.astype(np.float32)
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center).to(self.device)
        world_view_transform = torch.from_numpy(world_view_transform)

        full_proj_transform = (world_view_transform.unsqueeze(0).bmm((self.projection_matrix.type(torch.DoubleTensor)).unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(self.device)
        world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        camera_center = camera_center.type(torch.FloatTensor).to(self.device)
        # print(f'FoVy: {self.FoVy.dtype}')
        # print(f'world_view_transform: {world_view_transform.dtype}')
        # print(f'full_proj_transform: {full_proj_transform.dtype}')
        # print(f'camera_center: {camera_center.dtype}')
        # print(f'background: {background.dtype}')

        render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, world_view_transform, full_proj_transform, camera_center, self.gaussian, self.pipe, background, 1.0)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        img_rgb = image.permute(1, 2, 0).detach().to("cpu").numpy()
        cv2.imshow("gaussian", img_rgb)
        cv2.waitKey(1)


    def OptimizeGaussian(self):
        for i in range(len(self.KF_pose_list)):
            pose = self.KF_pose_list[i]
            img_gt = torch.from_numpy(self.KF_rgb_list[i]).permute(2, 0, 1).to(self.device)/255
            render_pkg = self.RenderGaussian("opti", pose)
            img = render_pkg["render"]
            lambda_dssim = 0.2

            Ll1 = l1_loss(img, img_gt)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
            loss.backward()

            np_gt = img_gt.permute(1, 2, 0).detach().to("cpu").numpy()
            cv2.imshow("gt", np_gt )
            np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
            cv2.imshow("render", np_render )
            cv2.waitKey(1)



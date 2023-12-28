
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

class GaussianMapper:
    def __init__(self):
        self.device = "cuda"
        self.projection_matrix = None
        self.SetProjectionMatrix()
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))

        # from images
        self.SP_gray_list = []
        self.SP_rgb_list = []
        self.SP_xyz_list = []  # Converted from Depth map
        self.SP_pose_list = []  #
        self.SP_superpixel_list = []

        # points (2D, 3D)
        self.SP_ref_3d_list = []
        self.SP_ref_color_list = []

        # Super pixel
        self.sp_manager = SuperPixelManager()
        # Gaussians
        self.gaussian = GaussianModel(3, self.device)
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")

        self.iteration = 0

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

    def TMPConvertCamera2World(self, point_3d_list, pose):
        pointcloud_xyz = np.empty((0, 3), dtype=np.float32)
        ones_list = np.expand_dims(np.ones(point_3d_list.shape[0], dtype=np.float32), axis=1)
        point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
        point_3d_list = np.matmul(point_3d_list, pose.T)  # Transpose!
        point_3d_list = point_3d_list[:, :] / point_3d_list[:, [-1]]

        pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
        return pointcloud_xyz

    def CreateKeyframe(self, rgb, gray, SP_xyz, pose):
        self.SP_rgb_list.append(rgb)
        self.SP_gray_list.append(gray)
        self.SP_xyz_list.append(SP_xyz)
        self.SP_pose_list.append(pose)
        super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
        self.SP_superpixel_list.append(super_pixel_index)
        sp_3d_list = []
        sp_color_list = []
        for sp_index in super_pixel_index:
            int_sp_y = int(sp_index[0])
            int_sp_x = int(sp_index[1])
            if (int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
                continue
            sp_3d_list.append(SP_xyz[int_sp_y][int_sp_x])
            sp_color_list.append(rgb[int_sp_y][int_sp_x])
        sp_3d_list = np.array(sp_3d_list)
        sp_color_list = np.array(sp_color_list)

        point_list_for_gaussian = self.TMPConvertCamera2World(sp_3d_list, pose)
        self.gaussian.AddGaussian(point_list_for_gaussian, sp_color_list)
        self.SP_ref_3d_list.append(sp_3d_list)
        self.SP_ref_color_list.append(sp_color_list)
        self.gaussian.InitializeOptimizer()
        self.OptimizeGaussianSinglePose(pose, rgb)


    def RenderGaussian(self, title, pose):
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device)
        world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        camera_center = camera_center.type(torch.FloatTensor).to(self.device)

        render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, world_view_transform, full_proj_transform, camera_center, self.gaussian, self.pipe, self.background, 1.0)
        # image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], \
        # render_pkg["visibility_filter"], render_pkg["radii"]

        return render_pkg


    def OptimizeGaussian(self):
        if len(self.SP_pose_list) > 0:
            self.iteration+=1
            print(f"iter: {self.iteration} | SP cnt: {len(self.SP_pose_list)}")
            # render_pkg = self.RenderGaussian("viz", np.eye(4))
            # img = render_pkg["render"]
            # np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
            # cv2.imshow("start_gs", np_render)
            # cv2.waitKey(1)

        # self.gaussian.update_learning_rate(self.iteration)

        for i in range(len(self.SP_pose_list)):
            # for j in range(100):
            pose = self.SP_pose_list[i]
            img_gt = torch.from_numpy(self.SP_rgb_list[i]).permute(2, 0, 1).to(self.device)/255
            # for j in range(100):
            render_pkg = self.RenderGaussian("opti", pose)
            img = render_pkg["render"]
            lambda_dssim = 0.2

            Ll1 = l1_loss(img, img_gt)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
            # print(f"{i}th loss: {loss}")
            loss.backward()
            # if loss < 0.1:
            #     break
            # self.gaussian.GetGradient()
            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)



            # np_gt = img_gt.permute(1, 2, 0).detach().to("cpu").numpy()
            # cv2.imshow("gt", np_gt )
            # np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
            # cv2.imshow("render", np_render )
            # cv2.waitKey(1)

    def OptimizeGaussianSinglePose(self, pose, img_gt):
        img_gt = torch.from_numpy(img_gt).permute(2, 0, 1).to(self.device) / 255
        for i in range(50):
            render_pkg = self.RenderGaussian("opti", pose)
            img = render_pkg["render"]
            lambda_dssim = 0.2

            Ll1 = l1_loss(img, img_gt)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
            # print(f"{i}th loss: {loss}")
            loss.backward()
            if loss < 0.1:
                break
            # self.gaussian.GetGradient()
            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)

    def Visualize(self, pose):
        if len(self.SP_pose_list) > 0:
            pose = self.SP_pose_list[0]
            render_pkg = self.RenderGaussian("viz", pose)
            img = render_pkg["render"]
            np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
            cv2.imshow("start_gs", np_render )
            cv2.waitKey(1)
    def GaussianMap(self, mapping_result_instance):
        if not mapping_result_instance[0]:
            return
        rgb = mapping_result_instance[2]
        gray = mapping_result_instance[3]
        SP_xyz = mapping_result_instance[4]
        pose = mapping_result_instance[5]

        if not mapping_result_instance[1]: #First KF
            # First Keyframe
            self.CreateKeyframe(rgb, gray, SP_xyz, pose)

        else:
            self.CreateKeyframe(rgb, gray, SP_xyz, pose)


        # render_pkg = self.RenderGaussian("viz", np.eye(4))
        # img = render_pkg["render"]
        # np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
        # cv2.imshow("start_gs", np_render )
        # cv2.waitKey(1)

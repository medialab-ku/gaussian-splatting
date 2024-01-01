
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
import datetime

class GaussianMapper:
    def __init__(self):
        self.device = "cuda"
        self.projection_matrix = None
        self.SetProjectionMatrix()
        self.pipe = PipelineParams(ArgumentParser(description="Training script parameters"))

        self.intr = np.zeros((3, 3), dtype=np.float32)
        self.SetIntrinsics()
        self.full_proj_transform_list = []
        self.world_view_transform_list = []
        self.camera_center_list = []

        # from images
        self.SP_gray_list = []
        self.SP_rgb_list = []
        self.SP_img_gt_list = []
        self.SP_xyz_list = []  # Converted from Depth map
        self.SP_pose_list = []  #
        self.SP_superpixel_list = []

        # points (2D, 3D)
        self.SP_ref_3d_list = []
        self.SP_ref_color_list = []
        self.SP_global_3d_list = []

        # Super pixel
        self.sp_manager = SuperPixelManager()
        # Gaussians
        self.gaussian = GaussianModel(3, self.device)
        self.background = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        self.cam_centers = []
        self.cameras_extent = 0
        self.size_threshold = 20
        self.densify_grad_threshold = 0.001
        self.loss_threshold = 0.1

        self.iteration = 1
        # viz
        self.viz_full_proj_transform_list = []
        self.viz_world_view_transform_list = []
        self.viz_camera_center_list = []
        self.SetVizParams()




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

    def getNerfppNorm(self, pose):
        def get_center_and_diag():
            cam_centers = np.hstack(self.cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_center = pose[:3, 3:4]
        self.cam_centers.append(cam_center)

        center, diagonal = get_center_and_diag()
        radius = diagonal * 1.1

        translate = -center
        self.cameras_extent = radius
        return {"translate": translate, "radius": radius}

    def TMPConvertCamera2World(self, point_3d_list, pose):
        pointcloud_xyz = np.empty((0, 3), dtype=np.float32)
        ones_list = np.expand_dims(np.ones(point_3d_list.shape[0], dtype=np.float32), axis=1)
        point_3d_list = np.concatenate((point_3d_list, ones_list), axis=1)
        point_3d_list = np.matmul(point_3d_list, pose.T)  # Transpose!
        point_3d_list = point_3d_list[:, :] / point_3d_list[:, [-1]]

        pointcloud_xyz = np.concatenate((pointcloud_xyz, point_3d_list[:, :3]), axis=0)
        return pointcloud_xyz

    def CreateInitialKeyframe(self, rgb, gray, SP_xyz, pose):
        img_gt = (torch.from_numpy(rgb).permute(2, 0, 1)/255).to(self.device)
        self.SP_rgb_list.append(rgb)
        self.SP_gray_list.append(gray)
        self.SP_xyz_list.append(SP_xyz)
        self.SP_pose_list.append(pose)
        self.SP_img_gt_list.append(img_gt)
        super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
        sp_3d_list = []
        sp_2d_list = []
        sp_color_list = []
        for sp_index in super_pixel_index:
            int_sp_y = int(sp_index[0])
            int_sp_x = int(sp_index[1])
            if (int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
                continue
            sp_2d_list.append([int_sp_y, int_sp_x])
            sp_3d_list.append(SP_xyz[int_sp_y][int_sp_x])
            sp_color_list.append(rgb[int_sp_y][int_sp_x])

        sp_2d_list = np.array(sp_2d_list)
        sp_3d_list = np.array(sp_3d_list)
        sp_color_list = np.array(sp_color_list)
        z_mask_0 = sp_3d_list[:, 2] > 0.2
        sp_3d_list = sp_3d_list[z_mask_0]
        sp_2d_list = sp_2d_list[z_mask_0]
        sp_color_list = sp_color_list[z_mask_0]
        # self.SP_superpixel_list.append(sp_2d_list)



        point_list_for_gaussian = self.TMPConvertCamera2World(sp_3d_list, pose)
        self.gaussian.AddGaussian(point_list_for_gaussian, sp_color_list)
        self.SP_global_3d_list.append(point_list_for_gaussian)


        # self.SP_ref_3d_list.append(sp_3d_list)
        # self.SP_ref_color_list.append(sp_color_list)

        # pre-compute poses
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device)
        world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        camera_center = camera_center.type(torch.FloatTensor).to(self.device)

        self.full_proj_transform_list.append(full_proj_transform)
        self.world_view_transform_list.append(world_view_transform)
        self.camera_center_list.append(camera_center)

        # Gaussian
        self.gaussian.InitializeOptimizer()

    def SetVizParams(self):
        # pose = np.eye(4)
        # pose[2, 3] = 3
        # pose[0, 3] = -4
        #
        # pose[0, 0] = 0
        # pose[0, 2] = 1
        # pose[2, 0] = -1
        # pose[2, 2] = 0
        # # pre-compute poses
        # inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        # world_view_transform = inv_pose
        # camera_center = inv(world_view_transform)[3, :3]
        # camera_center = torch.from_numpy(camera_center)
        #
        # self.viz_full_proj_transform_list.append((world_view_transform.unsqueeze(0).bmm(
        #     (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
        #     self.device))
        # self.viz_world_view_transform_list.append(world_view_transform.type(torch.FloatTensor).to(self.device))
        # self.viz_camera_center_list.append(camera_center.type(torch.FloatTensor).to(self.device))
        #
        #
        #
        pose = np.eye(4)
        # pre-compute poses
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        self.viz_full_proj_transform_list.append((world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device))
        self.viz_world_view_transform_list.append(world_view_transform.type(torch.FloatTensor).to(self.device))
        self.viz_camera_center_list.append(camera_center.type(torch.FloatTensor).to(self.device))
        #
        #
        pose = np.eye(4)
        pose[2, 3] = 6
        pose[1, 3] = -4.5

        pose[0, 0] = -1
        pose[1, 1] = 0.5
        pose[1, 2] = 0.866
        pose[2, 1] = 0.866
        pose[2, 2] = -0.5

        # pre-compute poses
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        self.viz_full_proj_transform_list.append((world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device))
        self.viz_world_view_transform_list.append(world_view_transform.type(torch.FloatTensor).to(self.device))
        self.viz_camera_center_list.append(camera_center.type(torch.FloatTensor).to(self.device))

        pose = np.eye(4)
        pose[2, 3] = -5
        pose[1, 3] = -0.5
        # pre-compute poses
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        self.viz_full_proj_transform_list.append((world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device))
        self.viz_world_view_transform_list.append(world_view_transform.type(torch.FloatTensor).to(self.device))
        self.viz_camera_center_list.append(camera_center.type(torch.FloatTensor).to(self.device))



    def CreateKeyframe(self, rgb, gray, SP_xyz, pose, ref_3d_list, ref_color_list):
        self.gaussian.AddGaussian(ref_3d_list, ref_color_list)

        min_u = 640  # x
        min_v = 480  # y
        max_u = 0
        max_v = 0
        previous_SP_3d = self.TMPConvertCamera2World(self.SP_global_3d_list[-1], inv(pose))
        uv = np.dot(previous_SP_3d, self.intr.T)
        uv = uv[:, :]/ uv[:, [-1]]

        for uv_element in uv:
            if uv_element[0] < min_u:
                min_u = uv_element[0]
            if uv_element[1] < min_v:
                min_v = uv_element[1]
            if uv_element[0] > max_u:
                max_u = uv_element[0]
            if uv_element[1] > max_v:
                max_v = uv_element[1]

        w = max_u - min_u
        w_padding = (w * 0.15)
        h = max_v - min_v
        h_padding = h * 0.15
        min_u += w_padding
        max_u -= w_padding
        min_v += h_padding
        max_v -= h_padding

        if min_u < 0:
            min_u = 0
        if min_v < 0:
            min_v = 0
        if max_u > 639:
            max_u = 639
        if max_v > 479:
            max_v = 479
        img_gt = (torch.from_numpy(rgb).permute(2, 0, 1)/255).to(self.device)
        self.SP_rgb_list.append(rgb)
        self.SP_gray_list.append(gray)
        self.SP_xyz_list.append(SP_xyz)
        self.SP_pose_list.append(pose)
        self.SP_img_gt_list.append(img_gt)


        super_pixel_index = self.sp_manager.ComputeSuperPixel(rgb)
        # self.SP_superpixel_list.append(super_pixel_index)
        sp_3d_total_list = []
        sp_3d_list = []
        sp_color_list = []
        for sp_index in super_pixel_index:
            int_sp_y = int(sp_index[0])
            int_sp_x = int(sp_index[1])
            if (int_sp_y > max_v or int_sp_x > max_u or int_sp_y < min_v or int_sp_x < min_u):
                sp_3d_list.append(SP_xyz[int_sp_y][int_sp_x])
                sp_color_list.append(rgb[int_sp_y][int_sp_x])
                sp_3d_total_list.append(SP_xyz[int_sp_y][int_sp_x])
            elif (int_sp_y == 0 or int_sp_x == 0 or int_sp_y == 479 or int_sp_x == 639):
                continue
            else:
                sp_3d_total_list.append(SP_xyz[int_sp_y][int_sp_x])
        sp_3d_list = np.array(sp_3d_list)
        sp_color_list = np.array(sp_color_list)
        sp_3d_total_list = np.array(sp_3d_total_list)
        global_sp_3d_total = self.TMPConvertCamera2World(sp_3d_total_list, pose)
        self.SP_global_3d_list.append(global_sp_3d_total)

        if sp_3d_list.shape[0] > 0:
            z_mask_0 = sp_3d_list[:, 2] > 0.2
            sp_3d_list = sp_3d_list[z_mask_0]
            sp_color_list = sp_color_list[z_mask_0]
            point_list_for_gaussian = self.TMPConvertCamera2World(sp_3d_list, pose)
            self.gaussian.AddGaussian(point_list_for_gaussian, sp_color_list)
        # self.SP_ref_3d_list.append(sp_3d_list)
        # self.SP_ref_color_list.append(sp_color_list)

        # pre-compute poses
        inv_pose = torch.from_numpy(inv(pose).T).type(torch.FloatTensor)
        world_view_transform = inv_pose
        camera_center = inv(world_view_transform)[3, :3]
        camera_center = torch.from_numpy(camera_center)

        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(
            (self.projection_matrix).detach().to("cpu").unsqueeze(0))).squeeze(0).type(torch.FloatTensor).to(
            self.device)
        world_view_transform = world_view_transform.type(torch.FloatTensor).to(self.device)
        camera_center = camera_center.type(torch.FloatTensor).to(self.device)

        self.full_proj_transform_list.append(full_proj_transform)
        self.world_view_transform_list.append(world_view_transform)
        self.camera_center_list.append(camera_center)

        # Gaussian
        self.gaussian.InitializeOptimizer()



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
        lambda_dssim = 0.2
        # if len(self.SP_pose_list) > 0:
        #     self.iteration+=1

        # self.gaussian.update_learning_rate(self.iteration)
        flag = True
        for i in range(len(self.SP_pose_list)):
            #print(1, datetime.datetime.now().strftime("%H:%M:%S.%f"))
            img_gt = self.SP_img_gt_list[i]
            world_view_transform = self.world_view_transform_list[i]
            full_proj_transform = self.full_proj_transform_list[i]
            camera_center = self.camera_center_list[i]
            render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, world_view_transform, full_proj_transform,
                                   camera_center, self.gaussian, self.pipe, self.background, 1.0)

            img, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]


            Ll1 = l1_loss(img, img_gt)
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(img, img_gt))
            print(f"{self.iteration}th iter, {i}th kf, loss: {loss}")
            if(loss < self.loss_threshold):
                flag = False
                print(f"loss upgrade: {self.loss}")
                continue

            self.iteration+=1
            loss.backward()

            self.gaussian.max_radii2D[visibility_filter] = torch.max(self.gaussian.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])
            self.gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if self.iteration%80 == 0:
                self.gaussian.densify_and_prune(self.densify_grad_threshold, 0.005, self.cameras_extent, self.size_threshold)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)

        if flag:
            self.loss_threshold *=0.1

    def Visualize(self):
        if len(self.SP_pose_list) > 0:
            for i in range(len(self.viz_world_view_transform_list)):
                viz_world_view_transform = self.viz_world_view_transform_list[i]
                viz_full_proj_transform = self.viz_full_proj_transform_list[i]
                viz_camera_center = self.viz_camera_center_list[i]
                render_pkg = mg_render(self.FoVx, self.FoVy, 480, 640, viz_world_view_transform, viz_full_proj_transform,
                                       viz_camera_center, self.gaussian, self.pipe, self.background, 1.0)
                img = render_pkg["render"]
                np_render = img.permute(1, 2, 0).detach().to("cpu").numpy()
                cv2.imshow(f"start_gs{i}", np_render)
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
            self.CreateInitialKeyframe(rgb, gray, SP_xyz, pose)
            self.getNerfppNorm(pose)

        else:
            ref_3d_list= mapping_result_instance[6]
            ref_color_list= mapping_result_instance[7]
            self.CreateKeyframe(rgb, gray, SP_xyz, pose, ref_3d_list, ref_color_list)
            self.getNerfppNorm(pose)

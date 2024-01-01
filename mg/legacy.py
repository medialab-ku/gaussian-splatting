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
    print(f'Pose: {len(self.KF_pose_list)} 3D: {len(self.KF_ref_3d_list)} Color: {len(self.KF_ref_color_list)}')
    pointcloud_xyz = np.empty((0, 3))
    pointcloud_rgb = np.empty((0, 3))
    for i in range(len(self.KF_ref_3d_list)):
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
import math
import numpy as np
import collections
# import torch
# from plyfile import PlyData, PlyElement

def Rot2Quat(rot):
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


def QuaternionInfo(quaternion):
    axis = np.array((quaternion[0][0], quaternion[0][1], quaternion[0][2]), dtype = np.float32)
    axis_norm = np.linalg.norm(axis)
    if axis_norm == 0:
        return axis, axis_norm
    axis = axis / axis_norm
    angle = math.acos(quaternion[0][3]) * 2.0
    return axis, angle

def InfoBTWQuats(query, ref):
    # query x ref-1
    quaternion = np.zeros((1, 4), dtype=np.float32)  # xyz|w

    query_w = query[0][3]
    ref_w = ref[0][3]

    inv_ref = -ref[0]
    inv_ref[3] = ref_w

    inv_ref_vec = inv_ref[:3]
    query_vec = query[0][:3]

    cross = np.cross(query_vec, inv_ref_vec)

    vec = cross + ref_w * query_vec + query_w * inv_ref_vec
    dotpro = np.dot(query[0], inv_ref)

    quaternion[0][3] = ref_w * query_w - dotpro
    quaternion[0][:3] = vec

    quaternion[0] = quaternion[0] / np.linalg.norm(quaternion[0])
    return QuaternionInfo(quaternion)



BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]], dtype=np.float32)

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

# def TMPReadOneline(self):
#     path = "C:/lab/research/dataset/rgbd_dataset_freiburg3_long_office_household/gaussian_set/sp_50_100/sparse/0/images.txt"
#     images = {}
#     with open(path, "r") as fid:
#         while True:
#             line = fid.readline()
#             if not line:
#                 break
#             line = line.strip()
#             if len(line) > 0 and line[0] != "#":
#                 elems = line.split()
#                 image_id = int(elems[0])
#                 qvec = np.array(tuple(map(float, elems[1:5])))
#                 tvec = np.array(tuple(map(float, elems[5:8])))
#                 camera_id = int(elems[8])
#                 image_name = elems[9]
#                 elems = fid.readline().split()
#                 xys = np.column_stack([tuple(map(float, elems[0::3])),
#                                        tuple(map(float, elems[1::3]))])
#                 point3D_ids = np.array(tuple(map(int, elems[2::3])))
#
#                 print(f'qvec: {qvec.shape}')
#                 print(f'qvec: {qvec}')
#                 print(f'tvec: {tvec.shape}')
#                 print(f'tvec: {tvec}')
#
#
#                 print(f'xys: {xys.shape}')
#                 print(f'xys: {xys}')
#                 print(f'point3D_ids: {point3D_ids.shape}')
#                 print(f'point3D_ids: {point3D_ids}')
#
#                 images[image_id] = Image(
#                     id=image_id, qvec=qvec, tvec=tvec,
#                     camera_id=camera_id, name=image_name,
#                     xys=xys, point3D_ids=point3D_ids)
# def SavePLY(xyz_tensor, color_tensor, ply_path):
#     dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
#              ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
#              ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
#     xyz_array = torch.t(xyz_tensor).cpu().numpy()
#     color_array = torch.t(color_tensor).cpu().numpy()
#     normals = np.zeros_like(xyz_array)
#
#     elements = np.empty(xyz_array.shape[0], dtype=dtype)
#     attributes = np.concatenate((xyz_array, normals, color_array), axis=1)
#     elements[:] = list(map(tuple, attributes))
#     #
#     vertex_element = PlyElement.describe(elements, 'vertex')
#
#     ply_data = PlyData([vertex_element])
#     ply_data.write(ply_path)

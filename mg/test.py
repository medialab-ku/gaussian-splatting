from fast_slic import Slic
from superpixel import SuperPixelManager
import numpy as np
import cv2
import math


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
    axis = axis / np.linalg.norm(axis)
    angle = math.acos(quaternion[0][3]) * 2.0
    return axis, angle
def InfoBTWQuats(query, ref):
    # query x ref-1
    quaternion = np.empty((1, 4), dtype=np.float32)  # xyz|w

    ref_w = ref[0][3]
    query_w = query[0][3]

    inv_ref = -ref[0]
    inv_ref[3] = ref_w

    inv_ref_vec = inv_ref[:3]
    query_vec = query[0][:3]

    cross = np.cross(query_vec, inv_ref_vec)

    vec = cross + ref_w * query_vec + query_w * inv_ref_vec
    dotpro = np.dot(inv_ref, query[0])


    quaternion[0][3] = ref_w * query_w - dotpro
    quaternion[0][:3] = vec
    return QuaternionInfo(quaternion)



    # query x inv_ref
    # quaternion[0][0] = inv_ref[0][0] * query[0][3]  + query[0][0] * inv_ref[0][3] +

    # ref_w = ref[0][3]
    # inv_ref = -ref
    # inv_ref[0][3] = ref_w
    # print(inv_ref)
    #
    # new_quat = np.matmul(query, inv_ref)
    # print(new_quat)
    #
    # axis, angle = QuaternionInfo(new_quat)
    # print(angle)

A = np.zeros((1, 4), dtype=np.float32)  # xyz|w
B = np.zeros((1, 4), dtype=np.float32)  # xyz|w

B[0][0] = -3.0
B[0][1] = 2.0
B[0][2] = 5.0
B[0][3] = -1.0
A[0][0] = 1.0
A[0][1] = -3.0
A[0][2] = 4.0
A[0][3] = 1.0

InfoBTWQuats(A, B)

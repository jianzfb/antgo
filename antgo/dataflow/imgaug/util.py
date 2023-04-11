import cv2
import numpy as np
import torch
from pytorch3d.transforms.rotation_conversions import matrix_to_axis_angle


def get_rot6s(pose):
    num_pose = pose.shape[0]
    rot6s = list()
    for i in range(num_pose):
        rot6 = cv2.Rodrigues(pose[i])[0][:2]
        rot6s.append(rot6.reshape(-1))
    rot6s = np.array(rot6s)
    return rot6s


def matrix2dof(matrixs):
    # batch_size = matrixs.shape[0]
    # dofs = list()
    # for i in range(batch_size):
    #     dof = cv2.Rodrigues(matrixs[i].numpy())
    #     dofs.append(dof[0])
    # dofs = np.array(dofs)
    dofs = matrix_to_axis_angle(matrixs).reshape(-1, 3, 1)
    return dofs


def rot6tomatrix(rot6s):
    poses = rot6s.reshape(-1, 6)
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO

    # torch.det too slow, skip
    # assert (torch.stack([torch.det(mat) for mat in matrix]) < 0).sum() == 0

    return matrix


def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v / v_mag
    return v


def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)

    return out


def rot6todof(rot6s):
    # rot6s = torch.from_numpy(rot6s)
    matrix = rot6tomatrix(rot6s)
    output_dofs = matrix2dof(matrix)
    output_dofs = output_dofs.reshape(-1, 3)

    return output_dofs * -1


def eular_to_mat(roll, pitch, yaw):
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ]
    )
    ry = np.asarray(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ]
    )
    rz = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rx @ ry @ rz

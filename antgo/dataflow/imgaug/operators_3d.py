# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import cv2
import numpy as np
import random
import math

def load_img(path, order='RGB'):
    
    # load
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order=='RGB':
        img = img[:,:,::-1].copy()
    
    img = img.astype(np.float32)
    return img

def load_skeleton(path, joint_num):

    # load joint info (name, parent_id)
    skeleton = [{} for _ in range(joint_num)]
    with open(path) as fp:
        for line in fp:
            if line[0] == '#': continue
            splitted = line.split(' ')
            joint_name, joint_id, joint_parent_id = splitted
            joint_id, joint_parent_id = int(joint_id), int(joint_parent_id)
            skeleton[joint_id]['name'] = joint_name
            skeleton[joint_id]['parent_id'] = joint_parent_id
    # save child_id
    for i in range(len(skeleton)):
        joint_child_id = []
        for j in range(len(skeleton)):
            if skeleton[j]['parent_id'] == i:
                joint_child_id.append(j)
        skeleton[i]['child_id'] = joint_child_id
    
    return skeleton

def get_aug_config():
    trans_factor = 0.08
    scale_factor = 0.15
    rot_factor = 60
    color_factor = 0.1
    
    trans = [
        np.random.uniform(-trans_factor, trans_factor), 
        np.random.uniform(-trans_factor, trans_factor)
    ]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    # rot = np.clip(np.random.randn(), -2.0,
    #               2.0) * rot_factor if random.random() <= 0.6 else 0
    rot = (np.random.random() * 2 - 1) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    # 仅考虑灰度图情况
    color_scale = random.uniform(c_low, c_up)
    color_scale = np.array([color_scale, color_scale, color_scale])

    do_noise = np.random.random() <= 0.5
    # do_noise = False
    return trans, scale, rot, do_flip, color_scale, do_noise

def augmentation(
    img, bbox, joint_coord, joint_valid, hand_type, 
    mode, joint_type, input_img_shape, allow_geometric_aug=True, allow_flip=True, img_process_callback=None):
    img = img.copy(); 
    joint_coord = joint_coord.copy()
    hand_type = hand_type.copy()

    original_img_shape = img.shape
    joint_num = len(joint_coord)
    
    if mode == 'train':
        trans, scale, rot, do_flip, color_scale, do_noise = get_aug_config()
    else:
        trans, scale, rot, do_flip, color_scale, do_noise = [0,0], 1.0, 0.0, False, np.array([1,1,1]), False
    
    # TODO，考虑非刚性变形，以帮助不同手型的扩展

    if not allow_geometric_aug:
        trans, scale, rot, do_flip = [0,0], 1.0, 0.0, False
    if not allow_flip:
        do_flip = False

    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    # 生成局部patch
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, input_img_shape)

    if img_process_callback is not None:
        # 对局部图像进行定制化变换
        img = img_process_callback(img)

    # 颜色尺度变换
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    
    if do_noise:
        # 加噪声
        img = img.astype(np.float)
        noise_delta = (np.random.random(img.shape) * 2 - 1) * 10
        img = img + noise_delta
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
    
    if do_flip:
        # 加翻转
        joint_coord[:,0] = original_img_shape[1] - joint_coord[:,0] - 1
        # 左右手坐标切换
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), joint_coord[joint_type['right']].copy()
        # 左右手可见性切换
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), joint_valid[joint_type['right']].copy()
        # 左右手标签切换
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()

    # 需要重新设置关键点的可见性
    for i in range(joint_num):
        joint_coord[i,:2] = trans_point2d(joint_coord[i,:2], trans)
        joint_valid[i] = joint_valid[i] * (joint_coord[i,0] >= 0) * (joint_coord[i,0] < input_img_shape[1]) * (joint_coord[i,1] >= 0) * (joint_coord[i,1] < input_img_shape[0])

    return img, joint_coord, joint_valid, hand_type, inv_trans

def transform_input_to_output_space(joint_coord, joint_valid, rel_root_depth, root_valid, root_joint_idx, joint_type, input_img_shape, output_hm_shape, bbox_3d_size, output_root_hm_shape, bbox_3d_size_root):
    # transform to output heatmap space
    joint_coord = joint_coord.copy(); joint_valid = joint_valid.copy()
    
    joint_coord[:,0] = joint_coord[:,0] / input_img_shape[1] * output_hm_shape[2]
    joint_coord[:,1] = joint_coord[:,1] / input_img_shape[0] * output_hm_shape[1]
    joint_coord[joint_type['right'],2] = joint_coord[joint_type['right'],2] - joint_coord[root_joint_idx['right'],2]
    joint_coord[joint_type['left'],2] = joint_coord[joint_type['left'],2] - joint_coord[root_joint_idx['left'],2]
  
    joint_coord[:,2] = (joint_coord[:,2] / (bbox_3d_size/2) + 1)/2. * output_hm_shape[0]
    joint_valid = joint_valid * ((joint_coord[:,2] >= 0) * (joint_coord[:,2] < output_hm_shape[0])).astype(np.float32)
    rel_root_depth = (rel_root_depth / (bbox_3d_size_root/2) + 1)/2. * output_root_hm_shape
    root_valid = root_valid * ((rel_root_depth >= 0) * (rel_root_depth < output_root_hm_shape)).astype(np.float32)
    
    return joint_coord, joint_valid, rel_root_depth, root_valid

def get_bbox(joint_img, joint_valid):
    x_img = joint_img[:,0][joint_valid==1]; y_img = joint_img[:,1][joint_valid==1];
    xmin = min(x_img); ymin = min(y_img); xmax = max(x_img); ymax = max(y_img);

    x_center = (xmin+xmax)/2.; width = xmax-xmin;
    xmin = x_center - 0.5*width*1.2
    xmax = x_center + 0.5*width*1.2
    
    y_center = (ymin+ymax)/2.; height = ymax-ymin;
    ymin = y_center - 0.5*height*1.2
    ymax = y_center + 0.5*height*1.2

    bbox = np.array([xmin, ymin, xmax-xmin, ymax-ymin]).astype(np.float32)
    return bbox

def process_bbox(bbox, original_img_shape, input_img_shape):

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w/2.
    c_y = bbox[1] + h/2.
    aspect_ratio = input_img_shape[1]/input_img_shape[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w*1.25
    bbox[3] = h*1.25
    bbox[0] = c_x - bbox[2]/2.
    bbox[1] = c_y - bbox[3]/2.

    return bbox

def generate_patch_image(cvimg, bbox, do_flip, scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5*bbox[2])
    bb_c_y = float(bbox[1] + 0.5*bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    if do_flip:
        img = img[:, ::-1, :]
        bb_c_x = img_width - bb_c_x - 1
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def cam2pixel_omnidir(p, cam0d_i, cam0d_mu, cam0d_d):
    point = p.copy()
    point = point.reshape(p.shape[0],1,3)
    uv,_ = cv2.omnidir.projectPoints(point, np.zeros(3), np.zeros(3), cam0d_i, cam0d_mu, cam0d_d)

    return uv.squeeze(1)

def pixel2cam_omnidir(uv,z,K,D,xi):
    '''
    2.5d到3d投影
    :param uv:  图片 uv坐标
    :param z: 深度
    :param K: 相机内参
    :param D: 相机畸变
    :param xi: omni相机 xi参数
    :return: 3d坐标
    '''
    u = (uv[:, 0] - K[0, 2]) / K[0, 0]
    v = (uv[:, 1] - K[1, 2]) / K[1, 1]
    _u = u.copy()
    _v = v.copy()

    for i in range(20):
        r2 = u ** 2 + v ** 2
        r4 = r2 * r2
        u = (_u - 2 * D[2] * u * v - D[3] * (r2 + 2 * u * u)) / (1 + D[0] * r2 + D[1] * r4)
        v = (_v - 2 * D[3] * u * v - D[2] * (r2 + 2 * v * v)) / (1 + D[0] * r2 + D[1] * r4)

    r2 = u ** 2 + v ** 2
    a = r2 + 1
    b = 2 * xi * r2
    cc = r2 * xi * xi - 1
    Zs = (-b + np.sqrt(b * b - 4 * a * cc)) / (2 * a)
    xw = np.vstack([u * (Zs + xi), v * (Zs + xi), Zs])
    xyz = xw*(z/xw[2])
    xyz = np.transpose(xyz)
    return xyz

def pixel2cam_invK(pixel_coord, inv_K):
    z = pixel_coord[:,2:]
    uvones = pixel_coord.copy()    
    uvones[:,2] = 1.0
    test_tt = np.matmul(inv_K, np.transpose(uvones))
    test_tt = np.transpose(test_tt)
    test_tt[:,2:] = z
    return test_tt

def pixel2fisheyecam(pixel_coord, cam_param):
    pass

def fisheyecam2pixel(cam_coord, cam_param):
    pass

def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord

def cam2world(cam_coord, inv_R, T):
    world_coord = np.dot(inv_R, cam_coord) + T
    return world_coord
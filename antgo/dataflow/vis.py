# -*- coding: UTF-8 -*-
# @Time    : 2022/4/27 20:38
# @File    : vis.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import os.path as osp
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from PIL import Image, ImageDraw


def get_keypoint_rgb(skeleton):
    rgb_dict= {}
    for joint_id in range(len(skeleton)):
        joint_name = skeleton[joint_id]['name']

        if joint_name.endswith('thumb_null'):
            rgb_dict[joint_name] = (255, 0, 0)
        elif joint_name.endswith('thumb3'):
            rgb_dict[joint_name] = (255, 51, 51)
        elif joint_name.endswith('thumb2'):
            rgb_dict[joint_name] = (255, 102, 102)
        elif joint_name.endswith('thumb1'):
            rgb_dict[joint_name] = (255, 153, 153)
        elif joint_name.endswith('thumb0'):
            rgb_dict[joint_name] = (255, 204, 204)
        elif joint_name.endswith('index_null'):
            rgb_dict[joint_name] = (0, 255, 0)
        elif joint_name.endswith('index3'):
            rgb_dict[joint_name] = (51, 255, 51)
        elif joint_name.endswith('index2'):
            rgb_dict[joint_name] = (102, 255, 102)
        elif joint_name.endswith('index1'):
            rgb_dict[joint_name] = (153, 255, 153)
        elif joint_name.endswith('middle_null'):
            rgb_dict[joint_name] = (255, 128, 0)
        elif joint_name.endswith('middle3'):
            rgb_dict[joint_name] = (255, 153, 51)
        elif joint_name.endswith('middle2'):
            rgb_dict[joint_name] = (255, 178, 102)
        elif joint_name.endswith('middle1'):
            rgb_dict[joint_name] = (255, 204, 153)
        elif joint_name.endswith('ring_null'):
            rgb_dict[joint_name] = (0, 128, 255)
        elif joint_name.endswith('ring3'):
            rgb_dict[joint_name] = (51, 153, 255)
        elif joint_name.endswith('ring2'):
            rgb_dict[joint_name] = (102, 178, 255)
        elif joint_name.endswith('ring1'):
            rgb_dict[joint_name] = (153, 204, 255)
        elif joint_name.endswith('pinky_null'):
            rgb_dict[joint_name] = (255, 0, 255)
        elif joint_name.endswith('pinky3'):
            rgb_dict[joint_name] = (255, 51, 255)
        elif joint_name.endswith('pinky2'):
            rgb_dict[joint_name] = (255, 102, 255)
        elif joint_name.endswith('pinky1'):
            rgb_dict[joint_name] = (255, 153, 255)
        else:
            rgb_dict[joint_name] = (230, 230, 0)
        
    return rgb_dict

def vis_keypoints(img, kps, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad = 3, save_path='./'):
    rgb_dict = get_keypoint_rgb(skeleton)
    _img = Image.fromarray(img.transpose(1,2,0).astype('uint8')) 
    draw = ImageDraw.Draw(_img)
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']
        
        kps_i = (kps[i][0].astype(np.int32), kps[i][1].astype(np.int32))
        kps_pid = (kps[pid][0].astype(np.int32), kps[pid][1].astype(np.int32))

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            draw.line([(kps[i][0], kps[i][1]), (kps[pid][0], kps[pid][1])], fill=rgb_dict[parent_joint_name], width=line_width)
        if score[i] > score_thr:
            draw.ellipse((kps[i][0]-circle_rad, kps[i][1]-circle_rad, kps[i][0]+circle_rad, kps[i][1]+circle_rad), fill=rgb_dict[joint_name])
        if score[pid] > score_thr and pid != -1:
            draw.ellipse((kps[pid][0]-circle_rad, kps[pid][1]-circle_rad, kps[pid][0]+circle_rad, kps[pid][1]+circle_rad), fill=rgb_dict[parent_joint_name])
    
    _img.save(osp.join(save_path, filename))


def __render_2d_xyz(kps_3d,score, skeleton, ax, rgb_dict, score_thr=0.4, axis='x'):
    kps_3d = kps_3d.copy()

    kps_2d = np.zeros((kps_3d.shape[0], 2))
    a_name = 'x'
    b_name = 'y'
    if axis == 'x':
        # show y,z
        kps_2d[:,0] = kps_3d[:,2]
        kps_2d[:,1] = -kps_3d[:,1]
        a_name = 'z'
        b_name = 'y'
    elif axis == 'y':
        # show x,z
        kps_2d[:,0] = kps_3d[:,0]
        kps_2d[:,1] = kps_3d[:,2]
        a_name = 'x'
        b_name = 'z'
    else:
        # show x,y
        kps_2d[:,0] = kps_3d[:,0]
        kps_2d[:,1] = -kps_3d[:,1]
        a_name = 'x'
        b_name = 'y'
    ax.set_title(f'show {a_name}-{b_name} plane')
    ax.set_xlabel(a_name)
    ax.set_ylabel(b_name)    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        a = np.array([kps_2d[i,0], kps_2d[pid,0]])
        b = np.array([kps_2d[i,1], kps_2d[pid,1]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(a,b, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = 3)

        if score[i] > score_thr:
            ax.scatter(kps_2d[i,0], kps_2d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_2d[pid,0], kps_2d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')


def vis_3d_keypoints(kps_3d, score, skeleton, filename, score_thr=0.4, line_width=3, circle_rad=3, save_path='./'):
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.3, hspace=0.4)
    ax = fig.add_subplot(221, projection='3d')
    rgb_dict = get_keypoint_rgb(skeleton)
    
    for i in range(len(skeleton)):
        joint_name = skeleton[i]['name']
        pid = skeleton[i]['parent_id']
        parent_joint_name = skeleton[pid]['name']

        x = np.array([kps_3d[i,0], kps_3d[pid,0]])
        y = np.array([kps_3d[i,1], kps_3d[pid,1]])
        z = np.array([kps_3d[i,2], kps_3d[pid,2]])

        if score[i] > score_thr and score[pid] > score_thr and pid != -1:
            ax.plot(x, z, -y, c = np.array(rgb_dict[parent_joint_name])/255., linewidth = line_width)
        if score[i] > score_thr:
            ax.scatter(kps_3d[i,0], kps_3d[i,2], -kps_3d[i,1], c = np.array(rgb_dict[joint_name]).reshape(1,3)/255., marker='o')
        if score[pid] > score_thr and pid != -1:
            ax.scatter(kps_3d[pid,0], kps_3d[pid,2], -kps_3d[pid,1], c = np.array(rgb_dict[parent_joint_name]).reshape(1,3)/255., marker='o')

    # x 方向
    ax = fig.add_subplot(222)
    __render_2d_xyz(kps_3d,score, skeleton, ax, rgb_dict,score_thr, axis='x')

    # y
    ax = fig.add_subplot(223)
    __render_2d_xyz(kps_3d,score, skeleton, ax, rgb_dict,score_thr, axis='y')

    # z
    ax = fig.add_subplot(224)
    __render_2d_xyz(kps_3d,score, skeleton, ax, rgb_dict,score_thr, axis='z')
    
    fig.savefig(osp.join(save_path, filename), dpi=fig.dpi)
    plt.close()


def vis_2d_boxes_in_image(image, boxes, labels, save_path='./'):
    color_map = [(255,0,0),(0,0,255),(0,255,0),(255,255,0),(0,255,255),(255,0,255),(0,0,0),(255,255,255)]
    for box, label in zip(boxes, labels):
        x1,y1,x2,y2 = box
        
        color_index = int(label) % len(color_map)
        cv2.rectangle(image, (int(x1),int(y1)), (int(x2), int(y2)), color_map[color_index], 2)
    cv2.imwrite(save_path, image)


def vis_3d_boxes_in_image(image, boxes, camera_param, camera_model):
    pass

def vis_3d_points_in_image(image, points, skeleton, camera_param, camera_model='fisheye'):
    
    pass

def vis_2d_points_in_image(image, points, skeleton):
    pass


def vis_mano_in_image(image, pose, shape, trans, camera_param, camera_model='fisheye'):
    pass



def imshow(image, bboxes=None, keypoints=None):
    # Create figure and axes
    fig, ax = plt.subplots(1)

    # Display the image
    image = image.copy()
    if keypoints is not None:
        for k in range(keypoints.shape[0]):
            x, y = keypoints[k]
            image[int(y), int(x), :] = 255

    ax.imshow(image)

    if bboxes is not None:
        for b in range(bboxes.shape[0]):
            x0, y0, x1, y1 = bboxes[b]
            width = x1 - x0
            height = y1 - y0
            # Create a Rectangle patch
            rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)

    plt.show()
    plt.waitforbuttonpress()
# -*- coding: UTF-8 -*-
# @Time    : 2019-06-08 15:51
# @File    : lsp.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
from antgo.dataflow.dataset import *
import scipy.io as io
import os
import numpy as np
import cv2
import time
from antgo.framework.helper.fileio.file_client import *
from filelock import FileLock


__all__ = ['MPII']

class MPII(Dataset):
    def __init__(self, train_or_test, dir=None, ext_params=None):
        super(MPII, self).__init__(train_or_test, dir, ext_params)
        self.class_name = [
            'rankl',
            'rknee',
            'rhip',
            'lhip',
            'lknee',
            'lankl',
            'pelvis',
            'thorax',
            'upper_neck',
            'head',
            'rwri',
            'relb',
            'rsho',
            'lsho',
            'lelb',
            'lwri'
        ]    
        lock = FileLock('DATASET.lock')
        with lock:
            if not os.path.exists(os.path.join(self.dir, 'images')):
                # 数据集不存在，需要重新下载，并创建标记
                if not os.path.exists(self.dir):
                    os.makedirs(self.dir)

                ali = AliBackend()
                ali.download('ali:///dataset/mpii/mpii_human_pose_v1.tar.gz', self.dir)
                ali.download('ali:///dataset/mpii/mpii_human_pose_v1_u12_1.tar.gz', self.dir)   # 这是lspet的高精集合

                os.system(f'cd {self.dir} && unzip mpii_human_pose_v1.tar.gz')
                os.system(f'cd {self.dir} && unzip mpii_human_pose_v1_u12_1.tar.gz')

        matlab_mpii = io.loadmat(os.path.join(self.dir, 'mpii_human_pose_v1_u12_1' ,'mpii_human_pose_v1_u12_1.mat'), struct_as_record=False)['RELEASE'][0, 0]
        num_images = matlab_mpii.__dict__['annolist'][0].shape[0]

        self.info = []
        for img_idx in range(num_images):
            # Initialize empty placeholder        
            annotation_mpii = matlab_mpii.__dict__['annolist'][0, img_idx]
            train_test_mpii = matlab_mpii.__dict__['img_train'][0, img_idx].flatten()[0]
            person_id = matlab_mpii.__dict__['single_person'][img_idx][0].flatten()

            # Load the individual image. Throw an exception if image corresponding to filename not available.
            img_name = annotation_mpii.__dict__['image'][0, 0].__dict__['name'][0]
            img_path = os.path.join(self.dir, 'images', img_name)

            if self.train_or_test == 'train':
                if train_test_mpii != 1:
                    continue

            if self.train_or_test == 'test':
                if train_test_mpii == 0:
                    self.info.append({
                        'image_name': img_path,
                        'anno':{}
                    })
                else:
                    continue   

            # Iterate over persons
            joints2d = []
            joints_vis = []
            bboxes = []
            for person in (person_id - 1):
                try:
                    annopoints_img_mpii = annotation_mpii.__dict__['annorect'][0, person].__dict__['annopoints'][0, 0]
                    num_joints = annopoints_img_mpii.__dict__['point'][0].shape[0]
                    # Iterate over present joints
                    person_joints_2d = np.zeros((16, 2), dtype=np.float32)
                    person_joints_vis = np.zeros((16), dtype=np.int32)
                    person_box = np.zeros((4), dtype=np.float32)
                    for i in range(num_joints):
                        x = annopoints_img_mpii.__dict__['point'][0, i].__dict__['x'].flatten()[0]
                        y = annopoints_img_mpii.__dict__['point'][0, i].__dict__['y'].flatten()[0]
                        id_ = annopoints_img_mpii.__dict__['point'][0, i].__dict__['id'][0][0]
                        vis = annopoints_img_mpii.__dict__['point'][0, i].__dict__['is_visible'].flatten()

                        # No entry corresponding to visible
                        if vis.size == 0:
                            vis = 1
                        else:
                            vis = vis.item()

                        person_joints_2d[id_, 0] = x
                        person_joints_2d[id_, 1] = y
                        person_joints_vis[id_] = vis

                    person_box = np.array([
                        np.min(person_joints_2d[person_joints_vis, 0]), 
                        np.min(person_joints_2d[person_joints_vis, 1]), 
                        np.max(person_joints_2d[person_joints_vis, 0]), 
                        np.max(person_joints_2d[person_joints_vis, 1])
                        ])
                    bboxes.append(person_box)
                    joints2d.append(person_joints_2d)
                    joints_vis.append(person_joints_vis)
                    
                except KeyError:
                    # Person 'x' could not have annotated joints, hence move to person 'y'
                    continue

            if len(joints2d) == 0:
                continue

            anno = {
                'bboxes': np.stack(bboxes, 0),
                'labels': np.zeros((len(bboxes)), dtype=np.int32),
                'joints2d': np.stack(joints2d, 0),
                'joints_vis': np.stack(joints_vis, 0)
            }
            self.info.append({
                'image_name': img_path,
                'anno': anno
            })

    @property
    def size(self):
        return len(self.info)

    def at(self, img_idx):
        image = cv2.imread(self.info[img_idx]['image_name'])
        return (image, self.info[img_idx]['anno'])

    def split(self, split_params={}, split_method=''):
        raise NotImplementedError

# mpii = MPII('train', '/root/workspace/dataset/mpii')
# for i in range(1, mpii.size):
#     data = mpii.sample(i)
#     image = data['image']
#     joints2d = data['joints2d']
#     joints_vis = data['joints_vis']
    
#     for joint_i, (x,y) in enumerate(joints2d[0]):
#         x, y = int(x), int(y)
        
#         if joints_vis[0][joint_i]:
#             cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)

#     cv2.imwrite(f'./aabb_{i}.png', image)
#     print(i)
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


__all__ = ['LSP']

class LSP(Dataset):
    def __init__(self, train_or_test, dir=None, ext_params=None):
        super(LSP, self).__init__(train_or_test, dir, ext_params)
        # 不区分训练集和测试集，仅用来进行模型正确性测试
        url_address = 'http://image.mltalker.com/lsp_dataset.zip'
        self.class_num = [
            'Right ankle',
            'Right knee',
            'Right hip',
            'Left hip',
            'Left knee',
            'Left ankle',
            'Right wrist',
            'Right elbow',
            'Right shoulder',
            'Left shoulder',
            'Left elbow',
            'Left wrist',
            'Neck',
            'Head top'
        ]

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        if not os.path.exists(os.path.join(self.dir, 'joints.mat')):
            os.system(f'cd {self.dir} && wget {url_address} && unzip lsp_dataset.zip')

        matr = io.loadmat(os.path.join(self.dir, 'joints.mat'))
        self.dataset = matr['joints']   # 3x14x2000

    @property
    def size(self):
        return 2000

    def at(self, id):
        image_file = os.path.join(self.dir, 'images', 'im%04d.jpg'%(id+1))
        image = cv2.imread(image_file)

        anno = {
            'joints2d': np.transpose(self.dataset[:2,:,id], [1,0]),
            'joints_vis': 1-self.dataset[2,:,id]
        }
        return (image, anno)

    def split(self, split_params={}, split_method=''):
        raise NotImplementedError

# lsp = LSP('train', '/root/workspace/handtracking/lsp_dataset')
# for i in range(10):
#     data = lsp.sample(i)
#     image = data['image']
#     joints2d = data['joints2d']
#     joints_vis = data['joints_vis']
    
#     for joint_i, (x,y) in enumerate(joints2d):
#         x, y = int(x), int(y)
        
#         if joints_vis[joint_i]:
#             cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)

#     cv2.imwrite(f'./aabb/aabb_{i}.png', image)
#     print(i)
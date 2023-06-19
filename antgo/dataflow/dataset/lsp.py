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

__all__ = ['LSP']

class LSP(Dataset):
    def __init__(self, train_or_test, dir=None, ext_params=None):
        super(LSP, self).__init__(train_or_test, dir, ext_params)
        self.class_name = [
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
        lock = FileLock('DATASET.lock')
        with lock:
            if not os.path.exists(os.path.join(self.dir, 'lsp','joints.mat')):
                # 数据集不存在，需要重新下载，并创建标记
                if not os.path.exists(self.dir):
                    os.makedirs(self.dir)

                os.makedirs(os.path.join(self.dir, 'lsp'), exist_ok=True)
                os.makedirs(os.path.join(self.dir, 'lspet', 'images'), exist_ok=True)

                ali = AliBackend()
                ali.download('ali:///dataset/lsp/lsp_dataset.zip', os.path.join(self.dir, 'lsp'))
                # ali.download('ali:///dataset/lsp/lspet_dataset.zip', os.path.join(self.dir, 'lspet'))
                ali.download('ali:///dataset/lsp/hr-lspet.zip', self.dir)   # 这是lspet的高精集合
                
                os.system(f'cd {os.path.join(self.dir, "lsp")} && unzip lsp_dataset.zip')
                # os.system(f'cd {os.path.join(self.dir, "lspet")} && unzip lspet_dataset.zip')            
                os.system(f'cd {self.dir} && unzip hr-lspet.zip && mv hr-lspet/*.png lspet/images && mv hr-lspet/* lspet/')

        self.dataset = []
        # lsp (1000 train + 1000 test)
        lsp_matr = io.loadmat(os.path.join(self.dir, 'lsp','joints.mat'))['joints']
        train_idx = list(range(0,2000,2))
        test_idx = list(range(1,2000,2))
        if self.train_or_test == 'train':
            # path, 14x2, visible
            self.dataset = [(os.path.join(self.dir, 'lsp', 'images', 'im%04d.jpg'%(idx+1)), np.transpose(lsp_matr[:2,:,idx], [1,0]), 1-lsp_matr[2,:,idx]) for idx in train_idx]
        else:
            self.dataset = [(os.path.join(self.dir, 'lsp', 'images', 'im%04d.jpg'%(idx+1)), np.transpose(lsp_matr[:2,:,idx], [1,0]), 1-lsp_matr[2,:,idx]) for idx in test_idx]

        # lspet
        lspet_matr = io.loadmat(os.path.join(self.dir, 'lspet', 'joints.mat'))['joints']
        if self.train_or_test == 'train':
            lspet_dataset = []
            count = 0
            for idx in range(10000):
                if not os.path.exists(os.path.join(self.dir, 'lspet', 'images', 'im%05d.png'%(idx+1))):
                    continue

                lspet_dataset.append((
                    os.path.join(self.dir, 'lspet', 'images', 'im%05d.png'%(idx+1)),
                    lspet_matr[:,:2,count],
                    lspet_matr[:,2,count],
                ))
                count += 1
            
            self.dataset.extend(lspet_dataset)

    @property
    def size(self):
        return len(self.dataset)

    def at(self, id):
        
        image_file, joints2d, visible = self.dataset[id]
        image = cv2.imread(image_file)
        visible_pos = np.where(visible == 1)
        bboxes = np.array([[
          np.min(joints2d[visible_pos, 0]), 
          np.min(joints2d[visible_pos, 1]), 
          np.max(joints2d[visible_pos, 0]), 
          np.max(joints2d[visible_pos, 1])
        ]])
        anno = {
            'bboxes': bboxes.astype(np.float32),
            'labels': np.zeros((1), dtype=np.int32),
            'joints2d': np.expand_dims(joints2d, 0).astype(np.float32),
            'joints_vis': np.expand_dims(visible, 0).astype(np.int32)
        }
        return (image, anno)

    def split(self, split_params={}, split_method=''):
        raise NotImplementedError

# lsp = LSP('test', '/root/workspace/dataset/lsp')
# crjo = ConvertRandomObjJointsAndOffset(input_size=(128,128), heatmap_size=(16,16), num_joints=14)

# for i in range(lsp.size):
#     data = lsp.sample(i)
#     data = crjo(data)
    
#     image = data['image']
#     joints2d = data['joints2d']
#     joints_vis = data['joints_vis']
    
#     # for joint_i, (x,y) in enumerate(joints2d[0]):
#     #     x, y = int(x), int(y)
        
#     #     if joints_vis[0][joint_i]:
#     #         cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)

#     # cv2.imwrite(f'./aabb_{i}.png', image)
#     print(i)
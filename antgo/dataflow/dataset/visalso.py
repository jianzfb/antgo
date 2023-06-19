# -*- coding: UTF-8 -*-
# @Time    : 2019-06-08 15:51
# @File    : visalso.py
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
from filelock import FileLock

__all__ = ['VisalSO']

class VisalSO(Dataset):
    def __init__(self, train_or_test="train", dir=None, ext_params=None):
        super().__init__(train_or_test, dir, ext_params)
        # 不区分训练集和测试集，仅用来进行模型正确性测试        
        url_address = 'http://visal.cs.cityu.edu.hk/static/downloads/SmallObjectDataset.zip'

        self.class_name = [
            'fish',
            'fly',
            'honeybee',
            'seagull'
        ]

        lock = FileLock('DATASET.lock')
        with lock:
            if not os.path.exists(os.path.join(self.dir, 'Small Object dataset')):
                # 数据集不存在，需要重新下载，并创建标记
                if not os.path.exists(self.dir):
                    os.makedirs(self.dir)
    
                os.system(f'cd {self.dir} && wget {url_address} && unzip SmallObjectDataset.zip')

        self.image_file_list = []
        self.bboxes_list = []
        for ci, cn in enumerate(self.class_name):
            subfolder = os.path.join(self.dir, 'Small Object dataset', cn)
            cn_gt_bbox_folder = os.path.join(subfolder, 'gt-bbox')
            cn_img_folder = os.path.join(subfolder, 'img')

            image_id_to_file_name_map = {}
            for image_file_name in os.listdir(cn_img_folder):
                image_id = image_file_name.split('.')[0][-3:]
                image_id_to_file_name_map[image_id] = image_file_name

            for mat_file_name in os.listdir(cn_gt_bbox_folder):
                matr = io.loadmat(os.path.join(cn_gt_bbox_folder, mat_file_name))
                bboxes = matr['bbox_all'].astype(np.int32)
                x0 = bboxes[:,0] - bboxes[:,2]/2
                y0 = bboxes[:,1] - bboxes[:,3]/2
                x1 = bboxes[:,0] + bboxes[:,2]/2
                y1 = bboxes[:,1] + bboxes[:,3]/2

                bboxes[:,0] = x0
                bboxes[:,1] = y0
                bboxes[:,2] = x1
                bboxes[:,3] = y1
                self.bboxes_list.append((bboxes, np.array([ci for _ in range(len(bboxes))])))

                image_id = mat_file_name.split('.')[0].split('_')[-1]
                image_file_name = image_id_to_file_name_map[image_id]
                self.image_file_list.append(os.path.join(cn_img_folder, image_file_name))

    @property
    def size(self):
        return len(self.image_file_list)
    
    def at(self, id):
        image = cv2.imread(self.image_file_list[id])
        bboxes, labels = self.bboxes_list[id]
        return (image, {'bboxes': bboxes, 'labels': labels, 'image_meta': {'image_shape': (image.shape[0], image.shape[1])}})
    
    def split(self, split_params={}, split_method=''):
        raise NotImplementedError

# vso = VisalSO('train', './visalso_dataset')
# for i in range(4):
#     info = vso.sample(i)
    
#     image = info['image']
#     gt_bbox = info['bboxes']
#     for x0,y0,x1,y1 in gt_bbox:
#         image = cv2.rectangle(image, (int(x0), int(y0)), (int(x1), int(y1)) ,(0,0,255), 1)
        
#     cv2.imwrite('/root/workspace/handtracking/visalso_dataset/abcd.png', image)    
#     print('d')
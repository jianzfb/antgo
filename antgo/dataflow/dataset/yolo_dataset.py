# -*- coding: UTF-8 -*-
# @Time : 29/03/2018
# @File : yolo_dataset.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
from antgo.dataflow.dataset import *
import os
import numpy as np
import cv2
import time
import yaml

__all__ = ['YoloDataset']
class YoloDataset(Dataset):
    def __init__(self, train_or_test, dir=None, **kwargs):
        anno_file = None
        if dir.endswith('.yaml'):
            anno_file = dir
            dir = os.path.dirname(dir)
        super(YoloDataset, self).__init__(train_or_test, dir)

        if anno_file is None:
            # 默认文件夹结构
            # dir
            #   labels
            #     train
            #     test
            #   images
            #     train
            #     test
            self.label_folder = os.path.join(self.dir, 'labels', train_or_test)
            self.image_folder = os.path.join(self.dir, 'images', train_or_test)
        else:
            with open(anno_file) as f:
                anno = yaml.load(f, Loader=yaml.SafeLoader)
                # anno[train_or_test] 图像文件夹
                self.image_folder = os.path.join(self.dir, anno[train_or_test])
                # anno[train_or_test].replace('images', 'labels') 标签文件夹
                self.label_folder = os.path.join(self.dir, anno[train_or_test].replace('images', 'labels'))

        self.file_list = [file_name for file_name in os.listdir(self.image_folder) if file_name[0] != '.']
        self.sample_num = len(self.file_list)

    @property
    def size(self):
        # 返回数据集大小
        return self.sample_num

    def sample(self, id):
        # 根据id，返回对应样本
        anno_info = {
            'bboxes': [],
            'labels': []
        }
        file_name = self.file_list[id]
        pos = file_name.rfind('.')
        pure_name = file_name[:pos]
        with open(os.path.join(self.label_folder, f'{pure_name}.txt'), 'r') as fp:
            line = fp.readline()
            line = line.strip()
            while line:
                label, x0,y0,x1,y1 = line.split()
                anno_info['bboxes'].append([float(x0), float(y0), float(x1), float(y1)])
                anno_info['labels'].append(int(label))

                line = fp.readline()
                line = line.strip()

        image_file = os.path.join(self.image_folder, file_name)
        image = cv2.imread(image_file)
        image_h, image_w = image.shape[:2]
        if len(anno_info['bboxes']) == 0:
            return {
                'image': image,
                'bboxes': np.empty((0, 4), dtype=np.float32),
                'labels': np.empty((0), dtype=np.float32)
            }

        bboxes = np.array(anno_info['bboxes'], dtype=np.float32)
        bboxes = bboxes * np.array([[image_w, image_h, image_w, image_h]], dtype=np.float32)
        bboxes[:,:2] = bboxes[:,:2] - bboxes[:,2:]/2
        bboxes[:,2:] = bboxes[:,:2] + bboxes[:,2:]

        labels = np.array(anno_info['labels'], dtype=np.int32)
        return {
            'image': image,
            'bboxes': bboxes,
            'labels': labels
        }

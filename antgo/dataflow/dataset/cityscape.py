# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : cityscape.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import numpy as np
from antgo.dataflow.dataset import *
from antgo.framework.helper.fileio.file_client import *
import cv2
import time
from filelock import FileLock


__all__ = ['Cityscape']
class Cityscape(Dataset):
    def __init__(self, train_or_test, dir=None, ext_params=None):
        super(Cityscape, self).__init__(train_or_test, dir, ext_params=ext_params)
        assert(train_or_test in ['train', 'val', 'test'])
        lock = FileLock('DATASET.lock')
        with lock:
            if not os.path.exists(os.path.join(self.dir, 'cityscapes')):
                # 数据集不存在，需要重新下载，并创建标记                
                ali = AliBackend()
                ali.download('ali:///dataset/cityscapes/cityscapes.tar', self.dir)
                os.system(f'cd {self.dir} && tar -xf cityscapes.tar')

        self._image_file_list = []
        self._annotation_file_list = []
        index_file = f'{self.train_or_test}.list'
        with open(os.path.join(self.dir, 'cityscapes', index_file), 'r') as fp:
            content = fp.readline()
            content = content.strip()
            
            while content:
                image_file, label_file = content.split(' ')
                label_file = label_file[:-24]+'gtFine_labelIds.png'
                self._image_file_list.append(os.path.join(self.dir, 'cityscapes', image_file))
                self._annotation_file_list.append(os.path.join(self.dir, 'cityscapes', label_file))
                content = fp.readline()
                content = content.strip()
                if content == '':
                    break

    @property
    def size(self):
        return len(self._image_file_list)
        
    def at(self, id):
        image = cv2.imread(self._image_file_list[id])
        segments = cv2.imread(self._annotation_file_list[id], cv2.IMREAD_GRAYSCALE)

        return (
            image, 
            {
                'segments':segments, 
                'image_meta': {
                    'image_shape': (image.shape[0], image.shape[1]),
                    'image_file': self._image_file_list[id]
                }
            }
        )

# cityscape = Cityscape('train', '/root/workspace/dataset/cityscapes')
# size = cityscape.size
# print(f'cityscape size {size}')
# label_num_map = {}
# for i in range(size):
#   data = cityscape.sample(i)
#   ll = set(data['segments'].flatten().tolist())
#   for l in ll:
#     if l not in label_num_map:
#       label_num_map[l] = 0
#     label_num_map[l] += 1

# print(label_num_map)
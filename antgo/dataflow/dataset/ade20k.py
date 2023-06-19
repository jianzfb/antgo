# -*- coding: UTF-8 -*-
# @Time : 2018/8/13
# @File : ade20k.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import os
import numpy as np
import cv2
import time
from antgo.dataflow.dataset import *
from antgo.framework.helper.fileio.file_client import *
from filelock import FileLock


__all__ = ['ADE20K']
class ADE20K(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    if train_or_test != 'train':
      train_or_test = 'val'

    super(ADE20K, self).__init__(train_or_test, dir,ext_params=ext_params)
    assert(train_or_test in ['train', 'val', 'test'])
    lock = FileLock('DATASET.lock')
    with lock:
      if not os.path.exists(os.path.join(self.dir, 'ADEChallengeData2016')):
        # 数据集不存在，需要重新下载，并创建标记
        os.makedirs(self, exist_ok=True)
        ali = AliBackend()
        ali.download('ali:///dataset/ade20k/ADEChallengeData2016.zip', self.dir)
        os.system(f'cd {self.dir} && unzip ADEChallengeData2016.zip')

    subfolder_name = 'training' if self.train_or_test == 'train' else 'validation'
    self._image_file_list = []
    self._annotation_file_list = []

    # 读取图片路径 和 分割GT
    for image_file_name in os.listdir(os.path.join(self.dir, 'ADEChallengeData2016','images', subfolder_name)):
      pure_name = image_file_name.split('.')[0]
      anno_file_name = f'{pure_name}.png'

      self._image_file_list.append(os.path.join(self.dir, 'ADEChallengeData2016','images', subfolder_name, image_file_name))
      self._annotation_file_list.append(os.path.join(self.dir, 'ADEChallengeData2016','annotations', subfolder_name, anno_file_name))

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

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert(split_method == 'holdout')

    validation_dataet = ADE20K('val', self.dir)
    return self, validation_dataet


# vgg = ADE20K('train', '/opt/tiger/handdetJ/dataset/ade20k')
# size = vgg.size
# print(f'vgg size {size}')
# gt_label = 150
# label_num_map = {}
# for i in range(size):
#   data = vgg.sample(i)
#   # cv2.imwrite('./aabb_image.png', data['image'])
#   # cv2.imwrite('./aabb_segments.png', ((data['segments']/150)*255).astype(np.uint8))
#   ll = set(data['segments'].flatten().tolist())
#   for l in ll:
#     if l not in label_num_map:
#       label_num_map[l] = 0
#     label_num_map[l] += 1

# print(label_num_map)
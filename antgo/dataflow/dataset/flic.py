# -*- coding: UTF-8 -*-
# @Time    : 2019-06-08 15:51
# @File    : flic.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
import scipy.io as io
from antgo.dataflow.dataset import *
import os
import numpy as np
import cv2
import time
from antgo.framework.helper.fileio.file_client import *
from filelock import FileLock


__all__ = ['FLIC']
class FLIC(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(FLIC, self).__init__(train_or_test, dir, ext_params)
    
    if not os.path.exists(os.path.join(self.dir, 'examples.mat')):
      lock = FileLock('DATASET.lock')
      with lock:
        if not os.path.exists(os.path.join(self.dir, 'FLIC-full')):
          # 数据集不存在，需要重新下载，并创建标记
          ali = AliBackend()
          ali.download('ali:///dataset/flic/FLIC-full.zip', self.dir)
          os.system(f'cd {self.dir} && unzip FLIC-full.zip')

      self.dir = os.path.join(self.dir, 'FLIC-full')

    assert(train_or_test in ['train', 'val'])
    matr = io.loadmat(os.path.join(self.dir, 'examples.mat'))
    dataset = matr['examples'][0]

    self.ids = []
    self.files = []
    self.coords = []
    self.image_size = []
    self.torso = []

    self.valid_coords = [0,1,2,3,4,5,6,9,12,13,16]
    self.coords_name = ['Nose',
                        'Right Shoulder',
                        'Right Elbow',
                        'Right Wrist',
                        'Right Hip',
                        'Left Shoulder',
                        'Left Elbow',
                        'Left Wrist',
                        'Left Hip',
                        'Left Eye',
                        'Right Eye']
    count = 0
    for data in dataset:
      if train_or_test == 'train' and data[7][0][0]:
        # add train set
        self.ids.append(count)
        self.files.append(data[3][0])
        self.coords.append(data[2]) # 2 x 20
        self.image_size.append(data[4])
        self.torso.append(data[6])

        count += 1
      elif train_or_test == 'val' and data[8][0][0]:
        # add test set
        self.ids.append(count)
        self.files.append(data[3][0])
        self.coords.append(data[2]) # 2 x 20
        self.image_size.append(data[4])
        self.torso.append(data[6])

        count += 1

  @property
  def size(self):
    return len(self.ids)

  def at(self, id):
    img = cv2.imread(os.path.join(self.dir, 'images', self.files[id]))
    joints2d = np.transpose(self.coords[id][:,self.valid_coords], (1,0))
    bboxes = np.array([[
      np.min(joints2d[:, 0]), 
      np.min(joints2d[:, 1]), 
      np.max(joints2d[:, 0]), 
      np.max(joints2d[:, 1])
    ]])
    return (
      img, 
      {
        'joints2d': np.expand_dims(joints2d, 0),
        'joints_vis': np.ones((1, len(joints2d))),
        'bboxes': bboxes, 
        'labels': np.ones((1), dtype=np.int32),
        'image_meta': {
          'image_shape': (img.shape[0], img.shape[1])
        }        
      }
    )

  def split(self, split_params={}, split_method=''):
    raise NotImplementedError

# p2012 = FLIC('val', '/root/workspace/dataset/A')
# print(f'p2012 size {p2012.size}')
# for i in range(p2012.size):
#   data = p2012.sample(i)
#   # ss = result['segments']
#   image = data['image']
#   joints2d = data['joints2d']
#   for joint_i, (x,y) in enumerate(joints2d[0]):
#       x, y = int(x), int(y)
#       cv2.circle(image, (x, y), radius=2, color=(0,0,255), thickness=1)
#   cv2.imwrite('./1234.png', image.astype(np.uint8))

#   print(i)
# value = p2012.sample(0)
# print(value.keys())
# value = p2012.sample(1)
# print(value)
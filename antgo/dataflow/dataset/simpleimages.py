# -*- coding: UTF-8 -*-
# @Time    : 17-12-14
# @File    : simpleimages.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import *
import os
import copy
import cv2

__all__ = ['SimpleImages']


class SimpleImages(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(SimpleImages, self).__init__(train_or_test, dir, params)

    self.data_files = []
    self.label_files = []
    label_flag = getattr(self, 'flag', None)
    if os.path.exists(os.path.join(self.dir, self.train_or_test, 'data')) and \
        os.path.exists(os.path.join(self.dir, self.train_or_test, 'label')):
      # data , label (only support images)
      for data_file in os.listdir(os.path.join(self.dir, self.train_or_test, 'data')):
        if data_file.lower().split('.')[-1] in ['png', 'jpg', 'jpeg']:
          label_file = data_file if label_flag is None else '%s%s.%s'%(data_file.split('.')[0], label_flag, data_file.split('.')[-1])
          label_file_prefix = '.'.join(label_file.split('.')[0:-1])

          if os.path.exists(os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.png')):
            data_file_path = os.path.join(self.dir, self.train_or_test, 'data', data_file)
            label_file_path = os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.png')
            self.data_files.append(data_file_path)
            self.label_files.append(label_file_path)
          elif os.path.exists(os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.jpg')):
            data_file_path = os.path.join(self.dir, self.train_or_test, 'data', data_file)
            label_file_path = os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.jpg')
            self.data_files.append(data_file_path)
            self.label_files.append(label_file_path)
          elif os.path.exists(os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.jpeg')):
            data_file_path = os.path.join(self.dir, self.train_or_test, 'data', data_file)
            label_file_path = os.path.join(self.dir, self.train_or_test, 'label', label_file_prefix+'.jpeg')
            self.data_files.append(data_file_path)
            self.label_files.append(label_file_path)
    else:
      # data
      for data_file in os.listdir(os.path.join(self.dir, self.train_or_test)):
        if data_file.lower().split('.')[-1] in ['png', 'jpg', 'jpeg']:
          file_path = os.path.join(self.dir, self.train_or_test, data_file)
          self.data_files.append(file_path)
    
    assert(len(self.data_files) > 0)
    self.ids = [i for i in range(len(self.data_files))]
  
  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      ids = copy.copy(self.ids)
      if self.rng:
        self.rng.shuffle(ids)
      
      for id in ids:
        # img = imread(self.data_files[id])
        img = cv2.imread(self.data_files[id])
        if len(self.label_files) > 0:
          # label = imread(self.label_files[id])
          label = cv2.imread(self.label_files[id])
          seg_map = label
          if len(label.shape) == 3:
            seg_map = label[:,:,0]
          annotation = {'segmentation_map': seg_map, 'data':label, 'id': id, 'path': self.data_files[id]}
          yield [img, annotation]
        else:
          annotation = {'id': id, 'path': self.data_files[id]}
          yield [img, annotation]
        
  def at(self, id):
    # img = imread(self.data_files[id])
    img = cv2.imread(self.data_files[id])
    if len(self.label_files) > 0:
      # label = imread(self.label_files[id])
      label = cv2.imread(self.label_files[id])
      seg_map = label
      if len(label.shape) == 3:
        seg_map = label[:, :, 0]
      annotation = {'segmentation_map': seg_map, 'data':label, 'id': id, 'path': self.data_files[id]}
      return [img, annotation]
    else:
      annotation = {'id': id, 'path': self.data_files[id]}
      return [img, annotation]

  @property
  def size(self):
    return len(self.ids)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')

    return self, SimpleImages('val', self.dir, self.ext_params)

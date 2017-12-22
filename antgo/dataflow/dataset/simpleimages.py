# -*- coding: UTF-8 -*-
# @Time    : 17-12-14
# @File    : simpleimages.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import copy

__all__ = ['SimpleImages']


class SimpleImages(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(SimpleImages, self).__init__(train_or_test, dir, params)

    self.data_files = []
    self.label_files = []
    if os.path.exists(os.path.join(self.dir, self.train_or_test, 'data')) and \
        os.path.exists(os.path.join(self.dir, self.train_or_test, 'label')):
      # data , label (only support images)
      for data_file in os.listdir(os.path.join(self.dir, self.train_or_test, 'data')):
        if data_file.lower().split('.')[-1] in ['png', 'jpg', 'bmp', 'jpeg']:
          if os.path.exists(os.path.join(self.dir, self.train_or_test, 'label', data_file)):
            data_file_path = os.path.join(self.dir, self.train_or_test, 'data', data_file)
            label_file_path = os.path.join(self.dir, self.train_or_test, 'label', data_file)
            self.data_files.append(data_file_path)
            self.label_files.append(label_file_path)
      
      self.is_movie = False
    else:
      # data
      for data_file in os.listdir(os.path.join(self.dir, self.train_or_test)):
        if data_file.lower().split('.')[-1] in ['png', 'jpg', 'bmp', 'jpeg']:
          file_path = os.path.join(self.dir, self.train_or_test, data_file)
          self.data_files.append(file_path)
      
      self.is_movie = False
      if len(self.data_files) == 0:
        for data_file in os.listdir(os.path.join(self.dir, self.train_or_test)):
          if data_file.lower().split('.')[-1] in ['avi', 'mp4']:
            file_path = os.path.join(self.dir, self.train_or_test, data_file)
            self.data_files.append(file_path)
        
        if len(self.data_files) > 0:
          self.is_movie = True
    
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
      
      if not self.is_movie:
        for id in ids:
          img = imread(self.data_files[id])
          if len(self.label_files) > 0:
            label = imread(self.label_files[id])
            yield [img, label]
          else:
            yield [img]
      else:
        import cv2
        for id in ids:
          # read frame from movie
          cap = cv2.VideoCapture(self.data_files[id])
    
          while cap.isOpened():
            ret, frame = cap.read()
            b = frame[:, :, 0]
            g = frame[:, :, 1]
            r = frame[:, :, 2]
            rgb_frame = np.stack((r, g, b), 2)
            yield [rgb_frame]
        
  def at(self, id):
    if not self.is_movie:
      img = imread(self.data_files[id])
      if len(self.label_files) > 0:
        label = imread(self.label_files[id])
        return img, label
      else:
        return img
    else:
      raise NotImplementedError
    
  @property
  def size(self):
    return len(self.ids)
  
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
    for data_file in os.listdir(os.path.join(self.dir, self.train_or_test)):
      if data_file.lower().split('.')[-1] in ['png', 'jpg', 'bmp', 'jpeg']:
        file_path = os.path.join(self.dir, self.train_or_test, data_file)
        self.data_files.append(file_path)
        
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
        img = imread(self.data_files[id])
        yield [img]
        
  def at(self, id):
    img = imread(self.data_files[id])
    return img
    
  @property
  def size(self):
    return len(self.ids)
  
# -*- coding: UTF-8 -*-
# @Time    : 18-1-2
# @File    : zebra.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np
__all__ = ['Horse2Zebra']

ZEBRA_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip'
class Horse2Zebra(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(Horse2Zebra, self).__init__(train_or_test, dir)
    assert(train_or_test in ['train', 'test'])
    
    # 0.step maybe download
    if not os.path.exists(os.path.join(self.dir, 'horse2zebra')):
      self.download(self.dir, file_names=[], default_url=ZEBRA_URL, auto_unzip=True)
    
    # 1.step extract data list
    self.data_a_list = []
    self.data_b_list = []
    if train_or_test == 'train':
      for a in os.listdir(os.path.join(self.dir, 'horse2zebra', 'trainA')):
        if a[0] != '.':
          self.data_a_list.append(os.path.join(self.dir, 'horse2zebra', 'trainA', a))
      
      for b in os.listdir(os.path.join(self.dir, 'horse2zebra', 'trainB')):
        if b[0] != '.':
          self.data_b_list.append(os.path.join(self.dir, 'horse2zebra', 'trainB', b))
    else:
      for a in os.listdir(os.path.join(self.dir, 'horse2zebra', 'testA')):
        if a[0] != '.':
          self.data_a_list.append(os.path.join(self.dir, 'horse2zebra', 'testA', a))
  
      for b in os.listdir(os.path.join(self.dir, 'horse2zebra', 'testB')):
        if b[0] != '.':
          self.data_b_list.append(os.path.join(self.dir, 'horse2zebra', 'testB', b))
    
    self.ids = range(len(self.data_a_list))
    
  @property
  def size(self):
    return len(self.data_a_list)
    
  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1
    
      idxs = np.arange(len(self.ids))
      if self.rng:
        self.rng.shuffle(idxs)
      
      for k in idxs:
        a = self.data_a_list[k]
        random_b_index = np.random.randint(0, len(self.data_b_list), 1)[0]
        b = self.data_b_list[random_b_index]
        a_img = imread(a)
        b_img = imread(b)
        yield [a_img, b_img]
  
  def at(self, id):
    a = self.data_a_list[id]
    random_b_index = np.random.randint(0, len(self.data_b_list), 1)[0]
    b = self.data_b_list[random_b_index]
    a_img = imread(a)
    b_img = imread(b)
    return a_img, b_img
  
  def split(self, split_params={}, split_method=''):
    raise NotImplementedError

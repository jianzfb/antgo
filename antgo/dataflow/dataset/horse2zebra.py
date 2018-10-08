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
    assert(train_or_test in ['train', 'test', 'sample'])

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

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
    
    self.ids = list(range(len(self.data_a_list)))
    
  @property
  def size(self):
    return min(len(self.data_a_list), len(self.data_b_list))
    
  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1
    
      idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(idxs)

      b_idxs = list(range(len(self.data_b_list)))
      if self.rng:
        self.rng.shuffle(b_idxs)

      for a_k, b_k in zip(idxs, b_idxs):
        a = self.data_a_list[a_k]
        b = self.data_b_list[b_k]
        a_img = imread(a)
        b_img = imread(b)
        yield [a_img, b_img]
  
  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    a = self.data_a_list[id]
    random_b_index = np.random.randint(0, len(self.data_b_list), 1)[0]
    b = self.data_b_list[random_b_index]
    a_img = imread(a)
    b_img = imread(b)
    return a_img, b_img
  
  def split(self, split_params={}, split_method=''):
    raise NotImplementedError

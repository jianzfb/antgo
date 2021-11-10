# -*- coding: UTF-8 -*-
# @Time    : 2021/11/9 10:25 下午
# @File    : proxy_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from antgo.dataflow.dataset.dataset import *


class ProxyDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(ProxyDataset, self).__init__(train_or_test, dir, params)
    self.train_data = None
    self.val_data = None
    self.test_data = None

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert(split_method == 'holdout')
    val_pd = ProxyDataset('val', self.dir, self.ext_params)
    val_pd.register(None, self.val_data, None)
    return self, val_pd

  @property
  def size(self):
    if self.train_or_test == 'train':
      return len(self.train_data)
    elif self.train_or_test == 'val':
      return len(self.val_data)
    else:
      return len(self.test_data)

  def at(self, index):
    if self.train_or_test == 'train':
      return self.train_data[index]
    elif self.train_or_test == 'val':
      return self.val_data[index]
    else:
      return self.test_data[index]

  def data_pool(self):
    for index in range(self.size):
      yield self.at(index)
  
  def register(self, train=None, val=None, test=None):
    self.train_data = train
    self.val_data = val
    self.test_data = test

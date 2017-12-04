# -*- coding: UTF-8 -*-
# Time: 12/2/17
# File: tfdataset.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .dataset import Dataset

__all__ = ['TFDataset']

class TFDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(TFDataset, self).__init__(train_or_test,dir, params)
    pass

  def data_pool(self):
    pass

  def at(self,id):
    pass

  def split(self, split_params={}, split_method=""):
    pass


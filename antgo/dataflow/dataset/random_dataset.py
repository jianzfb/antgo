# -*- coding: UTF-8 -*-
# @Time    : 2018/11/9 5:50 PM
# @File    : random_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.dataset.dataset import Dataset
from antgo.utils import logger
import os
import copy
import sys
import time

class RandomDataset(Dataset):
  def __init__(self, train_or_test, dir, params={}):
    super(RandomDataset, self).__init__(train_or_test, dir, params)
    self._data_func = None
    logger.info('initialize random dataset')

  @property
  def size(self):
    return 10000

  def at(self, id):
    raise NotImplementedError

  def data_pool(self):
    assert(self._data_func is not None)

    for _ in range(self.size):
      data = self._data_func()
      yield data

  @property
  def data_func(self):
    return self._data_func

  @data_func.setter
  def data_func(self, val):
    self._data_func = val
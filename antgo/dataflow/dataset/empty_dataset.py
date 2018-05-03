# -*- coding: UTF-8 -*-
# @Time    : 18-5-3
# @File    : empty_dataset.py
# @Author  : 
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import Dataset
import os
import copy
import sys
import time


class EmptyDataset(Dataset):
  def __init__(self, train_or_test, dir, params={}):
    super(EmptyDataset, self).__init__(train_or_test, dir, params)

  @property
  def size(self):
    return 0

  def at(self, id):
    raise NotImplementedError

  def data_pool(self):
    time.sleep(sys.maxsize)
    yield None, None
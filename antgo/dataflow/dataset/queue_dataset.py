# -*- coding: UTF-8 -*-
# @Time : 2018/4/28
# @File : queue_dataset.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.dataset.dataset import Dataset
import os
import copy
import multiprocessing


class QueueDataset(Dataset):
  def __init__(self):
    super(QueueDataset, self).__init__('test', '', None)
    self._data_queue = multiprocessing.Queue()

  @property
  def size(self):
    return 10000000000

  @property
  def data_queue(self):
    return self._data_queue

  def at(self, id):
    raise NotImplementedError

  def data_pool(self):
    while True:
      data = self.data_queue.get()
      yield data
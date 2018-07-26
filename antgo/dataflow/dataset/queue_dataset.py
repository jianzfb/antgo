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
import sys
import zmq
from antgo.utils.serialize import *


class QueueDataset(Dataset):
  def __init__(self):
    super(QueueDataset, self).__init__('test', '', None)
    context = zmq.Context()
    self.socket = context.socket(zmq.REP)
    self.socket.connect('ipc://%s'%str(os.getpid()))

  @property
  def size(self):
    return sys.maxsize

  def at(self, id):
    raise NotImplementedError

  def put(self, data):
    self.socket.send(dumps(data))

  def data_pool(self):
    while True:
      data = self.socket.recv()
      data = loads(data)
      yield data
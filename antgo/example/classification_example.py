# -*- coding: UTF-8 -*-
# @Time    : 17-8-29
# @File    : classification_example.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from antgo.dataflow.dataset.heart import *

ctx = Context()

# heart_data = Heart('train', '/home/mi/dataset/heart/')
# batch_data = BatchData(Node.inputs(heart_data), 2, buffer_size=3)
#
# for epoch in range(10):
#   count = 0
#   for data in batch_data.iterator_value():
#     a, b = data
#     print(count)
#     count += 1
#
#   print('epoch %d'%epoch)
#
# ctx.wait_until_clear()


def train_callback(data_source, dump_dir):
  pass


def infer_callback(data_source, dump_dir):
  pass


ctx.training_process = train_callback
ctx.infer_process = infer_callback
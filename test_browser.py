# -*- coding: UTF-8 -*-
# @Time    : 2022/8/31 23:06
# @File    : test_browser.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import sys

# sys.path.append(os.path.dirname(__file__))
from antgo.dataflow.common import *
from antgo.interactcontext import *
from antgo.dataflow.dataset import *
import antvis.client.mlogger as mlogger
import numpy as np
import torch

# 全局上下文
# class Processor1(ActionNode):
#   def __init__(self, inputs=None):
#     super(Processor1, self).__init__(inputs=inputs)
#
#   def action(self, *args, **kwargs):
#     image, _ = args
#     # 添加一些数据处理
#     processed_image = ...
#     return processed_image
#

# class Processor2(ActionNode):
#   def __init__(self, inputs=None):
#     super(Processor2, self).__init__(inputs=inputs)
#
#   def action(self, *args, **kwargs):
#     image, = args
#     # 转换到可输出数据格式，'...':{'data': ..., 'type': ...}
#     data = {
#       'A': {
#         'data': image,
#         'type': 'IMAGE'
#       }
#     }
#     return data
#
# ctx.data_processor.add(Processor2)

ctx = InteractContext()

if __name__ == '__main__':
  with ctx.Browser('WWW', {}, browser={'size': 10}) as browser:
    for index in range(10):
      data = {
        'A': {
          'data': np.random.randint(0,255,(255,255), dtype=np.uint8),
          'type': 'IMAGE',
        },
        'B': {
          'data': 'hello',
          'type': 'STRING',
        },
        'image_file': f'hello_{index}.png'
      }
      browser.context.recorder.record(data)
    print('ssd')
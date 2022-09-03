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
  with ctx.Activelearning('WWW', {}, activelearning={
    'white_users': {
      'jian@baidu.com':{
        'password': '112233'
      }
    },
    'label_type': 'RECT',
    'label_metas':{
      'label_category': [
        {
          'class_name': 'A',
          'class_index': 0,
          'color': 'green',
          'background_color': '#00800026'
        },
        {
          'class_name': 'B',
          'class_index': 1,
          'color': 'blue',
          'background_color': '#0000ff26'
        },
      ],
    }
  }) as predict:
    round = 0
    while True:
      # 1.step 训练..

      # 2.step 采样..

      # 3.step 推送平台，等待结束
      unlabeled_samples = [1,2]
      for sample_i, sample in enumerate(unlabeled_samples):
        data = {
          'image': np.random.randint(0, 255, (255, 255), dtype=np.uint8),  # 第二优先级
          'label_info': [],
          'id': sample_i
        }

        predict.context.recorder.record(data)

      # dengdai fanhui
      # 启动并等待
      predict.context.recorder.start(round=round)
      round += 1
      pass

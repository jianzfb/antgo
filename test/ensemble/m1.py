# -*- coding: UTF-8 -*-
# @Time    : 2022/4/21 11:59
# @File    : m1.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.interactcontext import *
import numpy as np

def main():
  ctx = InteractContext()

  with ctx.Demo('A', {
    'system': {
      'ip': '127.0.0.1',
      'port': 8901},
    'demo': {
      'support_user_upload': True,
      'support_user_constraint': {
        'file_type': ['mp4']
      }
    }
  }, dataset='M') as demo:

    processed_data = []
    for data, annotation in demo.running_dataset.iterator_value():
      if annotation['frame_index'] == 0:
        # 这是一个新视频
        processed_data = []

      processed_data.append(data)

      if annotation['frame_index'] == annotation['frame_num'] - 1:
        # 结束
        ctx.recorder.record({
          "RESULT": {
            'data': processed_data,
            'type': "VIDEO",
            'id': annotation['id']
          }
        })


if __name__ == '__main__':
  main()


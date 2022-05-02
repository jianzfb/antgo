# -*- coding: UTF-8 -*-
# @Time    : 2022/4/20 17:53
# @File    : t1.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.interactcontext import *
import numpy as np

def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)


def main():
  ctx = InteractContext()

  with ctx.Ensemble('A', {
    'system': {
      'ip': '127.0.0.1',
      'port': 8903},
    'ensemble': {
      'method': 'bagging',
      'mode': 'online',
      'role': 'master',
      'worker': 2,
      'dataset': '',
      'weight': 1.0,
      'uncertain_vote': {
        'axis': 1,
        'thres': 2.1
      },
      'enable_data_record': True,
      'model_name': 'Swin',
      'feedback': False
    }
  }, 'merge', dataset='T') as ensemble:
    for i in range(10):
      # data = np.random.random((10, 10))
      data = np.random.random((1,11,720,1280))
      data = softmax(data, 1)
      online_fusion_A = ctx.recorder.avg({
        'id': {
          'data': i
        },
        'A': {
          'data': data
        }
      })

      print(online_fusion_A)
      # 保存结果

if __name__ == '__main__':
  main()

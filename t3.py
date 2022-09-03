# -*- coding: UTF-8 -*-
# @Time    : 2022/4/20 17:54
# @File    : t2.py
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
        'thres': -1
      },
      'enable_data_record': False,
      'model_name': 'r50',
      'feedback': False,
      'uuid': '015924b3-b8a8-4bfb-927d-00eaf64a0b68'
    }
  }, 'release', dataset='T', token='8d05781fad92480486812bbb75fd2fd7') as ensemble:
    for i in range(10):
      result = ctx.recorder.get({
        'id': {
          'data': i
        },
        'A': {
          'data': None
        }
      })

      print(result)

if __name__ == '__main__':
  main()

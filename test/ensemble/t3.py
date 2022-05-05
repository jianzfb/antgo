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
      'uuid': 'b2f5cbb9-6efd-4878-8496-0955acbd4e3d'
    }
  }, 'release', dataset='T', token='5dff9aaf18774b688221eedef367dc31') as ensemble:
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

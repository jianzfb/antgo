# encoding=utf-8
# @Time    : 17-7-25
# @File    : matting_task.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *
from antgo.dataflow.common import *
from antgo.measures.error import *

default = {'AntSADMatting': ('MATTING-SAD', 'MATTING'),
           'AntMSEMatting': ('MATTING-MSE', 'MATTING'),
           'AntGradientMatting': ('MATTING-GRADIENT', 'MATTING')}
class AntSADMatting(AntMeasure):
  def __init__(self, task):
    super(AntSADMatting, self).__init__(task, 'MATTING-SAD')
    assert (task.task_type == 'MATTING')

    self.is_support_rank = True

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    count = 0
    sad = 0.0
    for predict, gt in data:
      assert(len(predict.shape) == 2)
      assert(len(gt.shape) == 2)

      sad += np.sum(np.abs(predict - gt))
      count += 1

    val = sad / count
    return {'statistic':{'name':self.name, 'value':[{'name':self.name, 'value': val, 'type': 'SCALAR'}]}}


class AntMSEMatting(AntMeasure):
  def __init__(self, task):
    super(AntMSEMatting, self).__init__(task, 'MATTING-MSE')
    assert (task.task_type == 'MATTING')

    self.is_support_rank = True

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    count = 0
    res = 0.0
    for predict, gt in data:
      assert(len(predict.shape) == 2)
      assert(len(gt.shape) == 2)

      res += mse(gt, predict)
      count += 1

    val = res / count
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]}}


class AntBoundaryMSEMatting(AntMeasure):
  def __init__(self, task):
    pass

  def eva(self, data, label):
    pass


class AntGradientMatting(AntMeasure):
  def __init__(self, task):
    # paper: Christoph Rhemann, etc. A Perceptually Motivated Online Benchmark for Image Matting
    super(AntGradientMatting, self).__init__(task, 'MATTING-GRADIENT')
    assert (task.task_type == 'MATTING')
    # delta = 1.4, q = 2

    self.is_support_rank = True

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    count = 0
    res = 0.0
    for predict, gt in data:
      assert(len(predict.shape) == 2)
      assert(len(gt.shape) == 2)

      predict_grad = scipy.ndimage.filters.gaussian_filter(predict, 1.4, order=1)
      gt_grad = scipy.ndimage.filters.gaussian_filter(gt, 1.4, order=1)
      res += np.sum(np.power(predict_grad - gt_grad, 2))
      count += 1

    val = res / count
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]}}


class AntConnectivityMatting(AntMeasure):
  def __init__(self, task):
    # paper: Christoph Rhemann, etc. A Perceptually Motivated Online Benchmark for Image Matting
    super(AntConnectivityMatting, self).__init__(task, 'MATTING-CONNECTIVITY')
    assert (task.task_type == 'MATTING')
    # theta=0.15, p=1

    self.is_support_rank = True

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    count = 0
    res = 0.0
    for predict, gt in data:
      assert(len(predict.shape) == 2)
      assert(len(gt.shape) == 2)

      count += 1

    val = 0.0
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': val, 'type': 'SCALAR'}]}}


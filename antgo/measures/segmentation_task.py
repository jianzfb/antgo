# encoding=utf-8
# @Time    : 17-7-12
# @File    : segmentation_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *
from antgo.dataflow.common import *
from antgo.utils._resize import *
from scipy import signal
from scipy import ndimage
import cv2

default = {'AntMeanIOUSeg': ('MeanIOU', 'SEGMENTATION'),
           'AntMeanIOUBoundary': ('MeanIOUBoundary', 'SEGMENTATION'),
           }

class AntMeanIOUSeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUSeg, self).__init__(task, 'MeanIOU')
    assert(task.task_type == 'SEGMENTATION')

    self.is_support_rank = True

  def eva(self, data, label):
    classes_num = len(self.task.class_label)

    if label is not None:
      data = zip(data, label)

    sample_scores = []

    total_score = 0.0
    total_num = 0
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']

      gt_labels = set(gt.flatten())

      statistic_score = 0.0
      statistic_class_num = 0
      for l in gt_labels:
        l = int(l)

        _gt = np.zeros(gt.shape)
        _predict = np.zeros(predict.shape)

        _gt[np.where(gt == l)] = 1.0
        _predict[np.where(predict == l)] = 1.0

        if np.sum(_gt) < 1.0:
          continue

        intersection = np.sum(_gt * _predict)
        union = np.sum(_gt) + np.sum(_predict) - intersection + 0.0000000001

        l_score = intersection / union
        statistic_score += l_score
        statistic_class_num += 1

        if id is not None:
          sample_scores.append({'id': id, 'score': l_score, 'category': l})

      statistic_score /= statistic_class_num
      total_score += statistic_score
      total_num += 1

    avg_score = total_score / total_num

    avg_score = float(avg_score)
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': avg_score, 'type':'SCALAR'}]},
            'info': sample_scores}


class AntMeanIOUBoundary(AntMeasure):
  def __init__(self, task):
    # paper: Liang-Chieh Chen, etc. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUBoundary, self).__init__(task, 'MeanIOUBoundary')
    assert(task.task_type == 'SEGMENTATION')
    self.is_support_rank = True
  
  def eva(self, data, label):
    classes_num = len(self.task.class_label)
    trimap_width = int(getattr(self.task, 'MeanIOUBoundary_trimap_width', 3))

    if label is not None:
        data = zip(data, label)

    sample_scores = []
    total_score = 0.0
    total_num = 0
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']
      
      gt_labels = set(gt.flatten())
      statistic_score = 0.0
      statistic_class_num = 0
      for l in gt_labels:
        l = int(l)
        if l == 0:
          # class 0 consider as ignore class
          continue

        _gt = np.zeros(gt.shape)
        _predict = np.zeros(predict.shape)

        _gt[np.where(gt == l)] = 1.0
        _predict[np.where(predict == l)] = 1.0

        if np.sum(_gt) < 1.0:
          continue

        # generate gt object (class l) edge
        _gt_blur = ndimage.gaussian_filter(_gt, sigma=7)
        _gt_blur[np.where(_gt_blur > 0.99)] = 0.0
        _gt_blur[np.where(_gt_blur < 0.01)] = 0.0
        _gt_blur[np.where(_gt_blur > 0.0)] = 1.0
        if np.sum(_gt_blur) < 1.0:
          continue

        _gt_edge_mask = _gt_blur

        # generate predict object (class l) edge
        _predict_blur = ndimage.gaussian_filter(_predict, sigma=7)
        _predict_blur[np.where(_predict_blur > 0.99)] = 0.0
        _predict_blur[np.where(_predict_blur < 0.01)] = 0.0
        _predict_blur[np.where(_predict_blur > 0.0)] = 1.0

        _predict_edge_mask = _predict_blur

        _gt_edge = _gt * _gt_edge_mask
        _predict_edge = _predict * _predict_edge_mask

        intersection = np.sum(_gt_edge * _predict_edge)
        union = np.sum(_gt_edge) + np.sum(_predict_edge) - intersection + 0.0000000001

        l_score = intersection / union
        statistic_score += l_score
        statistic_class_num += 1

        if id is not None:
          sample_scores.append({'id': id, 'score': l_score, 'category': l})

      if statistic_class_num == 0:
        continue

      statistic_score /= statistic_class_num
      total_score += statistic_score
      total_num += 1

    avg_score = total_score / total_num
    avg_score = float(avg_score)
    return {'statistic': {'name': self.name, 'value': [{'name': self.name, 'value': avg_score, 'type':'SCALAR'}]},
            'info': sample_scores}
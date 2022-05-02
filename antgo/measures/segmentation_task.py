# encoding=utf-8
# @Time    : 17-7-12
# @File    : segmentation_task.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import sys
import numpy as np
from antgo.task.task import *
from antgo.measures.base import *
from antgo.dataflow.common import *
from scipy import signal
from scipy import ndimage
from scipy.sparse import csr_matrix


default = {'AntMeanIOUSeg': ('MeanIOU', 'SEGMENTATION'),
           'AntMeanIOUBoundary': ('MeanIOUBoundary', 'SEGMENTATION'),
           }

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix


class AntMeanIOUSeg(AntMeasure):
  def __init__(self, task):
    # paper: Jonathan Long, Evan Shelhamer, etc. Fully Convolutional Networks for Semantic Segmentation
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUSeg, self).__init__(task, 'MeanIOU')
    assert(task.task_type == 'SEGMENTATION')
    self.classes_num = len(self.task.class_label)
    self.is_support_rank = True

  def _fast_hist(self, label_pred, label_true):
    # 找出标签中需要计算的类别,去掉了背景
    mask = (label_true >= 0) & (label_true < self.classes_num)

    # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    hist = np.bincount(
      self.classes_num * label_true[mask].astype(np.int) +
      label_pred[mask].astype(np.int), minlength=self.classes_num ** 2).reshape(self.classes_num, self.classes_num)
    return hist

  def eva(self, data, label):
    if label is not None:
      data = zip(data, label)

    sample_scores = []
    confusion = np.zeros((self.classes_num, self.classes_num))
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']

      sample_confusion = self._fast_hist(np.array(predict).flatten(), np.array(gt).flatten())
      confusion += sample_confusion

      if id is not None:
        sample_iou = np.diag(sample_confusion) / np.maximum(1.0, sample_confusion.sum(axis=1) + sample_confusion.sum(axis=0) - np.diag(sample_confusion))
        sample_miou_val = np.nanmean(sample_iou)
        sample_scores.append({'id': id, 'score': sample_miou_val, 'category': 0})

    iou = np.diag(confusion) / np.maximum(1.0,confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    miou_val = float(np.nanmean(iou))

    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': miou_val, 'type':'SCALAR'}]},
            'info': sample_scores}


class AntMeanIOUBoundary(AntMeasure):
  def __init__(self, task):
    # paper: Liang-Chieh Chen, etc. Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs
    # formular: (1/n_{cl}) / \sum_i n_{ii}/(t_i+\sum_j n_{ji}-n_{ii})

    super(AntMeanIOUBoundary, self).__init__(task, 'MeanIOUBoundary')
    assert(task.task_type == 'SEGMENTATION')
    self.is_support_rank = True
    self.classes_num = len(self.task.class_label)

  def _fast_hist(self, label_pred, label_true):
    # 找出标签中需要计算的类别,去掉了背景
    mask = (label_true >= 0) & (label_true < self.classes_num)

    # # np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    hist = np.bincount(
      self.classes_num * label_true[mask].astype(int) +
      label_pred[mask].astype(int), minlength=self.classes_num ** 2).reshape(self.classes_num, self.classes_num)
    return hist

  def eva(self, data, label):
    trimap_width = int(getattr(self.task, 'MeanIOUBoundary_trimap_width', 3))
    if label is not None:
        data = zip(data, label)

    sample_scores = []
    confusion = np.zeros((self.classes_num, self.classes_num))
    for predict, gt in data:
      id = None
      if type(gt) == dict:
        if 'segmentation_map' not in gt:
          continue

        id = gt['id']
        gt = gt['segmentation_map']

      # finding object edge, ignore class 0
      ignore_gt = np.array(gt).astype(np.int)

      # ignore non edge region
      for l in range(1, self.classes_num):
        # finding edge region
        _gt = np.zeros(gt.shape)
        _gt[np.where(ignore_gt == l)] = 1.0

        _gt_blur = ndimage.gaussian_filter(_gt, sigma=7)
        _gt_blur[np.where(_gt_blur > 0.99)] = 0.0
        _gt_blur[np.where(_gt_blur < 0.01)] = 0.0
        _gt_blur[np.where(_gt_blur > 0.0)] = 1.0

        # ignore non edge region
        ignore_gt[np.where(_gt_blur < 1.0)] = 255

      sample_confusion = self._fast_hist(np.array(predict).flatten(), ignore_gt.flatten())
      confusion += sample_confusion

      if id is not None:
        sample_iou = np.diag(sample_confusion) / np.maximum(1.0, sample_confusion.sum(axis=1) + sample_confusion.sum(axis=0) - np.diag(sample_confusion))
        sample_miou_val = float(np.nanmean(sample_iou))
        sample_scores.append({'id': id, 'score': sample_miou_val, 'category': 0})

    iou = np.diag(confusion) / np.maximum(1.0, confusion.sum(axis=1) + confusion.sum(axis=0) - np.diag(confusion))
    miou_val = float(np.nanmean(iou))
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': miou_val, 'type':'SCALAR'}]},
            'info': sample_scores}
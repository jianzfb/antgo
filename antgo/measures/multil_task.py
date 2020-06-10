# -*- coding: UTF-8 -*-
# @Time : 2018/8/13
# @File : multil_task.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
import numpy as np

default = {'AntAccuracyMultiL': ('MULTILABEL_ACCURACY', 'MULTILABEL'),
           'AntHammingLossMultiL': ('MULTILABEL_HAMMINGLOSS', 'MULTILABEL')}

class AntAccuracyMultiL(AntMeasure):
  def __init__(self, task):
    super(AntAccuracyMultiL, self).__init__(task, 'MULTILABEL_ACCURACY')
    assert(task.task_type == 'MULTILABEL')
    self.is_support_rank = True

  def eva(self, data, label):
    '''

    :param data: (list) predict (N x [])
    :param label: (list) ground truth label (N x [])
    :return:
    '''

    if label is not None:
      data = zip(data, label)

    count = 0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1_measure = 0.0
    sample_scores = []

    for predict, gt in data:
      predict_set = set(np.array(predict).astype(np.uint32).tolist())

      gt_label = gt
      id = None
      if type(gt) == dict:
        gt_label = gt['category_id']
        id = gt['id']

      gt_set = set(np.array(gt_label).astype(np.uint32))

      and_num = len(predict_set & gt_set)
      or_num = len(predict_set | gt_set)

      accuracy += float(and_num) / float(or_num)
      precision += float(and_num) / float(len(predict_set))
      recall += float(and_num) / float(len(gt_set))
      f1_measure += (2.0 * float(and_num)) / (float(len(gt_set)) + float(len(predict_set)))

      if id is not None:
        score_list = [accuracy, precision, recall, f1_measure]
        s = score_list[self.support_rank_index]
        sample_scores.append({'id': id, 'score': s, 'category': gt_label})

      count += 1

    accuracy = accuracy / float(count)
    precision = precision / float(count)
    recall = recall / float(count)
    f1_measure = f1_measure / float(count)

    return {'statistic': {'name': self.name,
                          'value': [{'name': 'Accuracy', 'value': accuracy, 'type': 'SCALAR'},
                                    {'name': 'Precision', 'value': precision, 'type': 'SCALAR'},
                                    {'name': 'Recall', 'value': recall, 'type': 'SCALAR'},
                                    {'name': 'F1', 'value': f1_measure, 'type': 'SCALAR'}]},
            'info': sample_scores}


class AntHammingLossMultiL(AntMeasure):
  def __init__(self, task):
    super(AntHammingLossMultiL, self).__init__(task, 'MULTILABEL_HAMMINGLOSS')
    assert(task.task_type == 'MULTILABEL')
    self.is_support_rank = True

  def eva(self, data, label):
    '''

    :param data: (list) predict (N x [])
    :param label: (list) ground truth label (N x [])
    :return:
    '''

    if label is not None:
      data = zip(data, label)

    count = 0
    score = 0.0
    sample_scores = []

    for predict, gt in data:
      gt_label = gt
      id = None
      if type(gt) == dict:
        gt_label = gt['category_id']
        id = gt['id']

      # transform to onehot
      onehot_predict = np.zeros((len(self.task.class_label)), np.int32)
      onehot_predict[predict] = 1

      onehot_gt = np.zeros(len(self.task.class_label))
      onehot_gt[gt_label] = 1

      num = np.count_nonzero(onehot_predict != onehot_gt)
      score += num

      if id is not None:
        sample_scores.append({'id': id, 'score': score, 'category': gt_label})

      count += 1

    score = score / float(count)
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': score, 'type': 'SCALAR'},]},
            'info': sample_scores}
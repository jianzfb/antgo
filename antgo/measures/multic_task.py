# encoding=utf-8
# @Time    : 17-5-4
# @File    : multic_task.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
from antgo.measures.multi_c import *
from antgo.measures.confusion_matrix import *
import numpy as np


default = {'AntAccuracyMultiC': ('ACCURACY', 'CLASSIFICATION'),
           'AntConfusionMatrixMultiC': ('CONFUSION-MATRIX', 'CLASSIFICATION')}
class AntAccuracyMultiC(AntMeasure):
  def __init__(self, task):
    super(AntAccuracyMultiC, self).__init__(task, 'ACCURACY')
    assert(task.task_type == 'CLASSIFICATION')
    self.is_support_rank = True
    self.larger_is_better = 1

  def eva(self, data, label):
    '''
    :param data: logits (N x class_num)
    :param label: ground truth label (0 ~ ...) (N,)
    :return: accuracy
    '''
    if label is not None:
      data = zip(data, label)

    acutal_label = []
    predicated_label = []
    sample_scores = []
    for predict, gt in data:
      predicated_label.append(predict)

      id = None
      gt_label = gt
      if type(gt) == dict:
        gt_label = int(gt['category_id'])
        id = gt['id']

      if id is not None:
        s = 1 if int(predict) == int(gt_label) else 0
        sample_scores.append({'id': id, 'score': int(s), 'category': gt_label})
      acutal_label.append(gt_label)

    accuracy = multi_accuracy_labels(acutal_label, predicated_label)
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': float(accuracy), 'type': 'SCALAR'}]},
            'info': sample_scores}

    # curve_data = [[[0,0],[1,1],[2,2]]]
    # bar_data = [10,5,2]
    # return {'statistic': {'name': self.name,
    #                       'value': [{'name': self.name, 'value': float(accuracy), 'type': 'SCALAR'},
    #                                 {'name': 'test-curve', 'value': curve_data,
    #                                  'type': 'CURVE', 'x': 'x', 'y': 'y'},
    #                                 {'name': 'test-bar', 'value': bar_data,
    #                                  'type': 'SCALAR', 'x': ['a', 'b', 'c'], 'y': 'y'}]}}


class AntConfusionMatrixMultiC(AntMeasure):
  def __init__(self, task):
    super(AntConfusionMatrixMultiC, self).__init__(task, 'CONFUSION-MATRIX')
    assert(task.task_type == 'CLASSIFICATION')

  def eva(self, data, label):
    '''
    :param data: logits (N x class_num)
    :param label: ground truth label (0 ~ ...) (N,)
    :return: matrix
    '''
    if label is not None:
      data = zip(data, label)

    acutal_label = []
    predicated_label = []
    for predict, gt in data:
      predicated_label.append(int(predict))

      gt_label = -1
      if type(gt) == dict:
        gt_label = int(gt['category_id'])
      else:
        gt_label = int(gt)

      acutal_label.append(gt_label)

    class_num = len(self.task.class_label)
    cm = compute_confusion_matrix(acutal_label, predicated_label, class_num)

    # ######
    # curve_data = [[[0,0],[1,1],[2,2]]]
    # bar_data = [10,5,2]
    # image = np.random.random((50,50,3))
    # image = image*255
    # image = image.astype(np.uint8)
    # {'name': 'test-curve', 'value': curve_data,
    #  'type': 'CURVE', 'x': 'x', 'y': 'y'},
    # {'name': 'test-bar', 'value': bar_data,
    #  'type': 'SCALAR', 'x': ['a', 'b', 'c'], 'y': 'y'},
    # {'name': 'test-image', 'value': image,
    #  'type': 'IMAGE'}]
    # ######

    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': cm.tolist(),
                                     'type': 'MATRIX', 'x': self.task.class_label, 'y': self.task.class_label}
                                    ]}}

    # return {'statistic': {'name': self.name,
    #                       'value': [{'name': self.name, 'value': cm.tolist(),
    #                                  'type': 'MATRIX', 'x': self.task.class_label, 'y': self.task.class_label},
    #                                 {'name': 'test-curve', 'value': curve_data,
    #                                  'type': 'CURVE', 'x': 'x', 'y': 'y'},
    #                                 {'name': 'test-bar', 'value': bar_data,
    #                                 'type': 'SCALAR', 'x': ['a', 'b', 'c'], 'y': 'y'}]}}
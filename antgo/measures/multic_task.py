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


class AntAccuracyMultiC(AntMeasure):
  def __init__(self, task):
    super(AntAccuracyMultiC, self).__init__(task, 'ACCURACY')
    assert(task.task_type == 'CLASSIFICATION')
    self.is_support_rank = True

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
        gt_label = int(gt['data'])
        id = gt['id']

      if id is not None:
        s = 1 if int(predict) == int(gt_label) else 0
        sample_scores.append({'id': id, 'score': int(s), 'category': gt_label})
      acutal_label.append(gt_label)

    accuracy = multi_accuracy_labels(acutal_label, predicated_label)
    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': float(accuracy), 'type': 'SCALAR'}]},
            'info': sample_scores}


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
        gt_label = int(gt['data'])
      else:
        gt_label = int(gt)

      acutal_label.append(gt_label)

    class_num = len(self.task.class_label)
    cm = compute_confusion_matrix(acutal_label, predicated_label, class_num)

    # test-1 (histgram data)
    histgram_data = {'name': 'test-hist', 'value':[5,15,2,15], 'type': 'SCALAR', 'x': 'x', 'y':'y'}

    # test-2 (curve data)
    curve_data = {'name': 'test-curve', 'value':[[[0,0],[1,5],[2,15]]], 'type':'CURVE', 'x':'X-TT', 'y':'Y-HH', 'z': ['category-1']}

    # test-3 (image data)
    random_img = np.random.random((50,50,3))
    random_img = random_img * 255
    random_img = random_img.astype(np.uint8)
    image_data = {'name': 'test-image', 'value':random_img, 'type':'IMAGE'}

    # test-4 (Table data)
    table_data = {'name': 'test-table', 'value':[['-','hello','world'],[0,1,2]], 'type':'TABLE'}

    return {'statistic': {'name': self.name,
                          'value': [{'name': self.name, 'value': cm.tolist(),
                                     'type': 'MATRIX', 'x': self.task.class_label, 'y': self.task.class_label},
                                    histgram_data, curve_data,image_data,table_data]}}
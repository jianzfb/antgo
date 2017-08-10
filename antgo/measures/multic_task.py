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
        for predict, gt in data:
            predicated_label.append(predict)
            acutal_label.append(gt)

        accuracy = multi_accuracy(acutal_label, predicated_label)
        return {'statistic': {'name': self.name,
                              'value': [{'name': self.name, 'value': accuracy, 'type': 'SCALAR'}]},
                'info': {'label': self.task.class_label}}


class AntPerAccuracyMultiC(AntMeasure):
    def __init__(self, task):
        super(AntPerAccuracyMultiC, self).__init__(task, 'PER-ACCURACY')
        assert(task.task_type == 'CLASSIFICATION')

        self.is_support_rank = True

    def eva(self, data, label):
        '''
        :param data: logits (N x class_num)
        :param label: ground truth label (0 ~ ...) (N,)
        :return: accuracy of per class
        '''
        if label is not None:
            data = zip(data, label)

        acutal_label = []
        predicated_label = []
        for predict, gt in data:
            predicated_label.append(predict)
            acutal_label.append(gt)

        per_accuracy = per_class_accuracy(acutal_label, predicated_label)

        return {'statistic': {'name': self.name,
                              'value': [{'name': self.name, 'value':per_accuracy,
                                         'type': 'SCALAR', 'x': 'class', 'y': 'accuracy per class'}]},
                'info': {'label': self.task.class_label}}


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
            predicated_label.append(predict)
            acutal_label.append(gt)

        cm = confusion_matrix(acutal_label, predicated_label)

        return {'statistic': {'name': self.name,
                              'value': [{'name': self.name, 'value': cm.tolist(),
                                         'type': 'MATRIX', 'x': 'class', 'y': 'class'}]},
                'info': {'label': self.task.class_label}}
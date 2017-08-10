# encoding=utf-8
# @Time    : 17-5-4
# @File    : regression_task.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *
from antgo.measures.base import *
from antgo.measures.regression_metric import *


class AntMAPERegression(AntMeasure):
    def __init__(self, task):
        super(AntMAPERegression, self).__init__(task, 'MAPE')
        assert(task.task_type == 'REGRESSION')

        self.is_support_rank = True

    def eva(self, data, label):
        '''
        :param data: predicate value (N,)
        :param label: ground truth value (N,)
        :return: 
        '''
        # assert(data.shape[0] == label.shape[0])
        if label is not None:
            data = zip(data, label)

        acutal_s = []
        predicated_s = []
        for predict, gt in data:
            predicated_s.append(predict)
            acutal_s.append(gt)

        error = mape(actual_s=acutal_s, predicated_s=predicated_s)

        return {'statistic': {'name': self.name,
                              'value': [{'name':self.name, 'value': error}]}}


class AntAlmostCRegression(AntMeasure):
    def __init__(self, task):
        super(AntAlmostCRegression, self).__init__(task,'ALMOST-CORRECT')
        assert(task.task_type == 'REGRESSION')

        self.is_support_rank = True

    def eva(self, data, label):
        '''
        :param data: predicate value (N,)
        :param label: ground truth value (N,)
        :return: 
        '''
        # assert(data.shape[0] == label.shape[0])
        if label is not None:
            data = zip(data, label)

        acutal_s = []
        predicated_s = []
        for predict, gt in data:
            predicated_s.append(predict)
            acutal_s.append(gt)

        percentage = getattr(self.task, 'almost_correct', 1.0)
        error = almost_correct(acutal_s, predicated_s, self.task.percent, percentage)

        return {'statistic': {'name': self.name,
                              'value': [{'name': self.name, 'value': error}]}}
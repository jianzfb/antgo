# encoding=utf-8
# @Time    : 17-5-15
# @File    : moving_statistic.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import numpy as np


class MovingStatistic(object):
    def __init__(self):
        self.value = None

    def add(self, value):
        pass

    def get(self):
        pass


class MovingAverage(MovingStatistic):
    def __init__(self, ws=1):
        super(MovingAverage,self).__init__()
        self.value = []
        self.window_size = ws

    def add(self, value):
        if len(self.value) < self.window_size:
            self.value.append(value)
        else:
            self.value.pop(0)
            self.value.append(value)

    def get(self):
        if len(self.value) == 0:
            return None
        return np.mean(self.value)
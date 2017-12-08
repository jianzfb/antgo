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
    super(MovingAverage, self).__init__()
    self.value = np.zeros((ws))
    self.window_size = ws
    self.count = 0
    
  def add(self, value):
    index = self.count % self.window_size
    self.value[index] = value
    
    self.count = index + 1

  def get(self):
    if len(self.value) == 0:
      return None
    return np.mean(self.value)
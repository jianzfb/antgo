# encoding=utf-8
# @Time    : 17-5-3
# @File    : base.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *


class AntMeasure(object):
  _BASE_MEASURE=True

  def __init__(self, task,name=None):
    self.task = task
    if name != None:
      self.name = name
    else:
      self.name = AntMeasure.__class__.__name__

    self._is_inverse = False
    self._is_support_rank = False
    self._support_rank_index = 0
    self._crowdsource = False
    self.larger_is_better = True

  @property
  def is_support_rank(self):
    return self._is_support_rank
  @is_support_rank.setter
  def is_support_rank(self,val):
    self._is_support_rank = val

  @property
  def support_rank_index(self):
    return self._support_rank_index
  @support_rank_index.setter
  def support_rank_index(self, index):
    self._support_rank_index = index

  @property
  def crowdsource(self):
    return self._crowdsource
  @crowdsource.setter
  def crowdsource(self, val):
    self._crowdsource = val
# encoding=utf-8
# @Time    : 17-5-3
# @File    : base.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.task.task import *


class AntMeasure(object):
  def __init__(self, task,name=None):
    self.task = task
    if name != None:
      self.name = name
    else:
      self.name = AntMeasure.__class__.__name__

    self._is_support_rank = False
    self._crowdsource = False


  @property
  def is_support_rank(self):
    return self._is_support_rank
  @is_support_rank.setter
  def is_support_rank(self,val):
    self._is_support_rank = val

  @property
  def crowdsource(self):
    return self._crowdsource
  @crowdsource.setter
  def crowdsource(self, val):
    self._crowdsource = val
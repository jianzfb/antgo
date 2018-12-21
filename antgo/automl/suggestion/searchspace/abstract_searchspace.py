# -*- coding: UTF-8 -*-
# @Time    : 2018/12/17 4:53 PM
# @File    : abstract_searchspace.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


class AbstractSearchSpace(object):
  def __init__(self, study, **kwargs):
    self.study = study
    self._params = {}

  def get_new_suggestions(self, number=1, **kwargs):
    raise NotImplementedError

  def fit(self, x_queue, y_queue):
    pass

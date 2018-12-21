# -*- coding: UTF-8 -*-
# @Time    : 2018/12/21 2:20 PM
# @File    : hyperparameters_factory.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.suggestion.algorithm.grid_search import *


class HyperparametersFactory(object):
  @staticmethod
  def get(name):
    if name not in ['GridSearch']:
      return None

    if name == 'GridSearch':
      return GridSearchAlgorithm

  @staticmethod
  def all():
    return ['GridSearch']
# -*- coding: UTF-8 -*-
# @Time    : 2018/12/21 11:13 AM
# @File    : searchspace_factory.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.suggestion.searchspace.evolution import *


class SearchspaceFactory(object):
  @staticmethod
  def get(name):
    if name not in ['Evolution']:
      return None

    if name == 'Evolution':
      return EvolutionSearchSpace

  @staticmethod
  def all():
    return ['Evolution']
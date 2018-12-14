# -*- coding: UTF-8 -*-
# @Time    : 2018/11/29 6:05 PM
# @File    : models.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function


class Study(object):
  contents = []

  def __init__(self,
               name,
               study_configuration,
               algorithm,
               search_space=None,
               status='stop',
               created_time=None,
               updated_time=None,
               flops=0.0):
    self.id = None
    self.name = name
    self.study_configuration = study_configuration
    self.algorithm = algorithm
    self.search_space = search_space
    self.status = status
    self.created_time = created_time
    self.updated_time = updated_time
    self.flops = flops

  @staticmethod
  def get(key, value):
    for s in Study.contents:
      if getattr(s, key) == value:
        return s

    return None

  @staticmethod
  def filter(key, value):
    ss = []
    for s in Trial.contents:
      if getattr(s, key) == value:
        ss.append(s)

    return ss

  @staticmethod
  def create(S):
    S.id = len(Study.contents)
    Study.contents.append(S)

    return S

  @staticmethod
  def delete(S):
    if S in Study.contents:
      delete_index = -1
      for s_i, s in enumerate(Study.contents):
        if s == S:
          delete_index = s_i
          break

      Study.contents.pop(delete_index)
      return S

    return None

  def save(self):
    pass


class Trial(object):
  contents = []

  def __init__(self,
               study_name,
               name,
               parameter_values=None,
               structure=None,
               md5=None,
               objective_value=-1.0,
               status=None,
               created_time=None,
               updated_time=None):
    self.id = id
    self.study_name = study_name
    self.name = name
    self.parameter_values = parameter_values
    self.structure = structure
    self.md5 = md5
    self.objective_value = objective_value
    self.status = status
    self.created_time = created_time
    self.updated_time = updated_time

  @staticmethod
  def get(key, value):
    for s in Trial.contents:
      if getattr(s, key) == value:
        return s

    return None

  @staticmethod
  def create(T):
    T.id = len(Trial.contents)
    Trial.contents.append(T)
    return T

  @staticmethod
  def delete(T):
    if T in Trial.contents:
      Study.contents.pop(T)
      return T

    return None

  @staticmethod
  def filter(key, value):
    ss = []
    for s in Trial.contents:
      if getattr(s, key) == value:
        ss.append(s)

    return ss

  def save(self):
    pass
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

  @staticmethod
  def get(**kwargs):
    for s in Study.contents:
      for key, value in kwargs.items():
        if getattr(s, key) == value:
          return s
        break

    return None

  @staticmethod
  def filter(**kwargs):
    ss = []
    for s in Study.contents:
      is_ok = True
      for key, value in kwargs.items():
        if getattr(s, key) != value:
          is_ok = False
          break
      if is_ok:
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
               structure_encoder=None,
               objective_value=-1.0,
               status=None,
               created_time=None,
               updated_time=None,
               address=None,
               tag=None):
    self.id = id
    self.study_name = study_name
    self.name = name
    self.parameter_values = parameter_values
    self.structure = structure
    self.structure_encoder = structure_encoder
    self.objective_value = objective_value
    self.status = status
    self.created_time = created_time
    self.updated_time = updated_time
    self.address = address
    self.tag = tag

  @staticmethod
  def get(**kwargs):
    for s in Trial.contents:
      for key, value in kwargs.items():
        if getattr(s, key) == value:
          return s
        break

    return None

  @staticmethod
  def create(T):
    T.id = len(Trial.contents)
    Trial.contents.append(T)
    return T

  @staticmethod
  def delete(T):
    if T in Trial.contents:
      delete_index = -1
      for t_i, t in enumerate(Trial.contents):
        if t == T:
          delete_index = t_i
          break

      Trial.contents.pop(delete_index)
      return T

    return None

  @staticmethod
  def filter(**kwargs):
    ss = []
    for s in Trial.contents:
      is_ok = True
      for key, value in kwargs.items():
        if getattr(s, key) != value:
          is_ok = False
          break
      if is_ok:
        ss.append(s)

    return ss

  def save(self):
    pass
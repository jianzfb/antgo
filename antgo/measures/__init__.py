# -*- coding: UTF-8 -*-
# @Time    : 18-3-26
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from pkgutil import walk_packages
import os


class AntMeasuresFactory(object):
  factory_measures = {}

  def __init__(self, task):
    self.task = task
    self.support_measures = []
    if self.task.task_type is not None and self.task.task_type != '':
      if self.task.task_type in AntMeasuresFactory.factory_measures:
        self.support_measures = AntMeasuresFactory.factory_measures[self.task.task_type]

    if 'CUSTOM' in AntMeasuresFactory.factory_measures:
      self.support_measures.extend(AntMeasuresFactory.factory_measures['CUSTOM'])

  def measures(self, measure_names=None):
    if measure_names is not None:
      if type(measure_names) != list:
        measure_names = [measure_names]

      applied_measures = []
      for measure_obj, measure_name in self.support_measures:
        if measure_name is not None:
          if measure_name in measure_names:
            applied_measures.append(measure_obj(self.task))
        else:
          applied_measures.append(measure_obj(self.task))

      return applied_measures
    else:
      return [measure_obj(self.task) for measure_obj, _ in self.support_measures]

  @staticmethod
  def add_custom_measure(custom_measure):
    for task_type, task_related_measures in AntMeasuresFactory.factory_measures.items():
      for task_measure, task_measure_name in task_related_measures:
        if custom_measure == task_measure:
          return

    if 'CUSTOM' not in AntMeasuresFactory.factory_measures:
      AntMeasuresFactory.factory_measures['CUSTOM'] = []

    AntMeasuresFactory.factory_measures['CUSTOM'].append((custom_measure, None))


def _global_import(name):
  p = __import__(name, globals(), locals(), level=1)
  lst = p.default if 'default' in dir(p) else {}
  if len(lst) > 0:
    globals().pop(name)

  for measure_method, keys in lst.items():
    globals()[measure_method] = p.__dict__[measure_method]
    measure_name, task_type = keys
    if task_type not in AntMeasuresFactory.factory_measures:
      AntMeasuresFactory.factory_measures[task_type] = []
    AntMeasuresFactory.factory_measures[task_type].append((p.__dict__[measure_method], measure_name))


for _, module_name, _ in walk_packages([os.path.dirname(__file__)]):
  if not module_name.startswith('_'):
    _global_import(module_name)

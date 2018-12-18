# -*- coding: UTF-8 -*-
# @Time    : 2018/12/17 11:04 PM
# @File    : metric.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from abc import abstractmethod
from sklearn.metrics import accuracy_score, mean_squared_error


class Metric:

  @classmethod
  @abstractmethod
  def higher_better(cls):
    pass

  @classmethod
  @abstractmethod
  def compute(cls, prediction, target):
    pass

  @classmethod
  @abstractmethod
  def evaluate(cls, prediction, target):
    pass


class Accuracy(Metric):
  @classmethod
  def higher_better(cls):
    return True

  @classmethod
  def compute(cls, prediction, target):
    prediction = list(map(lambda x: x.argmax(), prediction))
    target = list(map(lambda x: x.argmax(), target))
    return cls.evaluate(prediction, target)

  @classmethod
  def evaluate(cls, prediction, target):
    return accuracy_score(prediction, target)

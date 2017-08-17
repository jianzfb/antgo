# -*- coding: UTF-8 -*-
# @Time    : 17-6-22
# @File    : trainer.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
import re
from abc import ABCMeta, abstractmethod
from antgo.context import *

trainer_default_context = {'progress_step': 10,
                           'batch_size': 256,
                           'max_epochs': 100,
                           'decay_rate': 1,
                           'decay_steps': 1,
                           'staircase': False,
                           'lr': 0.0001,
                           'iter': 0,
                           'pre_trained_model': None,
                           'optimization': None,
                           'snapshot_prefix': 'alpha',
                           'snapshot_infix': 'train'}


class Trainer(object):
    def __init__(self, trainer_context=None, is_training=True):
        # 1.step config trainer context
        for k, v in trainer_default_context.items():
            if trainer_context is not None:
                setattr(self, k, getattr(trainer_context, k, v))
            else:
                setattr(self, k, v)

        # 3.step other
        self.iter = 0
        self.is_training = is_training

        # context
        self.ctx = get_global_context()
        self.ctx.registry_clear_callback(self.clear)

    def deploy(self, model):
        pass

    def run(self, data_generator, binds):
        if self.ctx is not None:
            for k, v, c, f in self.ctx.registried_trainer_callbacks:
                cur_value = getattr(self, k, None)
                if cur_value is not None:
                    if c == 'equal':
                        if cur_value == v:
                            f()
                    elif c == 'less':
                        if cur_value < v:
                            f()
                    elif c == 'greater':
                        if cur_value > v:
                            f()
                    elif c == 'mod':
                        if int(cur_value) % int(v) == 0:
                            f()

    def snapshot(self, dump_dir, epoch):
        pass

    def watch(self, name, fuzzy=True):
        # add watch var list
        pass

    @property
    def iter_at(self):
        return self.iter
    @iter_at.setter
    def iter_at(self, val):
        self.iter = val

    def clear(self):
        pass


class ModelDesc(object):
  __metaclass__ = ABCMeta

  def __init__(self, model_context=None, model_name=None):
    if model_name is not None:
      self.model_name = model_name
    else:
      self.model_name = self.__class__.__name__

    if model_context is not None:
      for k in dir(model_context):
        if not k.startswith('__'):
          setattr(self, k, getattr(model_context, k, None))

  @property
  def name(self):
    return self.model_name

  @abstractmethod
  def build(self, is_training=True):
    '''
    :return: 
    '''

  def arg_scope(self):
    return None

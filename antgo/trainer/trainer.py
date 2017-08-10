# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: jian(jian@mltalker.com)
from __future__ import division
from __future__ import unicode_literals

import tensorflow as tf
import re
from abc import ABCMeta, abstractmethod
from antgo.context import *

_CurrentTowerContext = None


class TowerContext(object):
    def __init__(self, tower_name, is_training=None):
        """ tower_name: 'tower0', 'towerp0', or '' """
        self._name = tower_name
        if is_training is None:
            is_training = not self._name.startswith('towerp')
        self._is_training = is_training

    @property
    def is_main_training_tower(self):
        return self.is_training and (self._name == '' or self._name == 'tower0')

    @property
    def is_main_tower(self):
        return self._name == '' or self._name == 'tower0'

    @property
    def is_training(self):
        return self._is_training

    @property
    def name(self):
        return self._name

    @staticmethod
    def _get_tensors(graph):
      ts = []
      for op in graph.get_operations():
        ts += op.outputs
      return ts

    @staticmethod
    def find_tensor(graph,names=[],fuzzy=False):
        required_names_and_vars = {}

        check_tensors = TowerContext._get_tensors(graph)
        with graph.as_default():
            check_tensors.extend(tf.global_variables())

        for tensor in check_tensors:
            if not fuzzy:
                try:
                    terms = tensor.split('/')
                    fuzzy_name = '/'.join(terms[2:])
                    if fuzzy_name in names:
                        if fuzzy_name not in required_names_and_vars:
                            required_names_and_vars[fuzzy_name] = [tensor]
                        else:
                            required_names_and_vars[fuzzy_name].append(tensor)
                except:
                    pass
            else:
                try:
                    res = re.findall('/[^/]+:', tensor.name)
                    fuzzy_name = res[-1][1:-1]
                    if fuzzy_name in names:
                        if fuzzy_name not in required_names_and_vars:
                            required_names_and_vars[fuzzy_name] = [tensor]
                        else:
                            required_names_and_vars[fuzzy_name].append(tensor)
                except:
                    pass

        required_tensor = []
        for required_name in names:
            assert(required_name in required_names_and_vars)
            required_vars = required_names_and_vars[required_name]
            if len(required_vars) == 1:
                required_tensor.append(required_vars)
            else:
                expanded_list = []
                for v in required_vars:
                    expanded_v = tf.expand_dims(v, 0)
                    expanded_list.append(expanded_v)

                required_tensor.append(tf.concat(expanded_list, 0))

        return required_tensor

    def find_tensor_in_tower(self, graph, name,index=0):
        if self.is_main_tower:
            return graph.get_tensor_by_name('tower0/model/{}:{}'.format(name,index))

        return graph.get_tensor_by_name('{}/model/{}:{}'.format(self.name, name, index))

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, "Nesting TowerContext!"
        _CurrentTowerContext = self

        if len(self._name):
            self._scope = tf.name_scope(self._name)
            return self._scope.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None
        if len(self._name):
            self._scope.__exit__(exc_type, exc_val, exc_tb)
        return False


def get_current_tower_context():
    global _CurrentTowerContext
    return _CurrentTowerContext

trainer_default_context={'min_grad':-10000.0,
                         'max_grad':10000.0,
                         'progress_step':10,
                         'gpu':[],
                         'cpu':[0],
                         'batch_size':256,
                         'max_epochs':100,
                         'decay_rate':1,
                         'decay_steps':1,
                         'staircase':False,
                         'lr':0.0001,
                         'iter':0,
                         'pre_trained_model':None,
                         'optimization':None,
                         'snapshot_prefix':'alpha',
                         'snapshot_infix':'train'}


class Trainer(object):
    def __init__(self,trainer_context=None,is_training=True):
        # 1.step config trainer context
        for k,v in trainer_default_context.items():
            if trainer_context is not None:
                setattr(self,k,getattr(trainer_context,k,v))
            else:
                setattr(self,k,v)

        # 2.step config device
        if len(self.gpu) > 0:
            self.device_list = self.gpu
            self.device_prefix = 'gpu'
        elif len(self.cpu) > 0:
            self.device_list = self.cpu
            self.device_prefix = 'cpu'
        else:
            self.device_list = [0]
            self.device_prefix = 'cpu'

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

    def snapshot(self,dump_dir,epoch):
        pass

    def watch(self,name,fuzzy=True):
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
    sub_models = None

    class SubModel(object):
        def __init__(self, sub_model_name):
            self.sub_model_name = sub_model_name
            self.saveble_vars = []
            self.trainable_vars = []

        def __enter__(self):
            if self.sub_model_name not in ModelDesc.sub_models:
                # record saveble_vars and trainable_vars
                # existed saveble_vars
                self.saveble_vars = tf.model_variables()

                # existed trainable_vars
                self.trainable_vars = tf.trainable_variables()

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.sub_model_name not in ModelDesc.sub_models:
                # record saveble_vars and trainable_vars
                increment_trainable_vars = tf.trainable_variables()
                increment_saveble_vars = tf.model_variables()

                trainable_vars = list(set(increment_trainable_vars) - set(self.trainable_vars))
                saveble_vars = list(set(increment_saveble_vars) - set(self.saveble_vars))

                ModelDesc.sub_models[self.sub_model_name] = [trainable_vars, saveble_vars]

    def __init__(self, model_context=None, model_name=None):
        if model_name is not None:
            self.model_name = model_name
        else:
            self.model_name = self.__class__.__name__

        if model_context is not None:
            for k in dir(model_context):
                if not k.startswith('__'):
                    setattr(self, k, getattr(model_context, k, None))

        # sub model
        ModelDesc.sub_models = {}

    @property
    def name(self):
        return self.model_name

    @property
    def saveble_vars(self):
        tower_saveable_vars = tf.global_variables()
        return tower_saveable_vars

    @property
    def trainable_vars(self):
        tower_trainable_vars = tf.trainable_variables()
        return tower_trainable_vars

    @abstractmethod
    def build(self):
        '''
        :return: 
        '''

    def graph_vars(self, graph_name):
        return ModelDesc.sub_models[graph_name][1] if graph_name in ModelDesc.sub_models else None

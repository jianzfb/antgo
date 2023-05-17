# -*- coding: UTF-8 -*-
# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 14:44
# @File    : factory.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import threading
from typing import Any, Dict, List, Tuple

from .execution.base_execution import *
from antgo.pipeline.hparam.hyperparameter import *
from antgo.pipeline.engine.operator_registry import *
from antgo.pipeline.engine.operator_loader import *


def op(operator_src: str,
       tag: str = 'main',
       arg: List[Any] = [],
       kwargs: Dict[str, Any] = {}):
  """
  Entry method which takes either operators tasks or paths to python files or class in notebook.
  An `Operator` object is created with the init args(kwargs).
  Args:
      operator_src (`str`):
          operators name or python file location or class in notebook.
      tag (`str`):
          Which tag to use for operators on hub, defaults to `main`.
  Returns
      (`typing.Any`)
          The `Operator` output.
  """
  if isinstance(operator_src, type):
    class_op = type('operators', (operator_src,), kwargs)
    return class_op.__new__(class_op, **kwargs)

  loader = OperatorLoader()
  return loader.load_operator(operator_src, arg, kwargs, tag)


class _OperatorLazyWrapper(  #
  BaseExecution):
  """
  operators wrapper for lazy initialization.
  """

  def __init__(self,
               real_name: str,
               index: Tuple[str],
               tag: str = 'main',
               arg: List[Any] = [],
               kws: Dict[str, Any] = {}) -> None:
    self._name = real_name.replace('.', '/').replace('_', '-')
    self._index = index
    self._tag = tag
    self._arg = arg
    self._kws = kws
    self._op = None
    self._lock = threading.Lock()
    self._op_config = self._kws.pop('op_config', None)
    # TODO: (How to apply such config)

  def __check_init__(self):
    with self._lock:
      if self._op is None:
        with param_scope(index=self._index):
          self._op = op(self._name,
                        self._tag,
                        arg=self._arg,
                        kwargs=self._kws)
          if hasattr(self._op, '__vcall__'):
            self.__has_vcall__ = True

  def get_op(self):
    self.__check_init__()
    return self._op

  @property
  def op_config(self):
    self.__check_init__()
    return self._op_config

  @property
  def function(self):
    return self._name

  @property
  def init_args(self):
    return self._kws

  @staticmethod
  def callback(real_name: str, index: Tuple[str], *arg, **kws):
    return _OperatorLazyWrapper(real_name, index, arg=arg, kws=kws)


@dynamic_dispatch
def ops(*arg, **kws):
    """
    Entry point for creating operators instances, for example:

    >>> op_instance = ops.my_namespace.my_repo_name(init_arg1=xxx, init_arg2=xxx)
    """

    # pylint: disable=protected-access
    with param_scope() as hp:
        real_name = hp._name
        index = hp._index
    return _OperatorLazyWrapper.callback(real_name, index, *arg, **kws)

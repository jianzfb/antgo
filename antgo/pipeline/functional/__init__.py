# -*- coding: UTF-8 -*-
# @Time    : 2022/9/11 23:01
# @File    : __init__.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from .data_collection import DataCollection, DataFrame
from .entity import Entity
from .image import *
from .common import *
from antgo.pipeline.hparam import HyperParameter as State

from antgo.pipeline.hparam import param_scope
from antgo.pipeline.hparam import dynamic_dispatch
read_camera = DataCollection.read_camera
read_video = DataCollection.read_video


@dynamic_dispatch
def glob(*arg):  # pragma: no cover
    """
    Return a DataCollection of paths matching a pathname pattern.
    """

    index = param_scope()._index
    if index is None:
        return DataCollection.from_glob(*arg)
    return DataFrame.from_glob(*arg).map(lambda x: Entity(**{index: x}))


@dynamic_dispatch
def video_dc(*args):
    index = param_scope()._index
    if index is None:
        return DataCollection.read_video(*args)
      
    return DataFrame.read_video(*args).map(lambda x: Entity(**{index: x}))


@dynamic_dispatch
def camera_dc(*args):
    index = param_scope()._index
    if index is None:
        return DataCollection.read_camera(*args)
      
    return DataFrame.read_camera(*args).map(lambda x: Entity(**{index: x}))


def _api():
  """
  Create an API input, for building RestFul API or application API.
  """
  return DataFrame.api(index=param_scope()._index)


api = dynamic_dispatch(_api)


def _web():
  return DataFrame.web(index=param_scope()._index)


web = dynamic_dispatch(_web)


def _dummy_input():
  """
  Create a dummy input.
  """
  return _api().__enter__()


dummy_input = dynamic_dispatch(_dummy_input)


def _dc(iterable):
  """
  Return a DataCollection.
  """

  index = param_scope()._index
  if index is None:
    return DataCollection(iterable)
  if isinstance(index, (list, tuple)):
    return DataFrame(iterable).map(lambda x: Entity(**dict(zip(index, x))))
  return DataFrame(iterable).map(lambda x: Entity(**{index: x}))


dc = dynamic_dispatch(_dc)

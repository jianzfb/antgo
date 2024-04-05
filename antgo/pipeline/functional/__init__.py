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
from antgo.pipeline.functional.common.config import *
import numpy as np
import json
import os
import cv2

read_camera = DataCollection.read_camera
read_video = DataCollection.read_video


@dynamic_dispatch
def glob(*args, repeat=1):  # pragma: no cover
    """
    Return a DataCollection of paths matching a pathname pattern.
    """

    global_entity = Entity()
    index = param_scope()._index
    def inner():
        from glob import glob
        for _ in range(repeat):
          files = []
          for path in args:
              files.extend(glob(path))
          if len(files) == 0:
              raise FileNotFoundError(f'There is no files with {args}.')

          # sort
          files.sort()
          for ff in files:
            yield global_entity(**{index: ff})

    return DataFrame(inner())


@dynamic_dispatch
def json_dc(*args):
  index = param_scope()._index

  global_entity = Entity()
  def inner():
    for json_path in args:
      with open(json_path, 'r') as fp:
        info_list = json.load(fp)
        for data in info_list:
          yield global_entity(**{index: data})

  return DataFrame(inner())


@dynamic_dispatch
def txt_dc(*args):
  index = param_scope()._index

  global_entity = Entity()
  def inner():
    for json_path in args:
      with open(json_path, 'r') as fp:
        string = fp.readline()
        while string:
            data = json.loads(string)
            string = fp.readline()
            yield global_entity(**{index: data})

  return DataFrame(inner())


@dynamic_dispatch
def video_dc(*args):
    index = param_scope()._index
    if not isinstance(index, tuple):
      print('Video dc neef (frame, frame_index) export')
      return

    if len(args) == 1 and isinstance(args[0], str):
      video_entity = Entity()
      return DataFrame.read_video(*args).map(lambda x: video_entity(**{key: value for key,value in zip(index, x)}))

    def inner():
      source_iterator_list = []
      for video_path in args:
        source_iterator_list.append(
          iter(DataFrame.read_video(video_path))
        )

      source_num = len(source_iterator_list)
      frame_index = 0
      global_entity = Entity()
      while True:
        source_data = []
        none_num = 0
        for source_i in range(source_num):
          # for a in source_iterator_list[source_i]:
          #   print(a)
          try:
            value = next(source_iterator_list[source_i])
          except:
            value = None
            none_num += 1

          source_data.append(value[0])
        if none_num == source_num:
          break

        source_data += [frame_index]
        data_dict = {}
        for ii,vv in zip(index, source_data):
          data_dict[ii] = vv
        yield global_entity(**data_dict)
        frame_index += 1

    return DataFrame(inner())


@dynamic_dispatch
def camera_dc(*args):
    index = param_scope()._index
    if not isinstance(index, tuple):
      print('Camera dc neef (frame, frame_index) export')
      return

    camera_entity = Entity()
    return DataFrame.read_video(*args).map(lambda x: camera_entity(**{key: value for key,value in zip(index, x)}))


@dynamic_dispatch
def imread_dc(*args):
  index = param_scope()._index
  assert(args[0], str)
  assert(os.path.exists(args[0]))
  image = cv2.imread(args[0])

  def inner():
    yield Entity()(**{index: image})

  return DataFrame(inner())


@dynamic_dispatch
def serial_imread_dc(*args):
  index = param_scope()._index
  assert(args[0], str)

  def inner():
    for image_path in args:
      image = cv2.imread(args[0])
      yield Entity()(**{index: image})

  return DataFrame(inner())


def _api(name='serve'):
  """
  Create an API input, for building RestFul API or application API.
  """
  return DataFrame.api(index=param_scope()._index, name=name)


api = dynamic_dispatch(_api)


def _web(name='demo', **kwargs):
  return DataFrame.web(index=param_scope()._index, name=name, **kwargs)


web = dynamic_dispatch(_web)


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


def _placeholder(*arg):
  index = param_scope()._index

  # 
  if isinstance(index, tuple):
    for ii,xx in zip(index, arg):
      
      data_type = -1
      data_shape = []
      if isinstance(xx, np.ndarray):
        data_shape = list(xx.shape)
        if xx.dtype == np.float32:
          data_type = 6
        elif xx.dtype == np.float64:
          data_type = 7
        elif xx.dtype == np.int32:
          data_type = 4
        elif xx.dtype == np.uint8:
          data_type = 1
      elif isinstance(xx, str):
        data_shape = []
        data_type = 11
      elif isinstance(xx, bool):
        data_shape = []
        data_type = 10

      if data_type < 0:
        print('placeholder type abnormal.')
      
      add_op_info(
        'placeholder_op', 
        (None, (ii,)), 
        (), 
        {
          'memory_type': 2,     # CPU_BUFFER
          'data_format': 1000,  # AUTO
          'data_type': data_type,        # EAGLEEYE_UCHAR, EAGLEEYE_FLOAT
          'shape': data_shape,  # 
        }
      )

    temp = list((x,y) for x,y in zip(index, arg))
    return DataFrame.placeholder(temp).map(
      lambda mm: Entity(**{ii: xx for ii, xx in mm })
    )
  else:
    for ii, xx in zip([index], arg):
      data_shape = []
      data_type = -1
      if isinstance(xx, np.ndarray):
        data_shape = list(xx.shape)
        if xx.dtype == np.float32:
          data_type = 6
        elif xx.dtype == np.int32:
          data_type = 4
        elif xx.dtype == np.uint8:
          data_type = 1
      elif isinstance(xx, str):
        data_shape = []
        data_type = 11
      elif isinstance(xx, bool):
        data_shape = []
        data_type = 10
  
      if data_type < 0:
        print('placeholder type abnormal.')
      
      add_op_info(
        'placeholder_op', 
        (None, (ii,)), 
        (), 
        {
          'memory_type': 2,               # CPU_BUFFER
          'data_format': 1000,            # AUTO
          'data_type': data_type,         # EAGLEEYE_UCHAR, EAGLEEYE_FLOAT
          'shape': data_shape,        # 
        }
      )

    return DataFrame.placeholder(*arg).map(lambda x: Entity(**{index: x }))

placeholder = dynamic_dispatch(_placeholder)

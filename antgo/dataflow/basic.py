# encoding=utf-8
# @Time    : 17-7-31
# @File    : basic.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.utils.serialize import loads, dumps
import numpy as np
import os
import traceback
import sys
import yaml
from antgo import config
import antgo.utils.pickledb as pickledb
from contextlib import contextmanager
from antgo.utils import logger


class Sample(object):
  def __init__(self, **kwargs):
    self.data = kwargs

  def serialize(self):
    return self.data

  @staticmethod
  def unserialize(bytes_data):
    return loads(bytes_data)


class RecordWriter(object):
  def __init__(self, record_path):
    self._record_path = os.path.join(record_path, 'data.db')
    
    try:
      self._db = pickledb.load(self._record_path, False)
      count = self._db.get(str('attrib-count'))
      if count is None:
        self._db.set(str('attrib-count'), str(0))
    except Exception as e:
      print(e)
      logger.error('Couldnt open db')
    
  def close(self):
    self._db.dump()

  def write(self, sample, sample_index=-1):
    count = self._db.get(str('attrib-count'))
    assert(count is not None)
    count = int(count)

    sample_key = str(count)
    if sample_index > 0:
      sample_key = str(sample_index)
    self._db.set(sample_key, sample.serialize())

    count += 1
    self._db.set('attrib-count', str(count))
    
  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    for k,v in kwargs.items():
      self._db.set(str('attrib-%s'%k), str('attrib-%s'%v))
  
  @property
  def size(self):
    count = self._db.get(str('attrib-count'))
    if count is None:
      return 0
    
    count = int(count)
    return count

@contextmanager
def safe_recorder_manager(recorder):
  try:
    yield recorder
    recorder.close()
  except:
    traceback.print_exc()
    raise sys.exc_info()[0]

@contextmanager
def safe_manager(handler):
  try:
    yield handler
  except:
    traceback.print_exc()
    raise sys.exc_info()[0]


class RecordReader(object):
  def __init__(self, record_path, read_only=True):
    # db
    if os.path.exists(os.path.join(record_path, 'record')):
      record_path = os.path.join(record_path, 'record')

    self._record_path = os.path.join(record_path, 'data.db')

    try:
      self._db = pickledb.load(self._record_path, False)
      # db attributes
      self._db_attrs = {}
      self.count = self._db.get('attrib-count')

      # attrib
      # it = self._db.iteritems()
      # it.seek('attrib-')
      # for attrib_item in it:
      #   key, value = attrib_item
      #   if key.startswith('attrib-'):
      #     key = key.replace('attrib-', '')
      #     value = value.replace('attrib-', '')
      #     self._db_attrs[key] = value
      #     setattr(self, key, value)
    except:
      logger.error('couldnt open rocksdb')

  def close(self):
    pass

  def record_attrs(self):
    return self._db_attrs

  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    for k, v in kwargs.items():
      self._db.set(str('attrib-%s' % k), str('attrib-%s' % v))

  def put(self, key, data):
    self._db.set(str(key), str(data))

  def get(self, key):
    return self._db.get(str(key))

  def write(self, sample, sample_index=-1):
    count = self._db.get(str('attrib-count'))
    if count is None:
      self._db.set(str('attrib-count'), '0')
      count = 0

    count = int(count)
    sample_key = str(count)
    if sample_index >= 0 :
      sample_key = str(sample_index)
    self._db.set(sample_key, sample.serialize())

    count += 1
    self._db.set('attrib-count', str(count))

  def read(self, index, *args):
    try:
      data = self._db.get(str(index))
      # data = Sample.unserialize(ss)
      sample = []
      if len(args) == 0:
        for k, v in data.items():
          sample.append(v)
      else:
        for k in args:
          if k in data:
            sample.append(data[k])
          else:
            sample.append(None)

      return sample
    except:
      return [None for _ in range(len(args))]

  def iterate_read(self, *args):
    for k in range(int(self.count)):
      data = self._db.get(str(k))
      # data = Sample.unserialize(ss)
      sample = []
      if len(args) == 0:
        for data_key, data_val in data.items():
          sample.append(data_val)
        yield sample
      else:
        for data_key in args:
          if data_key in data:
            sample.append(data[data_key])
          else:
            sample.append(None)
        yield sample

  def iterate_sampling_read(self, index, *args):
    for i in index:
      data = self._db.get(str(i))
      # data = Sample.unserialize(ss)
      sample = []
      if len(args) == 0:
        for data_key, data_val in data.items():
          sample.append(data_val)
        yield sample
      else:
        for data_key in args:
          if data_key in data:
            sample.append(data[data_key])
          else:
            sample.append(None)
        yield sample

  @property
  def size(self):
    count = self._db.get(str('attrib-count'))
    if count is None:
      return 0

    count = int(count)
    return count


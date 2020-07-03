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
try:
  import rocksdb
except:
  rocksdb = None
from antgo import config
from contextlib import contextmanager
from antgo.utils import logger
import multiprocessing

class Sample(object):
  def __init__(self, **kwargs):
    self.data = kwargs

  def serialize(self):
    return dumps(self.data)

  @staticmethod
  def unserialize(bytes_data):
    return loads(bytes_data)


class RecordWriter(object):
  def __init__(self, record_path):
    self._record_path = record_path
    
    try:
      opts = rocksdb.Options()
      opts.create_if_missing = True
      self._db = rocksdb.DB(record_path, opts)
    
      count = self._db.get(str('attrib-count').encode('utf-8'))
      if count is None:
        self._db.put(str('attrib-count').encode('utf-8'), b'0')
    except:
      logger.error('couldnt open rocksdb')
    
  def close(self):
    pass

  def write(self, sample, sample_index=-1):
    count = self._db.get(str('attrib-count').encode('utf-8'))
    assert(count is not None)
    count = int(count)

    sample_key = str(count).encode('utf-8')
    if sample_index > 0:
      sample_key = str(sample_index).encode('utf-8')
    self._db.put(sample_key, sample.serialize())

    count += 1
    self._db.put('attrib-count'.encode('utf-8'), str(count).encode('utf-8'))
    
  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    for k,v in kwargs.items():
      self._db.put(str('attrib-%s'%k).encode('utf-8'), str('attrib-%s'%v).encode('utf-8'))
  
  @property
  def size(self):
    count = self._db.get(str('attrib-count').encode('utf-8'))
    if count is None:
      return 0
    
    count = int(count)
    return count

@contextmanager
def safe_recorder_manager(recorder):
  try:
    yield recorder
  except:
    traceback.print_exc()
    raise sys.exc_info()[0]

class RecordReader(object):
  def __init__(self, record_path, read_only=True):
    # db
    if os.path.exists(os.path.join(record_path, 'record')):
      record_path = os.path.join(record_path, 'record')
    
    try:
      opts = rocksdb.Options(create_if_missing=False if read_only else True)
      self._db = rocksdb.DB(record_path, opts, read_only=read_only)
  
      # db path
      self._record_path = record_path
  
      # db attributes
      self._db_attrs = {}
      
      # attrib
      it = self._db.iteritems()
      it.seek('attrib-'.encode('utf-8'))
      for attrib_item in it:
        key, value = attrib_item
        key = key.decode('utf-8')
        value = value.decode('utf-8')
        if key.startswith('attrib-'):
          key = key.replace('attrib-', '')
          value = value.replace('attrib-', '')
          self._db_attrs[key] = value
          setattr(self, key, value)
    except:
      logger.error('couldnt open rocksdb')

  def close(self):
    pass

  def record_attrs(self):
    return self._db_attrs

  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    for k, v in kwargs.items():
      self._db.put(str('attrib-%s' % k).encode('utf-8'), str('attrib-%s' % v).encode('utf-8'))

  def put(self, key, data):
    self._db.put(str(key).encode('utf-8'), str(data).encode('utf-8'))

  def get(self, key):
    return self._db.get(str(key).encode('utf-8'))

  def write(self, sample, sample_index=-1):
    count = self._db.get(str('attrib-count').encode('utf-8'))
    if count is None:
      self._db.put(str('attrib-count').encode('utf-8'), b'0')
      count = 0

    count = int(count)
    sample_key = str(count).encode('utf-8')
    if sample_index >=0 :
      sample_key = str(sample_index).encode('utf-8')
    self._db.put(sample_key, sample.serialize())

    count += 1
    self._db.put('attrib-count'.encode('utf-8'), str(count).encode('utf-8'))

  def read(self, index, *args):
    try:
      ss = self._db.get(str(index).encode('utf-8'))
      data = Sample.unserialize(ss)
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
      ss = self._db.get(str(k).encode('utf-8'))
      data = Sample.unserialize(ss)
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
      ss = self._db.get(str(i).encode('utf-8'))
      data = Sample.unserialize(ss)
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
    count = self._db.get(str('attrib-count').encode('utf-8'))
    if count is None:
      return 0

    count = int(count)
    return count


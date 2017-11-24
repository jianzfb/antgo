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
import yaml
import rocksdb
from antgo import config
from contextlib import contextmanager
from antgo.utils import logger


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
    opts = rocksdb.Options()
    opts.create_if_missing = True
    self._db = rocksdb.DB(record_path, opts)
    
    count = self._db.get(str('attrib-count').encode('utf-8'))
    if count is None:
      self._db.put(str('attrib-count').encode('utf-8'), b'0')
    
  def close(self):
    pass

  def write(self, sample):
    count = self._db.get(str('attrib-count').encode('utf-8'))
    assert(count is not None)
    count = int(count)
    self._db.put(str(count).encode('utf-8'), sample.serialize())

    count += 1
    self._db.put('attrib-count'.encode('utf-8'), str(count).encode('utf-8'))
    

  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    for k,v in kwargs.items():
      self._db.put(str('attrib-%s'%k).encode('utf-8'), str('attrib-%s'%v).encode('utf-8'))


@contextmanager
def safe_recorder_manager(recorder):
  try:
    yield recorder
  except:
    error_info = sys.exc_info()
    logger.error(error_info)
    raise error_info[0]


class RecordReader(object):
  def __init__(self, record_path, daemon=False):
    # db
    opts = rocksdb.Options(create_if_missing=False)

    self._db = rocksdb.DB(record_path, opts, read_only=True)
    # daemon
    self._daemon = daemon

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


  def close(self):
    pass

  def record_attrs(self):
    return self._db_attrs

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

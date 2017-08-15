# encoding=utf-8
# @Time    : 17-7-31
# @File    : basic.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.utils.serialize import loads, dumps
import numpy as np
import os
import yaml
import plyvel

global_dbs = {}


def _global_db(db_path):
  global global_dbs
  if db_path in global_dbs:
    return global_dbs[db_path]
  else:
    global_dbs[db_path] = plyvel.DB(db_path)

  return global_dbs[db_path]


class Sample(object):
  def __init__(self, **kwargs):
    self.data = kwargs

  def serialize(self):
    return dumps(self.data)

  @staticmethod
  def unserialize(bytes_data):
    return loads(bytes_data)


class RecordWriter(object):
  def __init__(self, record_path, block_size=None):
    self._record_path = record_path
    self._db = plyvel.DB(record_path, create_if_missing=True, block_size=block_size)
    self._count = 0

  def close(self):
    # 1.step db attributes
    if self._count > 0:
      db_attrs = {}
      if os.path.exists(os.path.join(self._record_path, 'attrs.yaml')):
        attrs = yaml.load(open(os.path.join(self._record_path, 'attrs.yaml'), 'r'))
        db_attrs.update(attrs)

      db_attrs['count'] = self._count

      with open(os.path.join(self._record_path, 'attrs.yaml'), 'w') as outfile:
        yaml.dump(db_attrs, outfile, default_flow_style=False)

    # 2.step close db
    self._db.close()

  def write(self, sample):
    self._db.put(bytes(self._count), sample.serialize())
    self._count += 1

  def bind_attrs(self, **kwargs):
    # bind extra db attributes
    db_attrs = kwargs
    # load existed attrs
    if os.path.exists(os.path.join(self._record_path, 'attrs.yaml')):
      attrs = yaml.load(open(os.path.join(self._record_path, 'attrs.yaml'), 'r'))
      db_attrs.update(attrs)

    with open(os.path.join(self._record_path, 'attrs.yaml'), 'w') as outfile:
      yaml.dump(db_attrs, outfile, default_flow_style=False)


class RecordReader(object):
  def __init__(self, record_path):
    # db
    self._record_path = record_path
    self._db = _global_db(record_path)

    # db attributes
    self._db_attrs = {}
    if os.path.exists(os.path.join(record_path, 'attrs.yaml')):
      attrs = yaml.load(open(os.path.join(record_path, 'attrs.yaml'), 'r'))
      self._db_attrs.update(attrs)

    for attr_key, attr_value in self._db_attrs.items():
      setattr(self, attr_key, attr_value)

  def close(self):
    global global_dbs
    if self._record_path in global_dbs:
      global_dbs.pop(self._record_path)
    self._db.close()

  def record_attrs(self):
    return self._db_attrs

  def read(self, index, *args):
    try:
      data = Sample.unserialize(self._db.get(bytes(index)))
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
    for _, v in self._db:
      data = Sample.unserialize(v)
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
      data = Sample.unserialize(self._db.get(bytes(i)))
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

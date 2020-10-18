# encoding=utf-8
# @Time     : 17-8-14
# @File     : serialize.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import importlib
import msgpack
import msgpack_numpy
msgpack_numpy.patch()

import json
__all__ = ['loads', 'dumps', 'Encoder', 'Decoder']


def dumps(obj):
  return msgpack.dumps(obj)


def loads(buf):
  return msgpack.loads(buf)


class Decoder(json.JSONDecoder):
  def __init__(self):
    json.JSONDecoder.__init__(self, object_hook=self.dict_to_object)

  def dict_to_object(self, d):
    if '__class__' in d:
      class_name = d.pop('__class__')
      module_name = d.pop('__module__')
      # module = __import__(module_name.split('.')[1])
      # module = __import__(module_name)
      module = importlib.import_module(module_name)
      class_ = getattr(module, class_name)
      args = dict((key, value) for key, value in d.items())

      inst = class_(**args)
    else:
      inst = d
    return inst


class Encoder(json.JSONEncoder):
  def default(self, obj):
    # Convert objects to a dictionary of their representation
    d = {'__class__': obj.__class__.__name__,
         '__module__': obj.__module__,
         }
    d.update(obj.__dict__)
    return d

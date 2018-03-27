#  -*- coding: UTF-8 -*-
# @Time    : 18-3-26
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from pkgutil import walk_packages
import os
import os.path
from .simplecsvs import *
from .simpleimages import *
from .standard import *
is_support_tf = True
try:
  from .tfrecordsreader import *
  is_support_tf = True
except:
  is_support_tf = False


class AntDatasetFactory(object):
  factory_dataset = {}

  @staticmethod
  def dataset(name, parse_flag=''):
    if 'CUSTOM' in AntDatasetFactory.factory_dataset:
      return AntDatasetFactory.factory_dataset['CUSTOM']

    for dataset_name, dataset_obj in AntDatasetFactory.factory_dataset.items():
      if dataset_name.lower() == name.lower():
        return dataset_obj

    if parse_flag == 'csv':
      return CSV
    elif (parse_flag == 'tfrecord' or parse_flag == 'tfrecords') and is_support_tf:
      return TFRecordsReader

    if name.startswith('tf') and is_support_tf:
      return TFRecordsReader
    elif name.startswith('image'):
      return SimpleImages

    return Standard

  @staticmethod
  def add_custom_dataset(custom_dataset):
    for dataset_name, dataset_obj in AntDatasetFactory.factory_dataset.items():
      if dataset_obj == custom_dataset:
        return

    AntDatasetFactory.factory_dataset['CUSTOM'] = custom_dataset

def _global_import(name):
  p = __import__(name, globals(), locals(), level=1)
  globals().pop(name)
  lst = p.__all__ if '__all__' in dir(p) else dir(p)
  for k in lst:
    # add global varaible
    globals()[k] = p.__dict__[k]

    # register in Dataset Factory
    AntDatasetFactory.factory_dataset[k] = p.__dict__[k]


for _, module_name, _ in walk_packages([os.path.dirname(__file__)]):
  if not module_name.startswith('_'):
    if module_name in ['tfrecordsreader', 'dataset', 'simplecsvs', 'simpleimages', 'standard']:
      continue

    _global_import(module_name)

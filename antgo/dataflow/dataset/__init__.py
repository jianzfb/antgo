#  -*- coding: UTF-8 -*-
# @Time    : 18-3-26
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
from pkgutil import walk_packages
import os
import os.path
from .simplecsvs import *
from .simpleimages import *
from .simplevideos import *
from .standard import *

class AntDatasetFactory(object):
  factory_dataset = {}

  @staticmethod
  def dataset(name, parse_flag=''):
    if name in AntDatasetFactory.factory_dataset:
      return AntDatasetFactory.factory_dataset[name]

    for dataset_name, dataset_obj in AntDatasetFactory.factory_dataset.items():
      if dataset_name.lower() == name.lower():
        return dataset_obj

    if parse_flag == 'csv':
      return CSV
      
    if name.lower().startswith('image'):
      return SimpleImages
    elif name.lower().startswith('video'):
      return SimpleVideos

    return Standard

  @staticmethod
  def add_custom_dataset(custom_dataset):
    for dataset_name, dataset_obj in AntDatasetFactory.factory_dataset.items():
      if dataset_obj == custom_dataset:
        return

    AntDatasetFactory.factory_dataset[custom_dataset.__name__] = custom_dataset

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

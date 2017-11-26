#  -*- coding: UTF-8 -*-
from pkgutil import walk_packages
import os
import os.path
from .standard import *
from .csvs import *
from mnist import *
def global_import(name):
  p = __import__(name, globals(), locals(), level=1)
  globals().pop(name)
  lst = p.__all__ if '__all__' in dir(p) else dir(p)
  for k in lst:
    globals()[k] = p.__dict__[k]

for _, module_name, _ in walk_packages([os.path.dirname(__file__)]):
  if not module_name.startswith('_'):
    global_import(module_name)


def AntDataset(dataset_name, parse_flag=''):
  # absorb some small error
  if dataset_name in globals():
    return globals()[dataset_name]
  elif dataset_name.capitalize() in globals():
    return globals()[dataset_name.capitalize()]
  elif dataset_name.title() in globals():
    return globals()[dataset_name.title()]
  elif dataset_name.upper() in globals():
    return globals()[dataset_name.upper()]
  elif dataset_name.lower() in globals():
    return globals()[dataset_name.lower()]
  else:
    if parse_flag == 'csv':
      return CSV

    return Standard

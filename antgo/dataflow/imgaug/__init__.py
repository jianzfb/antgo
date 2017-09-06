# encoding=utf-8
# @Time    : 17-6-22
# @File    : __init__.py
# @Author  : jian<jian@mltalker.com>
import os
from pkgutil import walk_packages

__all__ = []


def global_import(name):
  p = __import__(name, globals(), locals(), level=1)
  lst = p.__all__ if '__all__' in dir(p) else dir(p)
  del globals()[name]
  for k in lst:
    globals()[k] = p.__dict__[k]

for _, module_name, _ in walk_packages(
        [os.path.dirname(__file__)]):
  if not module_name.startswith('_'):
    global_import(module_name)


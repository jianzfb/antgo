#  -*- coding: UTF-8 -*-
#  File: __init__.py
#  Author: jian(jian@mltalker.com)
from __future__ import unicode_literals

from pkgutil import walk_packages
import os

__is_colab__ = False
if 'COLAB_RELEASE_TAG' in os.environ:
    __is_colab__ = True


def _global_import(name):
    p = __import__(name, globals(), None, level=1)
    lst = p.__all__ if '__all__' in dir(p) else dir(p)
    del globals()[name]
    for k in lst:
        globals()[k] = p.__dict__[k]

_global_import('utils')


def is_in_colab():
    global __is_colab__
    return __is_colab__
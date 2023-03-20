# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 13:18
# @File    : util.py
# @Author  : jian<jian@mltalker.com>

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import argparse
import importlib
import os
import sys

def load_extmodule(ext_module_file):
    ext_module_dir = os.path.dirname(ext_module_file)
    key_model = ext_module_file
    dot_pos = key_model.rfind(".")
    if dot_pos != -1:
        key_model = key_model[0:dot_pos]

    sys.path.append(ext_module_dir)
    key_model = os.path.normpath(key_model)
    key_model = os.path.relpath(key_model, ext_module_dir)
    key_model = key_model.replace('/', '.')
    if key_model.startswith('.'):
        key_model = key_model[1:]
    
    importlib.import_module(key_model)
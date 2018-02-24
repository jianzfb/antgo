# -*- coding: UTF-8 -*-
# Time: 10/10/17
# File: utils.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import imp
import sys


def main_context(main_file, source_paths):
  # filter .py
  key_model = main_file
  dot_pos = key_model.rfind(".")
  if dot_pos != -1:
    key_model = key_model[0:dot_pos]

  sys.path.append(source_paths)
  f, p, d = imp.find_module(key_model, [source_paths])
  module = imp.load_module('mm', f, p, d)
  return module.get_global_context()

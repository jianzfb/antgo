# -*- coding: UTF-8 -*-
# Time: 10/10/17
# File: utils.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import importlib
import os
import sys
from antgo.measures import *
from antgo.dataflow.dataset import *


def main_context(main_file, source_paths):
  # filter .py
  # 1.step load user custom module
  key_model = main_file
  dot_pos = key_model.rfind(".")
  if dot_pos != -1:
    key_model = key_model[0:dot_pos]

  key_model = os.path.normpath(key_model)
  key_model = key_model.replace('/', '.')
  sys.path.append(source_paths)
  module = importlib.import_module(key_model)

  for k, v in module.__dict__.items():
    # 2.step check user custom measure method
    if '_BASE_MEASURE' in dir(v) and k != 'AntMeasure':
      AntMeasuresFactory.add_custom_measure(v)

    # 3.step check user custom dataset parse
    if '_BASE_DATASET' in dir(v) and k != 'Dataset':
      AntDatasetFactory.add_custom_dataset(v)

  return module.get_global_context()


def main_logo():
  logo_str='''

   ,---.      .-._        ,--.--------.     _,---.     _,.---._     
 .--.'  \    /==/ \  .-._/==/,  -   , -\_.='.'-,  \  ,-.' , -  `.   
 \==\-/\ \   |==|, \/ /, |==\.-.  - ,-./==.'-     / /==/_,  ,  - \  
 /==/-|_\ |  |==|-  \|  | `--`\==\- \ /==/ -   .-' |==|   .=.     | 
 \==\,   - \ |==| ,  | -|      \==\_ \|==|_   /_,-.|==|_ : ;=:  - | 
 /==/ -   ,| |==| -   _ |      |==|- ||==|  , \_.' )==| , '='     | 
/==/-  /\ - \|==|  /\ , |      |==|, |\==\-  ,    ( \==\ -    ,_ /  
\==\ _.\=\.-'/==/, | |- |      /==/ -/ /==/ _  ,  /  '.='. -   .'   
 `--`        `--`./  `--`      `--`--` `--`------'     `--`--''     

'''

  print(logo_str)



# encoding=utf-8
# @Time    : 17-8-10
# @File    : challenge_task.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.context import *
from antgo.dataflow.recorder import *

import tensorflow as tf

# 1.step set context (take control interaction with antgo)
ctx = Context()

# 2.step build custom infer process
def infer_callback(data_source, dump_dir):
  '''
  implement custom forward process
  :param data_source: 
    test data source
  :param dump_dir: 
    temp data savble path
  :return: 
  '''

  # 2.1 step load model

  # 2.2 step traverse data and running forward process
  for data in data_source.iterator_value():
    # ...
    # save forward result
    # mask is forward process result
    # if using batch, mask is a list
    mask = None
    ctx.recorder.record(mask)


# 3.step bind infer_callback
ctx.infer_process = infer_callback

# how to runing at terminal
# antgo challenge --main_file=challenge_task.py --main_folder=... --task=portrait-task.xml
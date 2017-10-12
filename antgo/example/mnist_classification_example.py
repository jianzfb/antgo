# -*- coding: UTF-8 -*-
# Time: 10/11/17
# File: mnist_classification_example.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
import numpy as np


##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 1.1 step define chart channel ###########
##################################################


##################################################
######## 2.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # reorganize as batch
  # batch_data = BatchData(Node.inputs(data_source), batch_size=64)

  # print('training: set batch size %d'%ctx.params.batch_size)

  iter = 0
  for data, label in data_source.iterator_value():
    print('iterator %d - batch-size: %d'%(iter, data.shape[0]))
    print(data.shape)
    print('id %d label'%(label['id']))
    iter += 1

  print('stop training process')


###################################################
######## 3.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  # print('inference: set batch size %d'%ctx.params.batch_size)

  # traverse the whole test dataset
  for data in data_source.iterator_value():
    predict_label = random.randint(0,9)

    ctx.recorder.record(predict_label)

  print('stop inference process')


###################################################
####### 4.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
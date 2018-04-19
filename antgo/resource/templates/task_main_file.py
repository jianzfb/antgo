# -*- coding: UTF-8 -*-
# @Time    : 18-4-18
# @File    : task_main_file.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
###### 1.1.step build visualization chart  #######
##################################################
# every channel bind value type (NUMERIC, HISTOGRAM, IMAGE)
loss_channel = ctx.job.create_channel("loss","NUMERIC")
## bind channel to chart,every chart could include some channels
ctx.job.create_chart([loss_channel], "Loss Curve", "step", "value")


##################################################
######## 2.step custom model building code  ######
##################################################
# your model code

##################################################
######## 2.1.step custom dataset parse code ######
##################################################
# your dataset parse code
# class MyDataset(Dataset):
#   def __init__(self, train_or_test, dir=None, params=None):
#     super(MyDataset, self).__init__(train_or_test, dir, params)
#
#   @property
#   def size(self):
#     return ...
#
#   def split(self, split_params, split_method):
#     assert (split_method == 'holdout')
#     return self, MyDataset('val', self.dir, self.ext_params)
#
#   def data_pool(self):
#     pass
#   def model_fn(self, *args, **kwargs):
#     # for tfrecords data
#     pass

##################################################
######## 2.2.step custom metric code        ######
##################################################
# your metric code
# class MyMeasure(AntMeasure):
#   def __init__(self, task):
#     super(MyMeasure, self).__init__(task, 'MyMeasure')
#
#   def eva(self, data, label):
#     return {'statistic': {'name': self.name,
#                           'value': [{'name': self.name, 'value': ..., 'type': 'SCALAR'}]}}


##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  iter = 0
  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size * ctx.params.num_clones))
    for _ in range(rounds):
      # execute training process
      _, loss_val = ...
      
      # increment 1
      iter = iter + 1


###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  for _ in range(data_source.size):
    # execute infer process
    logits = ...
    
    # record
    ctx.recorder.record(logits)


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
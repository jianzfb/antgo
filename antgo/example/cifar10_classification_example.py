# -*- coding: UTF-8 -*-
# @Time    : 17-11-22
# @File    : cifar10_classification_example.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tensorflow as tf
slim = tf.contrib.slim
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from dpn import *
from antgo.trainer.tftrainer import *
from antgo.utils._resize import *


##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2 step define chart channel ###########
##################################################
# channel 1
loss_channel = ctx.job.create_channel('loss', 'NUMERIC')
# chart 1
ctx.job.create_chart([loss_channel], 'loss', 'step', 'value')


##################################################
######## 3.step model building (tensorflow) ######
##################################################
class DPNModel(ModelDesc):
  def __init__(self):
    super(DPNModel, self).__init__()
  
  def model_fn(self, is_training=True, *args, **kwargs):
    # image x
    x_input = tf.placeholder(tf.float32, [64, 32, 32, 3], name='x')
    logits = DPN98(x_input, classes=10, pooling='avg')
    if is_training:
      y = tf.placeholder(tf.int32, [64], name='y')
      ce_loss = tf.losses.sparse_softmax_cross_entropy(y, logits)
      return ce_loss
    else:
      return logits


##################################################
######## 4.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  ##########  1.step reorganized as batch ########
  batch = BatchData(Node.inputs(data_source), batch_size=64)
  
  ##########  2.step building model ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.deploy(DPNModel())
  
  ##########  3.step start training ##############
  iter = 0
  for epoch in range(tf_trainer.max_epochs):
    data_generator = batch.iterator_value()
    while True:
      try:
        _, loss_val = tf_trainer.run(data_generator, {'x': 0, 'y': 1})
        # record loss value
        if iter % 50 == 0:
          loss_channel.send(iter, loss_val)
        iter += 1
      
        break
      except StopIteration:
        break
  
    # save
    tf_trainer.snapshot(epoch)


###################################################
######## 5.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  pass

###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
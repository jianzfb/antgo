# -*- coding: UTF-8 -*-
# @Time : 2018/8/6
# @File : lenet_cli.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.codebook.tf.preprocess import *
from antgo.trainer.tftrainer import *
from antgo.utils._resize import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

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
class LeNetModel(ModelDesc):
  def __init__(self):
    super(LeNetModel, self).__init__('LeNet')

  def model_fn(self, is_training=True, *args, **kwargs):
    batch_image = tf.placeholder(tf.float32, (ctx.params.batch_size, 28, 28, 1), name='image')
    batch_label = None
    if is_training:
      batch_label = tf.placeholder(tf.int32, (ctx.params.batch_size), name='label')

    # Implement the LeNet-5 neural network architecture
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=None,
                        weights_initializer=slim.variance_scaling_initializer()):

      # Layer 1: Convolutional. Input = batch x 32 x 32 x 1. Output = 28 x 28 x 6
      conv_1 = slim.conv2d(batch_image, 6, [5,5], stride=1, activation_fn=tf.nn.relu, padding='VALID')

      # Pooling. Input = batch x 28 x 28 x 6. Output = batch x 14 x 14 x 6
      pool_1 = slim.max_pool2d(conv_1, [2,2], stride=2, padding='VALID')

      # Layer 2: Output = batch x 10 x 10 x 16
      conv_2 = slim.conv2d(pool_1, 16, [5,5], stride=1, activation_fn=tf.nn.relu, padding='VALID')

      # POOLING. INPUT = batch x 10 x 10 x 16. Output = batch x 5 x 5 x 16
      pool_2 = slim.max_pool2d(conv_2, [2,2], stride=2, padding='VALID')

      # Flatten. Input = batch x 5 x 5 x 16. Output = batch x 400
      fc_1 = tf.contrib.layers.flatten(pool_2)

      # Layer 3: Fully Connected. Input = batch x 400. Output = batch x 120
      fc_1 = slim.fully_connected(fc_1, 120)

      # Activation
      fc_1 = tf.nn.relu(fc_1)

      # Layer 4: Fully Connected. Input = batch x 120. Output = batch x 84
      fc_2 = slim.fully_connected(fc_1, 84)
      # Activation
      fc_2 = tf.nn.relu(fc_2)

      # Layer 5: Fully Connected. Input = batch x 84. Output = batch x 10
      fc_3 = slim.fully_connected(fc_2, 10)

      # loss function
      if is_training:
        # return loss
        batch_label_one_hot = slim.one_hot_encoding(batch_label, 10)
        loss = tf.losses.softmax_cross_entropy(batch_label_one_hot, fc_3)
        return loss
      else:
        # return logits
        return fc_3


##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # 1.step build CNN model
  lenet_model = LeNetModel(data_source)
  tf_trainer = TFTrainer(ctx, dump_dir, is_training=True)
  tf_trainer.deploy(lenet_model)

  # 2.step build data source pipeline
  batch_data_source = BatchData(Node.inputs(data_source), ctx.params.batch_size)

  # 3.step connect data source to CNN
  for epoch in range(ctx.params.max_epochs):
    batch_data_generator = batch_data_source.iterator_value()
    while True:
      try:
        _, loss_val = tf_trainer.run(batch_data_generator, binds={'image': 0, 'label': 1})

        if tf_trainer.iter_at % 50 == 0:
          loss_channel.send(tf_trainer.iter_at, loss_val)
      except:
        break

    tf_trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  # 1.step build CNN model
  lenet_model = LeNetModel()
  tf_trainer = TFTrainer(ctx, dump_dir, is_training=False)
  tf_trainer.deploy(lenet_model)

  # 2.step infer
  for data in data_source.iterator_value():
    # execute infer process
    # logits = ...
    preprocess_data = None
    if len(data.shape) == 2:
      preprocess_data = np.expand_dims(data, -1)
    elif len(data.shape) == 3 and data.shape[2] > 1:
      preprocess_data = np.expand_dims(data[:,:,0], -1)
    else:
      preprocess_data = data

    height, width, _ = preprocess_data.shape
    if height != 28 or width != 28:
      preprocess_data = resize(preprocess_data, [28, 28])

    preprocess_data = np.expand_dims(preprocess_data, 0)
    logits = tf_trainer.run_dict({'image': preprocess_data})
    image_label = np.argmax(logits, -1)

    # record
    ctx.recorder.record({'RESULT':image_label[0][0]})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
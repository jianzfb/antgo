# -*- coding: UTF-8 -*-
# @Time    : 2018/10/31 1:56 PM
# @File    : mnist.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.trainer.tftrainer import *
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
loss_channel = ctx.job.create_channel("loss", "NUMERIC")
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
######## 2.3.step custom model code        ######
##################################################
# your model code
class MNISTModel(ModelDesc):
  def __init__(self):
    super(MNISTModel, self).__init__()

  def model_fn(self, is_training=True, *args, **kwargs):
    image = tf.placeholder(tf.float32, [None, 784], name='image')
    label = tf.placeholder(tf.float32, [None], name='label')

    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0)

    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(image, [-1, 28, 28, 1])

    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
      h_conv1 = slim.conv2d(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
      h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
      h_conv2 = slim.conv2d(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
      h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      # reshape tensor into a batch of vectors
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    tf.summary.histogram('activations', h_pool2)

    # Densely connected layer with 1024 neurons.
    h_fc1 = slim.dropout(
      slim.fully_connected(
        h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=is_training)

    # Compute logits (1 per class) and compute loss.
    logits = slim.fully_connected(h_fc1, 10, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # deploy model
  tf_trainer = TFTrainer(dump_dir,True)
  tf_trainer.deploy(MNISTModel())

  # organize data as batch
  batch_node = BatchData(Node.inputs(data_source), 12)

  iter = 0
  for epoch in range(100):
    for k,v in batch_node.iterator_value():
      # execute training process
      k = k.reshape([-1, 784])
      _, loss_val = tf_trainer.run(image=k, label=v)

      # increment 1
      iter = iter + 1

    tf_trainer.snapshot(epoch)

###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  for _ in range(data_source.size):
    # execute infer process
    # logits = ...

    # record
    ctx.recorder.record(logits)


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
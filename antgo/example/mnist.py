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
from antgo.ant.debug import *

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
    # 转换tensor形状到原始形状
    feature = tf.reshape(image, [-1, 28, 28, 1])

    # 第一组卷积
    with tf.variable_scope('conv_layer1'):
      h_conv1 = slim.conv2d(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
      h_pool1 = tf.nn.max_pool(
        h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第二组卷积
    with tf.variable_scope('conv_layer2'):
      h_conv2 = slim.conv2d(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
      h_pool2 = tf.nn.max_pool(
        h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # 记录激活分布
    tf.summary.histogram('activations', h_pool2)

    # 消除实验模块
    with ctx.block('auxilary_fc') as afc:
      if afc.activate:
        h_pool2_flat = slim.fully_connected(h_pool2_flat, 1024, activation_fn=tf.nn.relu)

    # 第一组全连接（紧跟dropout模块）
    h_fc1 = slim.dropout(
      slim.fully_connected(
        h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=is_training)

    # 第二组全连接
    logits = slim.fully_connected(h_fc1, 10, activation_fn=None)

    if is_training:
      # 训练阶段，计算损失函数
      label = tf.placeholder(tf.float32, [None], name='label')
      target = tf.one_hot(tf.cast(label, tf.int32), 10, 1, 0)
      loss = tf.losses.softmax_cross_entropy(target, logits)
      return loss
    else:
      # 推断阶段，返回计算概率值
      logits = tf.nn.softmax(logits)
      return logits


##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # 部署模型
  tf_trainer = TFTrainer(dump_dir,True)
  tf_trainer.deploy(MNISTModel())

  # 将数据源链接Batch模块
  batch_node = BatchData(Node.inputs(data_source), ctx.params.batch_size)

  iter = 0
  for epoch in range(tf_trainer.max_epochs):
    for k,v in batch_node.iterator_value():
      # 执行训练
      k = k.reshape([-1, 784])
      _, loss_val = tf_trainer.run(image=k, label=v['category_id'])

      # increment 1
      iter = iter + 1

    tf_trainer.snapshot(epoch)

###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  # 部署模型
  tf_trainer = TFTrainer(dump_dir,False)
  tf_trainer.deploy(MNISTModel())

  for k in data_source.iterator_value():
    # 推断
    k = k.reshape([-1, 784])
    logits = tf_trainer.run(image=k)

    # 记录到数据库
    ctx.recorder.record({'RESULT':np.argmax(logits[0]), 'RESULT_TYPE': 'SCALAR'})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback


###################################################
###########    6.step test run         ############
###########                            ############
###################################################
if __name__ == '__main__':
  # 1.step debug training process
  debug_training_process(lambda :(np.zeros((28,28)), 0))

  # 2.step debug infer process
  debug_infer_process(lambda :np.zeros((28,28)))
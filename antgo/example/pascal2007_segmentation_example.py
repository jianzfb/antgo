# -*- coding: UTF-8 -*-
# @Time    : 17-11-17
# @File    : pascal2007_segmentation_example.py.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import densenet
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

# channel 2
accuracy_channel = ctx.job.create_channel('accuracy', 'NUMERIC')

# chart 1
ctx.job.create_chart([loss_channel], 'loss', 'step', 'value')

# chart 2
ctx.job.create_chart([accuracy_channel], 'accuracy', 'step', 'value')


##################################################
######## 3.step model building (tensorflow) ######
##################################################
def gcn_block(inputs,
              num_class,
              kernel_size,
              scope=None):
  with tf.variable_scope(scope, 'gcn_block', [inputs]):
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=None,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
      left_conv1 = slim.conv2d(inputs, num_class, [kernel_size, 1])
      left_conv2 = slim.conv2d(left_conv1, num_class, [1, kernel_size])
      
      right_conv1 = slim.conv2d(inputs, num_class, [1, kernel_size])
      right_conv2 = slim.conv2d(right_conv1, num_class, [kernel_size, 1])
      
      result_sum = tf.add(left_conv2, right_conv2, name='gcn_module')
      return result_sum


def gcn_br(inputs, scope):
  with tf.variable_scope(scope, 'gcn_br', [inputs]):
    with slim.arg_scope([slim.conv2d],
                        padding='SAME',
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        normalizer_params=None,
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(0.0001),
                        biases_initializer=tf.zeros_initializer(),
                        biases_regularizer=tf.contrib.layers.l2_regularizer(0.0002)):
      num_class = inputs.get_shape()[3]
      conv = slim.conv2d(inputs, num_class, [3, 3])
      conv = slim.conv2d(conv, num_class, [3, 3], activation_fn=None)
      result_sum = tf.add(inputs, conv, name='fcn_br')
      return result_sum


def upsample(inputs, height, width, scope):
  with tf.variable_scope(scope, 'upsample', [inputs]):
    height = int(height)
    width = int(width)
    result = tf.image.resize_bilinear(inputs, [height, width])
    return result


def combine(low_inputs, high_inputs, name):
  with tf.variable_scope(name, 'combine_add', [low_inputs, high_inputs]):
    result = tf.add(low_inputs, high_inputs, name=name)
    return result


def gcn_net(each_layer_output, num_classes):
  res_2 = each_layer_output['densenet121/dense_block1']
  res_3 = each_layer_output['densenet121/dense_block2']
  res_4 = each_layer_output['densenet121/dense_block3']
  res_5 = each_layer_output['densenet121/dense_block4']

  res_2_gcn = gcn_block(res_2, num_classes, 15, 'gcn_2')
  res_3_gcn = gcn_block(res_3, num_classes, 15, 'gcn_3')
  res_4_gcn = gcn_block(res_4, num_classes, 15, 'gcn_4')
  res_5_gcn = gcn_block(res_5, num_classes, 15, 'gcn_5')

  res_2_br_1 = gcn_br(res_2_gcn, 'br_2_1')
  res_3_br_1 = gcn_br(res_3_gcn, 'br_3_1')
  res_4_br_1 = gcn_br(res_4_gcn, 'br_4_1')
  res_5_br_1 = gcn_br(res_5_gcn, 'br_5_1')

  _, height, width, _ = res_4_br_1.shape
  res_5_upsample = upsample(res_5_br_1, height, width, 'upsample_5')
  res_4_combine = combine(res_5_upsample, res_4_br_1, 'combine_4')

  res_4_br_2 = gcn_br(res_4_combine, 'br_4_2')
  _, height, width, _ = res_3_br_1.shape
  res_4_upsample = upsample(res_4_br_2, height, width, 'upsample_4')
  res_3_combine = combine(res_4_upsample, res_3_br_1, 'combine_3')

  res_3_br_2 = gcn_br(res_3_combine, 'br_3_2')
  _, height, width, _ = res_2_br_1.shape
  res_3_upsample = upsample(res_3_br_2, height, width, 'upsample_3')
  res_2_combine = combine(res_3_upsample, res_2_br_1, 'combine_2')

  res_2_br_2 = gcn_br(res_2_combine, 'br_2_2')
  _, height, width, _ = res_2_br_2.shape
  res_2_upsample = upsample(res_2_br_2, height * 2, width * 2, 'upsample_2')
  result_br_1 = gcn_br(res_2_upsample, 'br_result_1')
  _, height, width, _ = result_br_1.shape

  result_1 = upsample(result_br_1, height * 2, width * 2, 'upsample_x')
  result_br_2 = gcn_br(result_1, 'br_result_2')

  return result_br_2


class GCNSegModel(ModelDesc):
  def __init__(self):
    super(GCNSegModel, self).__init__()

  def model_fn(self, is_training=True, *args, **kwargs):
    # image x
    x_input = tf.placeholder(tf.float32, [1, None, None, 3], name='x')
    # resize x
    x = tf.image.resize_bilinear(x_input, [512, 512], align_corners=True)

    # dense net
    with slim.arg_scope(densenet.densenet_arg_scope()):
      _, each_layer_output = densenet.densenet121(inputs=x,
        num_classes=21,
        is_training=True,
        reuse=None)

    if is_training:
      # label y
      y_input = tf.placeholder(tf.uint8, [1, None, None, 1], name='y')

    # gcn net
    with tf.variable_scope(None, 'GCN', [each_layer_output], reuse=None):
      logits = gcn_net(each_layer_output, num_classes=21)

      if is_training:
        # resize y
        y = tf.image.resize_nearest_neighbor(y_input, [512, 512], align_corners=True)
        y = tf.squeeze(y, axis=3)

        # cross entropy loss
        labels = tf.one_hot(y, depth=21)
        ce_loss = tf.losses.softmax_cross_entropy(labels, logits)

        # regularization loss
        regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # gather all loss
        sum_loss = regularization_loss
        sum_loss.append(ce_loss)
        sum_loss = tf.add_n(sum_loss)

        # pixel accuracy
        predict_y = tf.argmax(logits, axis=3)
        accuracy = tf.reduce_mean(tf.abs(tf.cast(predict_y, tf.float32) - tf.cast(y, tf.float32)))

        return sum_loss, accuracy
      else:
        return logits


##################################################
######## 4.step define training process  #########
##################################################
def custom_train_data(*args, **kwargs):
  image, label = args[0]
  image = np.expand_dims(image, 0)
  seg_label = label['segmentation_map']
  seg_label = np.expand_dims(seg_label, 0)
  seg_label = np.expand_dims(seg_label, 3)

  return image, seg_label


def training_callback(data_source, dump_dir):
  ##########  1.step reorganized as batch ########
  seg_data_source = FilterNode(Node.inputs(data_source), lambda x: 'segmentation' in x[1])
  seg_data_source = Node(name='custom_organize', action=custom_train_data, inputs=Node.inputs(seg_data_source))
  
  ##########  2.step building model ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.deploy(GCNSegModel())
  
  ##########  3.step start training ##############
  iter = 0
  for epoch in range(tf_trainer.max_epochs):
    data_generator = seg_data_source.iterator_value()
    while True:
      try:
        _, loss_val, accuracy_val = tf_trainer.run(data_generator, {'x': 0, 'y': 1})
        # record loss value
        if iter % 50 == 0:
          loss_channel.send(iter, loss_val)
          accuracy_channel.send(iter, accuracy_val)
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
  ##########  1.step reorganized as batch #########
  batch_data_source = BatchData(Node.inputs(data_source), batch_size=1)

  ##########  2.step building model ###############
  tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=False)
  tf_trainer.deploy(GCNSegModel())

  ##########  3.step start inference ##############
  data_generator = batch_data_source.iterator_value()
  count = 0
  while True:
    try:
      logits, feed_data = tf_trainer.run(data_generator, {'x': 0}, whats=True)
      logits = np.squeeze(logits[0], axis=0)
      _, height, width = feed_data[0][0].shape[0:3]

      # resize to original size
      resized_logits = resize(logits, (height, width))
      mask = np.argmax(resized_logits, axis=2)

      # record
      ctx.recorder.record(mask)
      if count == 10:
        break
      count += 1
      
    except StopIteration:
      break


###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
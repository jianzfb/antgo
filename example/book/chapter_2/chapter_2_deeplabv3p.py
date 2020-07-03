# -*- coding: UTF-8 -*-
# @Time    : 2018/11/13 10:52 AM
# @File    : chapter_2_deeplabv3p.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from numpy.lib.stride_tricks import as_strided
from antgo.trainer.tftrainer import *
from antgo.utils._resize import *
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from antgo.codebook.tf.preprocess import *
import mobilenet_v2
import mobilenet
from antgo.codebook.tf.dataset import *

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


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

##################################################
######## 3.step model building (tensorflow) ######
##################################################
class Mobilenetv2Seg(ModelDesc):
  def __init__(self, data_source):
    super(Mobilenetv2Seg, self).__init__(model_data_source=data_source)

  def model_input(self, is_training, *args, **kwargs):
    data_source = self.data_source
    if is_training:
      image, label = data_source.model_fn()

      means = np.array([ctx.params.r_mean, ctx.params.g_mean, ctx.params.b_mean])
      image = tf.cast(image, tf.float32)

      channels = tf.split(axis=2, num_or_size_splits=3, value=image)
      for i in range(3):
        channels[i] -= means[i]

      image = tf.concat(axis=2, values=channels)
      image = tf.div(image, 255.0)
      image = tf.expand_dims(image, axis=0)
      image = tf.image.resize_bilinear(image, [ctx.params.output_size, ctx.params.output_size])
      image = tf.squeeze(image, 0)

      label = tf.where(label > 128,
                       tf.fill([ctx.params.origin_height, ctx.params.origin_width, 1], 1),
                       tf.fill([ctx.params.origin_height, ctx.params.origin_width, 1], 0))
      label = tf.cast(label, tf.uint8)
      label = tf.squeeze(label, axis=2)

      images, labels = tf.train.shuffle_batch([image, label],
                                              batch_size=ctx.params.batch_size,
                                              num_threads=32,
                                              capacity=4000,
                                              min_after_dequeue=1000)
      data_queue = slim.prefetch_queue.prefetch_queue([images, labels], capacity=100, num_threads=10)
      return data_queue
    else:
      image = data_source.model_fn()
      original_image = image

      means = np.array([ctx.params.r_mean, ctx.params.g_mean, ctx.params.b_mean])
      image = tf.cast(image, tf.float32)
      channels = tf.split(axis=2, num_or_size_splits=3, value=image)
      for i in range(3):
        channels[i] -= means[i]

      image = tf.concat(axis=2, values=channels)
      image = tf.div(image, 255.0)
      image = tf.expand_dims(image, axis=0)
      image = tf.image.resize_bilinear(image, [ctx.params.output_size, ctx.params.output_size])

      return image, original_image

  def split_separable_conv2d(self,
                             inputs,
                             filters,
                             kernel_size=3,
                             rate=1,
                             weight_decay=0.00004,
                             depthwise_weights_initializer_stddev=0.33,
                             pointwise_weights_initializer_stddev=0.06,
                             scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.

    This operation differs from `tf.layers.separable_conv2d` as this operation
    applies activation function between depthwise and pointwise conv2d.

    Args:
      inputs: Input tensor with shape [batch, height, width, channels].
      filters: Number of filters in the 1x1 pointwise convolution.
      kernel_size: A list of length 2: [kernel_height, kernel_width] of
        of the filters. Can be an int if both values are the same.
      rate: Atrous convolution rate for the depthwise convolution.
      weight_decay: The weight decay to use for regularizing the model.
      depthwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for depthwise convolution.
      pointwise_weights_initializer_stddev: The standard deviation of the
        truncated normal weight initializer for pointwise convolution.
      scope: Optional scope for the operation.

    Returns:
      Computed features after split separable conv2d.
    """
    outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
        stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
    return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
        stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')

  def decoder(self, small_layer, middle_layer, is_training, atrous_rates, weight_decay):
    # ASPP block
    depth = 256
    with tf.variable_scope(None, 'ASPP_SCOPE', [small_layer]):
      batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
      }
      with arg_scope([slim.conv2d, slim.separable_conv2d],
                     padding='SAME',
                     activation_fn=nn.relu,
                     normalizer_fn=slim.batch_norm,
                     weights_initializer=initializers.xavier_initializer(),
                     weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          logits_list = []
          # small_layer size 4 x 4
          # 1.step global pooling
          _, pool_h, pool_w, _ = small_layer.get_shape()
          image_feature = tf.reduce_mean(small_layer, axis=[1, 2], keep_dims=True)
          image_feature = slim.conv2d(image_feature, depth, 1)
          image_feature = tf.image.resize_bilinear(image_feature, [int(pool_h), int(pool_w)])
          logits_list.append(image_feature)

          # 2.step 1 x 1 convolution
          logits_list.append(slim.conv2d(small_layer, depth, 1, scope="ASPP_SCOPE" + str(0)))

          # 3.step ASPP
          for i, rate in enumerate(atrous_rates, 1):
            scope = 'ASPP_SCOPE' + str(i)
            aspp_features = self.split_separable_conv2d(
              small_layer,
              filters=depth,
              rate=rate,
              weight_decay=weight_decay,
              scope=scope)
            logits_list.append(aspp_features)

          # merge logits
          concat_logits = tf.concat(logits_list, 3)
          concat_logits = slim.conv2d(concat_logits, depth, 1)

    # drop out
    concat_logits = slim.dropout(concat_logits, keep_prob=0.9, is_training=is_training)

    # refined decoder
    with tf.variable_scope(None, 'RefineDecoder', [concat_logits, middle_layer]):
      batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'is_training': is_training
      }

      with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                          padding='SAME',
                          activation_fn=nn.relu,
                          normalizer_fn=slim.batch_norm,
                          normalizer_params=batch_norm_params,
                          weights_initializer=initializers.xavier_initializer(),
                          weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          middle_layer_height = int(middle_layer.shape[1])
          middle_layer_width = int(middle_layer.shape[2])
          middle_layer_channel = int(middle_layer.shape[3])

          decoder_features_list = []
          decoder_features_list.append(
            tf.image.resize_bilinear(concat_logits, [middle_layer_height, middle_layer_width]))
          decoder_features_list.append(slim.conv2d(middle_layer, 48, 1))
          concat_decoder_features = tf.concat(decoder_features_list, 3)

          decoder_depth = 256
          concat_decoder_features = self.split_separable_conv2d(concat_decoder_features,
                                                                filters=decoder_depth,
                                                                rate=1,
                                                                weight_decay=weight_decay,
                                                                scope='decoder_conv0')
          concat_decoder_features = self.split_separable_conv2d(concat_decoder_features,
                                                                filters=decoder_depth,
                                                                rate=1,
                                                                weight_decay=weight_decay,
                                                                scope='decoder_conv1')

          _, branch_h, branch_w, _ = concat_decoder_features.get_shape()
          concat_branch_feature = tf.image.resize_bilinear(concat_decoder_features,[int(branch_h) * 4, int(branch_w) * 4])
          logits = slim.conv2d(concat_branch_feature,2,1,activation_fn=None, normalizer_fn=None)
          return logits

  def _mobilenet_v2(self,
                    net,
                    depth_multiplier,
                    output_stride,
                    reuse=None,
                    scope=None,
                    final_endpoint=None):
    """Auxiliary function to add support for 'reuse' to mobilenet_v2.

    Args:
      net: Input tensor of shape [batch_size, height, width, channels].
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      reuse: Reuse model variables.
      scope: Optional variable scope.
      final_endpoint: The endpoint to construct the network up to.

    Returns:
      Features extracted by MobileNetv2.
    """
    with tf.variable_scope(
            scope, 'MobilenetV2', [net], reuse=reuse) as scope:
      return mobilenet_v2.mobilenet_base(
        net,
        conv_defs=mobilenet_v2.V2_DEF,
        depth_multiplier=depth_multiplier,
        min_depth=8 if depth_multiplier == 1.0 else 1,
        divisible_by=8 if depth_multiplier == 1.0 else 1,
        final_endpoint=final_endpoint,
        output_stride=output_stride,
        scope=scope)

  def model_fn(self, is_training=True, *args, **kwargs):
    x_input = None
    y_input = None
    original_image = None
    if len(args) > 0:
      if is_training:
        x_input, y_input = args[0].dequeue()
      else:
        x_input, original_image = args[0]
        x_input = tf.identity(x_input, 'input_node')
    else:
      x_input = tf.placeholder(tf.float32, shape=[1, ctx.params.output_size, ctx.params.output_size, 3],
                               name='input_node')

    with slim.arg_scope(mobilenet.training_scope(is_training=is_training)):
      _, end_points = self._mobilenet_v2(x_input, depth_multiplier=1.0, output_stride=16)

      last_layer = end_points['layer_18/output']              # _ x   8   x 8   x _
      middle_layer = end_points['layer_4/depthwise_output']   # _ x   32  x 32  x _

    with tf.variable_scope(None, 'DECODER', [last_layer, middle_layer], reuse=None):
      logits = self.decoder(last_layer, middle_layer, is_training, [2, 4, 8], 0.00004)

    if is_training:
      # resize to target size
      logits = tf.image.resize_bilinear(logits, [ctx.params.origin_height, ctx.params.origin_width])

      # for labels
      one_hot_labels = slim.one_hot_encoding(y_input, ctx.params.num_classes)
      cross_entroy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits)
      softmax_logits = tf.nn.softmax(logits, dim=-1)
      _, logits_1 = tf.split(softmax_logits, num_or_size_splits=2, axis=3)
      logits_1 = tf.reshape(logits_1, shape=[ctx.params.batch_size, ctx.params.origin_height, ctx.params.origin_width])
      label_batch_float = tf.cast(y_input, tf.float32)

      weights_pt = tf.pow(1 - (logits_1 * label_batch_float + (1 - logits_1) * (1 - label_batch_float)), 2)
      weights_total = (logits_1 * 5.02 + (1 - logits_1) * 4.98) * weights_pt
      focal_loss = weights_total * cross_entroy
      loss = tf.reduce_mean(focal_loss)

      tf.losses.add_loss(loss)
      return loss
    else:
      logits = tf.nn.softmax(logits)
      logits = tf.identity(logits, 'output_node')

      _, positive_s = tf.split(logits, 2, 3)
      return positive_s, original_image


##################################################
######## 4.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  # config trainer
  tf_trainer = TFTrainer(dump_dir)
  tf_trainer.deploy(Mobilenetv2Seg(data_source))

  for epoch in range(ctx.params.max_epochs):
    rounds = int(float(data_source.size) / float(ctx.params.batch_size * ctx.params.num_clones))
    # logger.error(rounds)
    for _ in range(rounds):
      _, loss_val = tf_trainer.run()

    # save
    tf_trainer.snapshot(epoch)


###################################################
######## 5.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  tf_trainer = TFTrainer(dump_dir, is_training=False)
  tf_trainer.deploy(Mobilenetv2Seg(data_source))

  for index in range(data_source.size):
    logits, original_image = tf_trainer.run()
    logits = np.squeeze(logits, axis=0)
    height, width, _ = original_image.shape

    logits = logits[:, :, 0]
    logits = resize(logits, (height, width))
    score_map = logits.copy()

    logits[np.where(logits > 0.5)] = 1
    logits[np.where(logits <= 0.5)] = 0
    mask = logits.astype(np.uint8)

    mask_rgb = original_image * np.expand_dims(mask, -1)
    mask_rgb = mask_rgb.astype(np.uint8)

    # record
    ctx.recorder.record({'RESULT': mask, 'RESULT_TYPE': 'IMAGE',
                         'Frontimage': mask_rgb, 'Frontimage_TYPE': 'IMAGE',
                         'ScoreMap': score_map, 'ScoreMap_TYPE': 'IMAGE'})


###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback

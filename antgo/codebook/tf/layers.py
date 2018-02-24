# -*- coding: UTF-8 -*-
# @Time    : 17-12-20
# @File    : layers.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tensorflow as tf
import numpy as np


def _bilinear_filter(filter_shape, upscale_factor):
  ##filter_shape is [width, height, num_in_channels, num_out_channels]
  kernel_size = filter_shape[1]
  ### Centre location of the filter for which value is calculated
  if kernel_size % 2 == 1:
    centre_location = upscale_factor - 1
  else:
    centre_location = upscale_factor - 0.5
  
  bilinear = np.zeros([filter_shape[0], filter_shape[1]])
  for x in range(filter_shape[0]):
    for y in range(filter_shape[1]):
      ##Interpolation Calculation
      value = (1 - abs((x - centre_location) / upscale_factor)) * (1 - abs((y - centre_location) / upscale_factor))
      bilinear[x, y] = value
  weights = np.zeros(filter_shape)
  for i in range(filter_shape[2]):
    weights[:, :, i, i] = bilinear
  init = tf.constant_initializer(value=weights, dtype=tf.float32)
  
  bilinear_weights = tf.get_variable(name="filter", initializer=init, shape=weights.shape)
  return bilinear_weights


def deconv_bilinear(bottom, out_channels, in_channels, upscale_factor, output_shape=None, name=None):
  kernel_size = 2 * upscale_factor - upscale_factor % 2
  stride = upscale_factor
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name, 'deconv'):
    # Shape of the bottom tensor
    in_shape = tf.shape(bottom)
    
    if output_shape is None:
      h = ((in_shape[1] - 1) * stride) + 1
      w = ((in_shape[2] - 1) * stride) + 1
      new_shape = [in_shape[0], h, w, out_channels]
      output_shape = tf.stack(new_shape)
    else:
      output_shape = tf.stack(output_shape)
    
    filter_shape = [kernel_size, kernel_size, out_channels, in_channels]

    weights = None
    if out_channels == in_channels:
      # filter with bilinear parameters
      weights = _bilinear_filter(filter_shape, upscale_factor)
    else:
      # filter with random
      weights = tf.get_variable("filter",
                                filter_shape,
                                dtype=tf.float32,
                                initializer=tf.random_normal_initializer(0, 0.02))

  return tf.nn.conv2d_transpose(bottom, weights, output_shape, strides=strides, padding='SAME')

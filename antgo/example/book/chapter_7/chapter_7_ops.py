# -*- coding: UTF-8 -*-
# @Time    : 2018/10/15 3:52 PM
# @File    : chapter_7_ops.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib as tf_contrib


##################################################################################
# Normalization function
##################################################################################
def instance_norm(input, name="instance_norm"):
  with tf.variable_scope(name):
    depth = input.get_shape()[3]
    scale = slim.model_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    offset = slim.model_variable("offset", [depth], initializer=tf.constant_initializer(0.0))

    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset


def conv2d(input, output_dim, kernel_size=4, stride=2, use_bias=False, padding='SAME', name="conv2d"):
  with tf.variable_scope(name):
    return slim.conv2d(input,
                       output_dim,
                       kernel_size,
                       stride,
                       padding=padding,
                       activation_fn=None,
                       weights_initializer=tf_contrib.layers.xavier_initializer(),
                       normalizer_fn=None,
                       biases_initializer=tf.zeros_initializer() if use_bias else None)

def deconv2d(input, output_dim, kernel_size=4, stride=2, use_bias=False, padding='SAME', name="deconv2d"):
  with tf.variable_scope(name):
    return slim.conv2d_transpose(input,
                                 output_dim,
                                 kernel_size,
                                 stride,
                                 padding=padding,
                                 activation_fn=None,
                                 weights_initializer=tf_contrib.layers.xavier_initializer(),
                                 biases_initializer=tf.zeros_initializer() if use_bias else None)


##################################################################################
# Residual-block
##################################################################################
def resblock(x_init, channels, kernel_size=3, use_bias=True, name='resblock'):
  with tf.variable_scope(name):
    with tf.variable_scope('res1'):
      x = conv2d(x_init, channels, kernel_size=kernel_size, stride=1, use_bias=use_bias)
      x = instance_norm(x)
      x = relu(x)

    with tf.variable_scope('res2'):
      x = conv2d(x, channels, kernel_size=kernel_size, stride=1, use_bias=use_bias)
      x = instance_norm(x)

    return x + x_init


##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

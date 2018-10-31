# -*- coding: UTF-8 -*-
# @Time    : 10-07-18
# @File    : chapter_8_ops.py
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

def adaptive_instance_norm(content, gamma, beta, epsilon=1e-5):
  # gamma, beta = style_mean, style_std from MLP

  c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
  c_std = tf.sqrt(c_var + epsilon)

  return gamma * ((content - c_mean) / c_std) + beta

##################################################################################
# regular layer function
##################################################################################
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

def linear(x, units, use_bias=True, name='linear'):
  with tf.variable_scope(name):
    x = tf.layers.flatten(x)
    x = slim.fully_connected(x, units)
    return x


##################################################################################
# Activation function
##################################################################################
def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Residual-block
##################################################################################
def resblock(x_init, channels, use_bias=True, name='resblock'):
  with tf.variable_scope(name):
    with tf.variable_scope('res1'):
      x = conv2d(x_init, channels, kernel_size=3, stride=1, use_bias=use_bias)
      x = instance_norm(x)
      x = relu(x)

    with tf.variable_scope('res2'):
      x = conv2d(x, channels, kernel_size=3, stride=1, use_bias=use_bias)
      x = instance_norm(x)

    return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, name='adaptive_resblock'):
  with tf.variable_scope(name):
    with tf.variable_scope('res1'):
      x = conv2d(x_init, channels, kernel_size=3, stride=1, use_bias=use_bias)
      x = adaptive_instance_norm(x, mu, sigma)
      x = relu(x)

    with tf.variable_scope('res2'):
      x = conv2d(x, channels, kernel_size=3, stride=1, use_bias=use_bias)
      x = adaptive_instance_norm(x, mu, sigma)

    return x + x_init


##################################################################################
# Loss function
##################################################################################
def classification_loss(logit, label):
  loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logit))
  return loss

def generator_loss(gan_type, fake):
  fake_loss = 0

  if 'wgan' in gan_type:
    fake_loss = -tf.reduce_mean(fake)

  if gan_type == 'lsgan':
    fake_loss = tf.reduce_mean(tf.squared_difference(fake, 1.0))

  if gan_type == 'gan' or gan_type == 'dragan':
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

  # if gan_type == 'hinge':
  #   fake_loss = -tf.reduce_mean(fake)

  return fake_loss

def discriminator_loss(gan_type, real, fake):
  real_loss = 0
  fake_loss = 0

  if "wgan" in gan_type:
    real_loss = -tf.reduce_mean(real)
    fake_loss = tf.reduce_mean(fake)

  if gan_type == 'lsgan':
    real_loss = tf.reduce_mean(tf.squared_difference(real, 1.0))
    fake_loss = tf.reduce_mean(tf.square(fake))

  if gan_type == 'gan' or gan_type == 'dragan':
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

  # if gan_type== 'hinge':
  #   real_loss = tf.reduce_mean(relu(1.0 - real))
  #   fake_loss = tf.reduce_mean(relu(1.0 + fake))

  loss = real_loss + fake_loss
  return loss

def discriminator_loss_list(type, real, fake):
  n_scale = len(real)
  loss = []

  real_loss = 0
  fake_loss = 0

  for i in range(n_scale):
    if type == 'lsgan':
      real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
      fake_loss = tf.reduce_mean(tf.square(fake[i]))

    if type == 'gan':
      real_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real[i]), logits=real[i]))
      fake_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake[i]), logits=fake[i]))

    loss.append(real_loss + fake_loss)

  return sum(loss)


def generator_loss_list(type, fake):
  n_scale = len(fake)
  loss = []

  fake_loss = 0

  for i in range(n_scale):
    if type == 'lsgan':
      fake_loss = tf.reduce_mean(tf.squared_difference(fake[i], 1.0))

    if type == 'gan':
      fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake[i]), logits=fake[i]))

    loss.append(fake_loss)

  return sum(loss)


def L1_loss(x, y):
  loss = tf.reduce_mean(tf.abs(x - y))
  return loss


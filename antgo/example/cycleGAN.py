# -*- coding: UTF-8 -*-
# @Time    : 18-1-2
# @File    : cycleGAN.py
# @Author  : 
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from antgo.codebook.tf.layers import *
import tensorflow.contrib.slim as slim
from antgo.dataflow.imgaug.regular import *
##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2.step model building (tensorflow) ######
##################################################
def _instance_norm(x):
  with tf.variable_scope("instance_norm"):
    epsilon = 1e-5
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    scale = tf.get_variable('scale', [x.get_shape()[-1]],
      initializer=tf.truncated_normal_initializer(
        mean=1.0, stddev=0.02
      ))
    offset = tf.get_variable(
      'offset', [x.get_shape()[-1]],
      initializer=tf.constant_initializer(0.0)
    )
    out = scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset
    
    return out


def _lrelu(x, leak=0.2, name="lrelu", alt_relu_impl=False):
  with tf.variable_scope(name):
    if alt_relu_impl:
      f1 = 0.5 * (1 + leak)
      f2 = 0.5 * (1 - leak)
      return f1 * x + f2 * abs(x)
    else:
      return tf.maximum(x, leak * x)


def _conv2d(input, o_channels, ks, stride, stddev=0.02, padding='VALID', name="conv2d", do_norm=True, do_relu=True, relufactor=0):
  with tf.variable_scope(name):
    
    conv = tf.contrib.layers.conv2d(
      input, o_channels, ks, stride, padding,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(
        stddev=stddev
      ),
      biases_initializer=tf.constant_initializer(0.0)
    )
    if do_norm:
      conv = _instance_norm(conv)
    
    if do_relu:
      if (relufactor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = _lrelu(conv, relufactor, "lrelu")
    
    return conv


def _deconv2d(inputconv, outshape, o_d=64, f_h=7, f_w=7, s_h=1, s_w=1,
                     stddev=0.02, padding="VALID", name="deconv2d",
                     do_norm=True, do_relu=True, relufactor=0):
  with tf.variable_scope(name):
  
    conv = tf.contrib.layers.conv2d_transpose(
      inputconv, o_d, [f_h, f_w],
      [s_h, s_w], padding,
      activation_fn=None,
      weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
      biases_initializer=tf.constant_initializer(0.0)
    )
  
    if do_norm:
      conv = _instance_norm(conv)
      # conv = tf.contrib.layers.batch_norm(conv, decay=0.9,
      # updates_collections=None, epsilon=1e-5, scale=True,
      # scope="batch_norm")
  
    if do_relu:
      if (relufactor == 0):
        conv = tf.nn.relu(conv, "relu")
      else:
        conv = _lrelu(conv, relufactor, "lrelu")
  
    return conv


def _resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
  """build a single block of resnet.

  :param inputres: inputres
  :param dim: dim
  :param name: name
  :param padding: for tensorflow version use REFLECT; for pytorch version use
   CONSTANT
  :return: a single block of resnet.
  """
  with tf.variable_scope(name):
    out_res = tf.pad(inputres, [[0, 0], [1, 1], [
      1, 1], [0, 0]], padding)
    out_res = _conv2d(
      out_res, dim, 3, 1, 0.02, "VALID", "c1")
    out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
    out_res = _conv2d(
      out_res, dim, 3, 1, 0.02, "VALID", "c2", do_relu=False)
    
    return tf.nn.relu(out_res + inputres)

  
def generator(input, name="generator", skip=False):
  with tf.variable_scope(name):
    f = 7
    ks = 3
    padding = "REFLECT"
    
    pad_input = tf.pad(input, [[0, 0], [ks, ks], [
      ks, ks], [0, 0]], padding)
    o_c1 = _conv2d(
      pad_input, ctx.params.ngf, f, 1, 0.02, name="c1")
    o_c2 = _conv2d(
      o_c1, ctx.params.ngf * 2, ks, 2, 0.02, "SAME", "c2")
    o_c3 = _conv2d(
      o_c2, ctx.params.ngf * 4, ks, 2, 0.02, "SAME", "c3")
    
    o_r1 = _resnet_block(o_c3, ctx.params.ngf * 4, "r1", padding)
    o_r2 = _resnet_block(o_r1, ctx.params.ngf * 4, "r2", padding)
    o_r3 = _resnet_block(o_r2, ctx.params.ngf * 4, "r3", padding)
    o_r4 = _resnet_block(o_r3, ctx.params.ngf * 4, "r4", padding)
    o_r5 = _resnet_block(o_r4, ctx.params.ngf * 4, "r5", padding)
    o_r6 = _resnet_block(o_r5, ctx.params.ngf * 4, "r6", padding)
    o_r7 = _resnet_block(o_r6, ctx.params.ngf * 4, "r7", padding)
    o_r8 = _resnet_block(o_r7, ctx.params.ngf * 4, "r8", padding)
    o_r9 = _resnet_block(o_r8, ctx.params.ngf * 4, "r9", padding)
    
    o_c4 = _deconv2d(
      o_r9, [ctx.params.batch_size, 128, 128, ctx.params.ngf * 2], ctx.params.ngf * 2, ks, ks, 2, 2, 0.02,
      "SAME", "c4")
    o_c5 = _deconv2d(
      o_c4, [ctx.params.batch_size, 256, 256, ctx.params.ngf], ctx.params.ngf, ks, ks, 2, 2, 0.02,
      "SAME", "c5")
    o_c6 = _conv2d(o_c5, 3, f,  1,
                                 0.02, "SAME", "c6",
                                 do_norm=False,
                                 do_relu=False)
    
    if skip is True:
      out_gen = tf.nn.tanh(input + o_c6, "t1")
    else:
      out_gen = tf.nn.tanh(o_c6, "t1")
    
    return out_gen


def discriminator(inputdisc, name="discriminator"):
  with tf.variable_scope(name):
    f = 4
  
    o_c1 = _conv2d(inputdisc, ctx.params.ndf, f, 2,
      0.02, "SAME", "c1", do_norm=False,
      relufactor=0.2)
    o_c2 = _conv2d(o_c1, ctx.params.ndf * 2, f, 2,
      0.02, "SAME", "c2", relufactor=0.2)
    o_c3 = _conv2d(o_c2, ctx.params.ndf * 4, f, 2,
      0.02, "SAME", "c3", relufactor=0.2)
    o_c4 = _conv2d(o_c3, ctx.params.ndf * 8, f, 1,
      0.02, "SAME", "c4", relufactor=0.2)
    o_c5 = _conv2d(
      o_c4, 1, f, 1, 0.02,
      "SAME", "c5", do_norm=False, do_relu=False
    )
  
    return o_c5

##################################################
######## 3.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  pass

###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  pass

###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
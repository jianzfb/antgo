# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : Hourglass_face_landmark.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
from antgo.dataflow.common import *
from antgo.context import *
import numpy as np
from antgo.codebook.tf.layers import *
import tensorflow.contrib.slim as slim
from antgo.trainer.tftrainer import *
from antgo.dataflow.imgaug.regular import *
from antgo.annotation.image_tool import *
##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2.step model building (tensorflow) ######
##################################################
class HourglassModel(ModelDesc):
  def __init__(self):
    super(HourglassModel, self).__init__()
    self._nstack = ctx.params.nstacks
    self._nfeat = ctx.params.nfeat
    self._nlow = ctx.params.nlow
    self._output_dim = ctx.params.output_dim # 16
    self._dropout_rate = ctx.params.dropout_rate
  
  def arg_scope(self, is_training):
    # branch side
    batch_norm_params = {
      'decay': 0.997,
      'epsilon': 1e-5,
      'scale': True,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      'is_training': is_training
    }

    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        activation_fn=None,
                        biases_initializer=None,
                        weights_initializer=slim.variance_scaling_initializer(),
                        ):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as scope:
            return scope

  def _residual(self, inputs, out_channels, name='residual_block'):
    with tf.variable_scope(name):
      # three convs
      norm_1 = slim.batch_norm(inputs, activation_fn=tf.nn.relu)
      conv_1 = slim.conv2d(norm_1, out_channels/2, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      norm_2 = slim.batch_norm(conv_1, activation_fn=tf.nn.relu)
      conv_2 =slim.conv2d(norm_2, out_channels/2, [3, 3], stride=1, normalizer_fn=None, activation_fn=None)
      
      norm_3 = slim.batch_norm(conv_2, activation_fn=tf.nn.relu)
      conv_3 = slim.conv2d(norm_3, out_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      # skip
      res_layer = None
      if inputs.get_shape().as_list()[3] == out_channels:
        res_layer = inputs
      else:
        res_layer = slim.conv2d(inputs, out_channels, [1, 1], stride=1, normalizer_fn=None, activation_fn=None)
      
      return tf.add(res_layer, conv_3)
  
  def _hourglass(self, inputs, n, num_out, name='hourglass'):
    with tf.variable_scope(name):
      # upper branch
      up_1 = self._residual(inputs, num_out, name='up_1')
      # lower branch
      low_ = slim.max_pool2d(inputs, [2, 2], stride=2)
      low_1 = self._residual(low_, num_out, name='low_1')
      
      if n > 0:
        low_2 = self._hourglass(low_1, n-1, num_out, name='low_2')
      else:
        low_2 = self._residual(low_1, num_out, name='low_2')
      
      low_3 = self._residual(low_2, num_out, name='low_3')
      up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name='upsampling')
      return tf.nn.relu(tf.add(up_2, up_1), name='out_hg')
  
  def model_fn(self, is_training=True, *args, **kwargs):
    # inputs: batch_size x 256 x 256 x 3
    # labels: batch_size x nstack x 64 x 64 x output_dim(landmark number)
    inputs = tf.placeholder(tf.float32, shape=(ctx.params.batch_size, 64, 64, 3), name='input_x')
    labels = None
    if is_training:
      labels = tf.placeholder(tf.float32,
        shape=(ctx.params.batch_size, ctx.params.nstacks, 32, 32, ctx.params.output_dim), name='label_y')

    with slim.arg_scope(self.arg_scope(is_training)):
      with tf.variable_scope('model'):
        with tf.variable_scope('preprocessing'):
          # inputs batch_size x 64 x 64 x 3
          pad1 = tf.pad(inputs, [[0, 0], [2, 2], [2, 2], [0, 0]], name='pad_1')
          # batch_size x 32 x 32 x 64
          conv_1 = slim.conv2d(pad1, self._nfeat, [6, 6], stride=2, scope='conv_64_to_32', padding='VALID')
          r3 = conv_1
          # r1 = self._residual(conv_1, 128, name='r1')
          # pool1 = slim.avg_pool2d(r1, kernel_size=[2,2],stride=2)
          # r2 = self._residual(pool1, int(self._nfeat/2), name='r2')
          # r3 = self._residual(r2, self._nfeat, name='r3')
        
        # storage table
        # root block
        hg = [None] * self._nstack
        ll = [None] * self._nstack
        ll_ = [None] * self._nstack
        drop = [None] * self._nstack
        out = [None] * self._nstack
        out_ = [None] * self._nstack
        sum_ = [None] * self._nstack
        with tf.variable_scope('stacks'):
          with tf.variable_scope('stage_0'):
            hg[0] = self._hourglass(r3, self._nlow, self._nfeat, 'hourglass')
            drop[0] = slim.dropout(hg[0], keep_prob=self._dropout_rate, is_training=is_training)
            ll[0] = slim.conv2d(drop[0], self._nfeat, [1, 1], stride=1, scope='cc')
            ll_[0] = slim.conv2d(ll[0], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='ll')
            
            out[0] = slim.conv2d(ll[0], self._output_dim, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
            out_[0] = slim.conv2d(out[0], self._nfeat, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='out_')
            sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
          
          for i in range(1, self._nstack - 1):
            with tf.variable_scope('stage_'+str(i)):
              hg[i] = self._hourglass(sum_[i-1], self._nlow, self._nfeat, 'hourglass')
              drop[i] = slim.dropout(hg[i], keep_prob=self._dropout_rate, is_training=is_training)
              ll[i] = slim.conv2d(drop[i], self._nfeat, [1, 1], stride=1, scope='cc')
              ll_[i] = slim.conv2d(ll[i], self._nfeat, [1,1], stride=1, normalizer_fn=None, activation_fn=None, scope='ll')
              
              out[i] = slim.conv2d(ll[i], self._output_dim, [1,1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
              out_[i] = slim.conv2d(out[i], self._nfeat, [1,1], stride=1, normalizer_fn=None, activation_fn=None, scope='out_')
              sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[i]], name='merge')
              
          with tf.variable_scope('stage_'+str(self._nstack - 1)):
            hg[self._nstack - 1] = self._hourglass(sum_[self._nstack - 2], self._nlow, self._nfeat, 'hourglass')
            drop[self._nstack - 1] = slim.dropout(hg[self._nstack - 1], keep_prob=self._dropout_rate, is_training=is_training)
            ll[self._nstack - 1] = slim.conv2d(drop[self._nstack - 1], self._nfeat, [1,1], stride=1, scope='cc')
            
            out[self._nstack - 1] = slim.conv2d(ll[self._nstack - 1], self._output_dim, [1,1], stride=1, normalizer_fn=None, activation_fn=None, scope='out')
            
          output = tf.stack(out, axis=1, name='final_output')

          if is_training:
            ss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output, labels=labels)
            loss = tf.reduce_mean(ss)
            tf.losses.add_loss(loss)
            return loss
          else:
            output = tf.nn.sigmoid(output[:, ctx.params.nstacks - 1])
            return output

##################################################
######## 3.step define training process  #########
##################################################
def _makeGaussian(height, width, sigma=3, center=None):
  """ Make a square gaussian kernel.
  size is the length of a side of the square
  sigma is full-width-half-maximum, which
  can be thought of as an effective radius.
  """
  x = np.arange(0, width, 1, float)
  y = np.arange(0, height, 1, float)[:, np.newaxis]
  if center is None:
    x0 = width // 2
    y0 = height // 2
  else:
    x0 = center[0]
    y0 = center[1]
  return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)


def _generate_hm(height, width, joints, maxlength, weight):
  num_joints = joints.shape[0]
  hm = np.zeros((height, width, num_joints), dtype=np.float32)
  for i in range(num_joints):
    if joints[i, 0] > 0 and joints[i, 1] > 0 and weight[i] == 1:
      s = int(np.sqrt(maxlength) * maxlength * 10 / 4096) + 2
      hm[:, :, i] = _makeGaussian(height, width, sigma=s, center=(joints[i, 0], joints[i, 1]))
    else:
      hm[:, :, i] = np.zeros((height, width))
  return hm


def _data_preprocess(*args, **kwargs):
  train_img = args[0][0]                # batch x 64 x 64 x 3
  train_img_annotations = args[0][1]    # [[], ..., []]
  batch_size, img_height, img_width = train_img.shape[0:3]
  
  train_img = train_img.astype(np.float32)
  train_gtmap = np.zeros((batch_size, ctx.params.nstacks, 32, 32, 5), np.float32)
  train_weights = np.zeros((batch_size, 5), np.float32)

  for i in range(batch_size):
    joints = train_img_annotations[i]['landmark']
    info = train_img_annotations[i]['info']
    old_height, old_width = info[0:2]
    joints = np.array(joints).reshape((-1, 2))
    joints[:, 0] = joints[:, 0] / float(old_width) * 32.0
    joints[:, 1] = joints[:, 1] / float(old_height) * 32.0
    train_weights[i] = 1
    
    hm = _generate_hm(32, 32, joints, 32, train_weights[i])
    hm = np.expand_dims(hm, axis=0)
    hm = np.repeat(hm, ctx.params.nstacks, axis=0)
    train_gtmap[i] = hm

  train_img = train_img / 255.0
  return train_img, train_gtmap, train_weights


def training_callback(data_source, dump_dir):
  ##########  1.step data preprocess ##############
  resized_data = Resize(Node.inputs(data_source), shape=(64, 64))
  batch_data = BatchData(Node.inputs(resized_data), ctx.params.batch_size)
  preprocess_node = Node('preprocess', _data_preprocess, Node.inputs(batch_data))

  ##########  2.step build model     ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.deploy(HourglassModel())

  iter = 0
  for epoch in range(ctx.params.max_epochs):
    generator = preprocess_node.iterator_value()
    while True:
      try:
        _, loss_val = tf_trainer.run(generator, binds={'input_x': 0, 'label_y': 1})
        iter = iter + 1
      except:
        break
    # save
    tf_trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
import cv2
def infer_callback(data_source, dump_dir):
  ##########  1.step data preprocess ##############
  resized_data = Resize(Node.inputs(data_source), shape=(ctx.params.input_size, ctx.params.input_size))
  def _div255(*args, **kwargs):
    return args[0][0] / 255.0
  normalize_data = Node('div255', _div255, Node.inputs(resized_data))
  batch_data = BatchData(Node.inputs(normalize_data), 1, True)

  ##########  2.step building model  ##############
  tf_trainer = TFTrainer(ctx.params, dump_dir, is_training=False)
  tf_trainer.deploy(HourglassModel())

  ##########  3.step predict sample  ##############
  generator = batch_data.iterator_value()
  while True:
    try:
      # predict landmark
      output = tf_trainer.run(generator, binds={'input_x': 0})
      output = output[0][0]
      
      # original data
      predict_img = data_source.get_value()
      height, width = predict_img.shape[0:2]

      predict_joints = -1 * np.ones((len(ctx.params.joints), 2))
      
      for joint_i in range(len(ctx.params.joints)):
        index = np.unravel_index(output[:, :, joint_i], (ctx.params.hm_size, ctx.params.hm_size))
        if output[index[0], index[1], joint_i] > ctx.params.thresh:
          # resize to original scale
          y = index[0]
          y = y / float(ctx.params.hm_size) * height
          x = index[1]
          x = x / float(ctx.params.hm_size) * width
          predict_joints[joint_i, 0] = y
          predict_joints[joint_i, 1] = x
        
          predict_img[int(y), int(x)] = 255
        
      # 4.step record predicted landmark
      ctx.recorder.record(predict_joints)
      
      # show
      cv2.imshow('ss1', predict_img.astype(np.uint8))
      cv2.waitKey()
    except:
      break


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
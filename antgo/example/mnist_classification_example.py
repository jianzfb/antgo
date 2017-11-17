# -*- coding: UTF-8 -*-
# Time: 10/11/17
# File: mnist_classification_example.py
# Author: Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

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
def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.
  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  with tf.name_scope('reshape'):
    x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  with tf.name_scope('pool1'):
    h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  with tf.name_scope('pool2'):
    h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


##################################################
######## 4.step define training process  #########
##################################################
def training_callback(data_source, dump_dir):
  ##########  1.step reorganized as batch ########
  batch = BatchData(Node.inputs(data_source), batch_size=50)

  ##########  2.step building model ##############
  # image x
  x = tf.placeholder(tf.float32, [None, 784])

  # label y
  y = tf.placeholder(tf.int32, [None])
  y_ = tf.one_hot(y, 10)
  y_ = tf.cast(y_, tf.float32)

  y_conv, keep_prob = deepnn(x)

  with tf.name_scope('loss'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
  cross_entropy = tf.reduce_mean(cross_entropy)

  with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

  with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)
  
  # saver
  saver = tf.train.Saver(max_to_keep=2)
  
  ##########  3.step start training ##############
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iter = 0
    for epoch in range(200):
      for data in batch.iterator_value():
        image, label = data
        image = np.reshape(image, (50, 784))
        _, loss = sess.run([train_step, cross_entropy], feed_dict={x: image, y: np.array(label), keep_prob: 0.5})
        # record loss
        loss_channel.send(iter, loss)
        
        if iter % 100 == 0:
          # computing classification accuracy
          train_accuracy = accuracy.eval(feed_dict={
            x: image, y: np.array(label), keep_prob: 1.0})
          
          # record accuracy
          accuracy_channel.send(iter, train_accuracy)
        
        iter += 1
      
      # save model after every epoch
      model_filename = "{prefix}_{infix}_{d}.ckpt".format(prefix='mnist',infix=epoch, d=iter)
      saver.save(sess, os.path.join(dump_dir, model_filename))

  print('stop training process')


###################################################
######## 5.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  ##########  1.step reorganized as batch ########
  batch = BatchData(Node.inputs(data_source), batch_size=1)
  
  ##########  2.step building model ##############
  # image x
  x = tf.placeholder(tf.float32, [1, 784])
  #
  y, keep_prob = deepnn(x)

  saver = tf.train.Saver()
  
  ##########  3.step start training ##############
  with tf.Session() as sess:
    # initialize all variables
    sess.run(tf.global_variables_initializer())
    # load preprained model
    saver.restore(sess, os.path.join('/home/mi/PycharmProjects/antgo/antgo/dump/20171116.192853.327571/train', 'mnist_0_1200.ckpt'))
    
    for image in batch.iterator_value():
      #
      image = np.reshape(image[0], (1, 784))
      y_val = sess.run(y, {x: image, keep_prob: 1.0})
      c = np.argmax(y_val)
      ctx.recorder.record(c)
    
  print('stop challenge process')


###################################################
####### 6.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
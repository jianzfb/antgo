# -*- coding: UTF-8 -*-
# @Time : 2018/8/1
# @File : fcn_seg.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.common import *
from antgo.context import *
from antgo.dataflow.dataset import *
from antgo.measures import *
from antgo.codebook.tf.preprocess import *
from antgo.trainer.tftrainer import *
from antgo.utils._resize import *
import tensorflow as tf
import tensorflow.contrib.slim as slim

##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()

##################################################
###### 1.1.step build visualization chart  #######
##################################################
# chart 1 (loss record)
loss_fcn_8s_channel = ctx.job.create_channel('loss_fcn_8s', 'NUMERIC')
ctx.job.create_chart([loss_fcn_8s_channel], 'FCN LOSS', 'step', 'value')


##################################################
######## 2.step custom model building code  ######
##################################################
class FCNModel(ModelDesc):
  def __init__(self):
    super(FCNModel, self).__init__('FCN')

  def model_fn(self, is_training=True, *args, **kwargs):
    batch_image = tf.placeholder(tf.float32, (ctx.params.batch_size, 512, 512, 3), name='image')
    batch_label = None
    if is_training:
      batch_label = tf.placeholder(tf.int32, (ctx.params.batch_size, 512, 512), name='label')

    # preprocess
    rgb_channels = tf.split(batch_image, 3,3)
    rgb_channels[0] = rgb_channels[0] - 128.0
    rgb_channels[1] = rgb_channels[1] - 128.0
    rgb_channels[2] = rgb_channels[2] - 128.0
    batch_image = tf.concat(rgb_channels, -1)

    # vgg 16
    layers = (
      'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
      'relu3_3', 'pool3',
      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
      'relu4_3', 'pool4',
      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
      'relu5_3', 'pool5',
    )

    net = batch_image
    net_collection = {}
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(0.0001),
                        normalizer_fn=None,
                        activation_fn=None,
                        weights_initializer=slim.variance_scaling_initializer()):
      for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
          block_i = int(name[4])
          output_channels = block_i*64 if block_i < 5 else 512
          net = slim.conv2d(net,output_channels,[3,3], stride=[1,1], padding='SAME')
          net_collection[name] = net
        elif kind == 'relu':
          net = tf.nn.relu(net)
          net_collection[name] = net
        elif kind == 'pool':
          net = slim.avg_pool2d(net,2,stride=2,padding='SAME')
          net_collection[name] = net

      pool5_output = net_collection['pool5']

      # fully conv
      conv6 = slim.conv2d(pool5_output, 4096, [7,7], stride=[1,1], padding='SAME')
      relu6 = tf.nn.relu(conv6)
      relu6 = slim.dropout(relu6, 0.5)

      conv7 = slim.conv2d(relu6, 4096, [1,1], stride=[1,1], padding='SAME')
      relu7 = tf.nn.relu(conv7)
      relu7 = slim.dropout(relu7)

      # FCN32S
      score_32 = slim.conv2d(relu7, ctx.params.class_num, [1,1], stride=[1,1], padding='SAME')
      score_32_up = slim.convolution2d_transpose(score_32, ctx.params.class_num, [4, 4], [2, 2])

      # FCN16S
      pool4_output = slim.conv2d(net_collection['pool4'], ctx.params.class_num, [1,1], stride=[1,1], padding='SAME')
      score_16 = score_32_up + pool4_output
      score_16_up = slim.convolution2d_transpose(score_16, ctx.params.class_num, [4, 4], [2, 2])

      # FCN8S
      pool3_output = slim.conv2d(net_collection['pool3'], ctx.params.class_num, [1,1], stride=[1,1], padding='SAME')
      score_8 = score_16_up + pool3_output
      score_8_up = slim.convolution2d_transpose(score_8, ctx.params.class_num, [4, 4], [2, 2])

      if is_training:
        one_hot_batch_label = tf.one_hot(batch_label, ctx.params.class_num)
        one_hot_batch_label = tf.image.resize_bilinear(one_hot_batch_label, [128, 128])

        # cross entropy
        fcn8_loss = tf.losses.softmax_cross_entropy(one_hot_batch_label, score_8_up)
        return fcn8_loss
      else:
        logits = tf.nn.softmax(score_8_up)
        return logits


##################################################
######## 3.step define training process  #########
##################################################
def _preprocess_data(*args, **kwargs):
  data = args[0][0]
  label = args[0][1]

  input_data = resize(data, [ctx.params.input_height, ctx.params.input_width])
  input_label = resize(label, [ctx.params.input_height, ctx.params.input_width])
  input_label = input_label.astype(np.int32)
  return input_data, input_label


def training_callback(data_source, dump_dir):
  # 1.step build CNN model
  fcn_model = FCNModel()
  tf_trainer = TFTrainer(ctx, dump_dir, is_training=True)
  tf_trainer.deploy(fcn_model)

  # 2.step build data source pipeline
  preprocess_node = Node("preprocess", _preprocess_data, Node.inputs(data_source))
  batch_data_source = BatchData(Node.inputs(preprocess_node), ctx.params.batch_size)

  # 3.step connect data source to CNN
  for epoch in range(ctx.params.max_epochs):
    batch_data_generator = batch_data_source.iterator_value()
    while True:
      try:
        _, fcn8_loss_val = tf_trainer.run(batch_data_generator, binds={'image': 0, 'label': 1})

        if tf_trainer.iter_at % 50 == 0:
          loss_fcn_8s_channel.send(tf_trainer.iter_at, fcn8_loss_val)
      except:
        break

    tf_trainer.snapshot(epoch)


###################################################
######## 4.step define infer process     ##########
###################################################
def infer_callback(data_source, dump_dir):
  # 1.step build CNN model
  fcn_model = FCNModel()
  tf_trainer = TFTrainer(ctx, dump_dir, is_training=False)
  tf_trainer.deploy(fcn_model)

  # 2.step infer
  for data in data_source.iterator_value():
    # execute infer process
    # logits = ...
    preprocess_data = data
    if len(data.shape) == 2:
      preprocess_data=np.concatenate((np.expand_dims(data, -1),
                                      np.expand_dims(data, -1),
                                      np.expand_dims(data, -1)), -1)
    else:
      preprocess_data = data[:,:,:3]

    height, width, _ = preprocess_data.shape
    preprocess_data = resize(preprocess_data, [ctx.params.input_height, ctx.params.input_width])
    preprocess_data = np.expand_dims(preprocess_data, 0)
    logits, = tf_trainer.run_dict({'image': preprocess_data})
    logits = logits[0]

    resized_logits = resize(logits, [height, width])
    image_label = np.argmax(resized_logits, -1)
    image_label = np.squeeze(image_label)
    # record
    ctx.recorder.record({'RESULT': image_label})


###################################################
####### 5.step link training and infer ############
#######        process to context      ############
###################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
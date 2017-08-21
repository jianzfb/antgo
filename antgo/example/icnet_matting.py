# -*- coding: UTF-8 -*-
# @Time    : 17-8-17
# @File    : icnet_matting.py
# @Author  :
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.trainer.tftrainer import *
from antgo.trainer.trainer import *
from antgo.dataflow.common import *
import tensorflow as tf
import collections
from functools import reduce
from operator import mul
slim = tf.contrib.slim


##################################################
######## 1.step global interaction handle ########
##################################################
ctx = Context()


##################################################
######## 2.step custom network framework #########
##################################################
@slim.add_arg_scope
def root_block(inputs, scope=None):
  with tf.variable_scope(scope, 'root', [inputs]) as sc:
    conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, scope='conv1',padding='SAME')
    conv2 = slim.conv2d(conv1,  32, [3, 3], stride=1, scope='conv2',padding='SAME')
    conv3 = slim.conv2d(conv2, 64, [3, 3], stride=1, scope='conv3',padding='SAME')
    pool1 = slim.max_pool2d(conv3, [3, 3], stride=2, scope='pool1',padding='SAME')

  return pool1


def subsample(inputs, factor, scope=None):
  if factor == 1:
    return inputs
  else:
    return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)


def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
  if stride == 1:
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate,
                       padding='SAME', scope=scope)
  else:
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    inputs = tf.pad(inputs,
                    [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride,
                       rate=rate, padding='VALID', scope=scope)


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):

  with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
    depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
    if depth == depth_in:
      shortcut = subsample(inputs, reduce(mul, stride), 'shortcut')
    else:
      shortcut = slim.conv2d(inputs, depth, [1, 1], stride=reduce(mul, stride),
                             activation_fn=None, scope='shortcut')

    residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=stride[0], scope='conv1')
    residual = conv2d_same(residual, depth_bottleneck, 3, stride[1], rate=rate, scope='conv2')
    residual = slim.conv2d(residual, depth, [1, 1], stride=stride[2],
                           activation_fn=None, scope='conv3')

    output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


@slim.add_arg_scope
def pyramid_pooling(inputs, pool_size, depth,
                    outputs_collections=None, scope=None):
  with tf.variable_scope(scope, 'pyramid_pool_v1', [inputs]) as sc:
    dims = inputs.get_shape().dims
    out_height, out_width = dims[1].value, dims[2].value

    pool1 = slim.avg_pool2d(inputs, pool_size, stride=pool_size, scope='pool1')
    conv1 = slim.conv2d(pool1, depth, [1, 1], stride=1, scope='conv1')
    output = tf.image.resize_bilinear(conv1, [out_height, out_width])

    return slim.utils.collect_named_outputs(outputs_collections,
                                            sc.original_name_scope,
                                            output)


@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, outputs_collections=None):
  block_id = 0
  net_branch = None
  for block in blocks:
    with tf.variable_scope(block.scope, 'block', [net]) as sc:
      for i, unit in enumerate(block.args):

        with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
          unit_depth, unit_depth_bottleneck, unit_stride, unit_rate = unit

          net = block.unit_fn(net,
                              depth=unit_depth,
                              depth_bottleneck=unit_depth_bottleneck,
                              stride=unit_stride,
                              rate=unit_rate)

          if block_id == 1 and net_branch == None:
            net_branch = net
            dims = net.get_shape().dims
            out_height, out_width = dims[1].value, dims[2].value
            net = tf.image.resize_bilinear(net, [int(out_height/2), int(out_width/2)])

      net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

    block_id += 1
  return net, net_branch


@slim.add_arg_scope
def pyramid_pooling_module(inputs, levels, outputs_collections=None):
  with tf.variable_scope('pyramid_pool_module', [inputs]) as sc:
    level_maps = [inputs]
    for level in reversed(levels):
      with tf.variable_scope(level.scope, 'level', [inputs]) as sc:
        level_size, level_depth = level.args
        level_map = level.fn(inputs, level_size, level_depth)
        level_maps.append(level_map)

    # net = tf.add_n(level_maps)
    net = tf.concat(level_maps, axis=3)
    net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)

  return net


@slim.add_arg_scope
def icnet_matting(inputs, blocks, levels, num_classes=None, is_training=True, scope=None):
  direct_endpoints = {}
  with tf.variable_scope(scope, 'icnet_v1', [inputs]) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d,
                         bottleneck,
                         pyramid_pooling,
                         stack_blocks_dense,
                         pyramid_pooling_module], outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], is_training=is_training):
        branch_123_data = inputs

        # branch 23 (low-resolution and mid-resolution)
        dims = branch_123_data.get_shape().dims
        branch_23_data = tf.image.resize_bilinear(branch_123_data, [int(dims[1].value / 2), int(dims[2].value / 2)])

        branch_23_net = root_block(branch_23_data)
        branch_3_net, branch_2_net = stack_blocks_dense(branch_23_net, blocks, None)

        # # branch 3
        branch_3_net = pyramid_pooling_module(branch_3_net, levels)
        branch_3_net = slim.conv2d(branch_3_net, 256, [1, 1], stride=1, padding='SAME')
        ## branch 3 dropout
        branch_3_net = slim.dropout(branch_3_net, keep_prob=0.9, is_training=is_training)
        ##
        dims = branch_3_net.get_shape().dims
        out_height, out_width = dims[1].value, dims[2].value
        branch_3_net = tf.image.resize_bilinear(branch_3_net,[out_height*2,out_width*2])

        # 1/16 output
        direct_endpoints['branch_3_classifier_guid'] = \
            slim.conv2d(branch_3_net,num_classes,[1,1],padding='SAME',activation_fn=None)

        # branch 2 + branch 3
        branch_3_net = slim.conv2d(branch_3_net, 128, [3, 3], stride=1, rate=2, padding='SAME',activation_fn=None)
        branch_23_net = branch_3_net + slim.conv2d(branch_2_net,128,[1,1],padding='SAME',activation_fn=None)
        branch_23_net = tf.nn.relu(branch_23_net)
        dims = branch_23_net.get_shape().dims
        out_height, out_width = dims[1].value, dims[2].value
        branch_23_net = tf.image.resize_bilinear(branch_23_net,[out_height*2,out_width*2])

        # 1/8 output
        direct_endpoints['branch_23_classifier_guid'] = \
            slim.conv2d(branch_23_net,num_classes,[1,1],padding='SAME',activation_fn=None)

        # branch 1
        branch_23_net = slim.conv2d(branch_23_net, 128, [3, 3], stride=1, rate=2, padding='SAME', activation_fn=None)
        branch_1_conv1 = slim.conv2d(inputs, 32, [3, 3], stride=2, padding='SAME')
        branch_1_conv2 = slim.conv2d(branch_1_conv1, 32, [3, 3], stride=2, padding='SAME')
        branch_1_conv3 = slim.conv2d(branch_1_conv2, 64, [3, 3], stride=2, padding='SAME')
        branch_1_proj = slim.conv2d(branch_1_conv3, 128, [1, 1], stride=1, activation_fn=None)

        branch_123_net = branch_1_proj + branch_23_net
        branch_123_net = tf.nn.relu(branch_123_net)
        dims = branch_123_net.get_shape().dims
        out_height, out_width = dims[1].value, dims[2].value
        branch_123_net = tf.image.resize_bilinear(branch_123_net, [out_height*2,out_width*2])

        # 1/4 output
        direct_endpoints['branch_123_2_classifier_guid'] = \
          slim.conv2d(branch_123_net, num_classes, [1, 1], padding='SAME', activation_fn=None)

        # only for test stage
        branch_123_net = slim.conv2d(branch_123_net, num_classes, [1, 1], stride=1, padding='SAME', activation_fn=None)
        branch_123_net = tf.image.resize_bilinear(branch_123_net, [out_height*8, out_width*8])

        # 1 output (mask)
        direct_endpoints['branch_123_classifier_guid'] = branch_123_net

        return direct_endpoints['branch_123_classifier_guid'], \
               direct_endpoints['branch_123_2_classifier_guid'],\
               direct_endpoints['branch_23_classifier_guid'], \
               direct_endpoints['branch_3_classifier_guid']


class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
  """
  """


class Level(collections.namedtuple('Level', ['scope', 'fn', 'args'])):
  """
  """


class icnet(ModelDesc):
  def __init__(self):
    super(icnet, self).__init__('icnet')

  def model_fn(self, is_training, *args, **kwargs):
    # prepare inputs
    images = tf.placeholder(shape=(self.batch_size, None, None, 3), dtype=tf.uint8, name='icnet-image')
    images = tf.image.resize_bilinear(images, size=[480, 480])

    # build model
    blocks = [
        Block('block1', bottleneck, [(128, 32, (1, 1, 1), 1)] * 3),
        Block('block2', bottleneck, [(256, 64, (2, 1, 1), 1)] + [(256, 64, (1, 1, 1), 1)] * 3),
        Block('block3', bottleneck, [(512, 128, (1, 1, 1), 2)] * 6),
        Block('block4', bottleneck, [(1024, 256, (1, 1, 1), 4)] * 3)
    ]

    levels = [Level('level1', pyramid_pooling, ((15, 15), 256)),
              Level('level2', pyramid_pooling, ((7, 7), 256)),
              Level('level3', pyramid_pooling, ((4, 4), 256)),
              Level('level4', pyramid_pooling, ((2, 2), 256)),
    ]

    branch_1, branch_1_4, branch_1_8, branch_1_16 = \
      icnet_matting(images, blocks, levels, self.num_classes, is_training, scope=self.name)

    # loss
    if is_training:
      labels = tf.placeholder(shape=(self.batch_size, None, None), dtype=tf.uint8, name='icnet-label')
      labels = tf.expand_dims(labels, 3)
      labels = tf.image.resize_nearest_neighbor(labels, size=[480, 480])
      labels = tf.squeeze(labels, axis=3)
      labels = tf.to_int32(labels / 255.0)
      labels = slim.one_hot_encoding(labels, self.num_classes)

      # segmentation task
      # 1/4
      dims = labels.get_shape().dims
      labels_1_4 = tf.image.resize_nearest_neighbor(labels, [int(dims[1].value / 4), int(dims[2].value / 4)])
      tf.losses.softmax_cross_entropy(labels_1_4, branch_1_4, scope='branch_1_4_loss')
      # 1/8
      labels_1_8 = tf.image.resize_nearest_neighbor(labels, [int(dims[1].value / 8), int(dims[2].value / 8)])
      tf.losses.softmax_cross_entropy(labels_1_8, branch_1_8, weights=0.6, scope='branch_1_8_loss')

      # 1/16
      labels_1_16 = tf.image.resize_nearest_neighbor(labels, [int(dims[1].value / 16), int(dims[2].value / 16)])
      tf.losses.softmax_cross_entropy(labels_1_16, branch_1_16, weights=0.4, scope='branch_1_16_loss')
    else:
      predictions = tf.squeeze(tf.argmax(branch_1, 3))
      return predictions

  def arg_scope_fn(self):
    weight_decay = 0.0001
    batch_norm_decay = 0.997
    batch_norm_epsilon = 1e-5
    batch_norm_scale = True

    batch_norm_params = {
      'decay': batch_norm_decay,
      'epsilon': batch_norm_epsilon,
      'scale': batch_norm_scale,
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
        weights_regularizer=slim.l2_regularizer(weight_decay),
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=tf.nn.relu,
        normalizer_fn=slim.batch_norm,
        normalizer_params=batch_norm_params):
      with slim.arg_scope([slim.batch_norm], **batch_norm_params):
        with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
          return arg_sc


##################################################
##### 3.step define training process #############
##################################################
def training_callback(data_source, dump_dir):
  # 1.step deploy model
  tf_trainer = TFTrainer(ctx.params, dump_dir)
  tf_trainer.deploy(icnet())
  
  # 2.step reorganize as batch
  batch_data = BatchData(Node.inputs(data_source), tf_trainer.batch_size)

  # 3.step start running
  for epoch in range(tf_trainer.max_epochs):
    data_generator = batch_data.iterator_value()
    while True:
      try:
        res = tf_trainer.run(data_generator, {'icnet-image': 0, 'icnet-label': 1})
      except StopIteration:
        break

    tf_trainer.snapshot(epoch)


##################################################
###### 4.step define infer process ###############
##################################################
def infer_callback(data_source, dump_dir):
  # 1.step deploy model
  tf_trainer = TFTrainer(ctx.params, dump_dir, False)
  tf_trainer.deploy(icnet())
  
  # 2.step reorganize as batch
  batch_data = BatchData(Node.inputs(data_source), tf_trainer.batch_size)

  # 3.step start running
  data_generator = batch_data.iterator_value()
  while True:
    try:
      mask = tf_trainer.run(data_generator, {'icnet-image': 0})
      ctx.recorder.record(mask)
    except StopIteration:
      break


##################################################
####### 5.step link training and infer ###########
#######        process to context      ###########
##################################################
ctx.training_process = training_callback
ctx.infer_process = infer_callback
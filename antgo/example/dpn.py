# -*- coding: UTF-8 -*-
# @Time    : 17-11-22
# @File    : dpn.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tensorflow as tf
import tensorflow.contrib.slim as slim


def _bn_relu_conv_block(input,
                        filters,
                        kernel=(3, 3),
                        stride=(1, 1),
                        weight_decay=5e-4):
  ''' Adds a Batchnorm-Relu-Conv block for DPN
  Args:
      input: input tensor
      filters: number of output filters
      kernel: convolution kernel size
      stride: stride of convolution
  Returns: a keras tensor
  '''
  channel_axis = -1
  
  x = slim.conv2d(input, filters, kernel, padding='SAME', stride=stride,
                  weights_regularizer=slim.l2_regularizer(weight_decay),
                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                  biases_initializer=None)

  x = slim.batch_norm(x)
  x = tf.nn.relu(x)
  return x


def _grouped_convolution_block(input, grouped_channels, cardinality, strides, weight_decay=5e-4):
  ''' Adds a grouped convolution block. It is an equivalent block from the paper
  Args:
      input: input tensor
      grouped_channels: grouped number of filters
      cardinality: cardinality factor describing the number of groups
      strides: performs strided convolution for downscaling if > 1
      weight_decay: weight decay term
  Returns: a keras tensor
  '''
  channel_axis = -1

  group_list = []

  if cardinality == 1:
    # with cardinality 1, it is a standard convolution
    x = slim.conv2d(input,
                    grouped_channels,
                    (3, 3),
                    padding='SAME',
                    stride=strides,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=None)
    x = slim.batch_norm(x)
    x = tf.nn.relu(x)
    return x

  input_slices = tf.split(input, cardinality, axis=3)
  for c in range(cardinality):
    x = slim.conv2d(input_slices[c],
                    grouped_channels,
                    (3, 3),
                    padding='SAME',
                    stride=strides,
                    weights_regularizer=slim.l2_regularizer(weight_decay),
                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                    biases_initializer=None)
    group_list.append(x)
  
  group_merge = tf.concat(group_list, axis=3)
  group_merge = slim.batch_norm(group_merge)
  group_merge = tf.nn.relu(group_merge)
  return group_merge


def _dual_path_block(input,
                     pointwise_filters_a,
                     grouped_conv_filters_b,
                     pointwise_filters_c,
                     filter_increment,
                     cardinality,
                     block_type='normal'):
  '''
  Creates a Dual Path Block. The first path is a ResNeXt type
  grouped convolution block. The second is a DenseNet type dense
  convolution block.
  Args:
      input: input tensor
      pointwise_filters_a: number of filters for the bottleneck
          pointwise convolution
      grouped_conv_filters_b: number of filters for the grouped
          convolution block
      pointwise_filters_c: number of filters for the bottleneck
          convolution block
      filter_increment: number of filters that will be added
      cardinality: cardinality factor
      block_type: determines what action the block will perform
          - `projection`: adds a projection connection
          - `downsample`: downsamples the spatial resolution
          - `normal`    : simple adds a dual path connection
  Returns: a list of two output tensors - one path of ResNeXt
      and another path for DenseNet
  '''
  grouped_channels = int(grouped_conv_filters_b / cardinality)
  
  init = tf.concat(input, axis=3) if isinstance(input, list) else input
  
  if block_type == 'projection':
    stride = (1, 1)
    projection = True
  elif block_type == 'downsample':
    stride = (2, 2)
    projection = True
  elif block_type == 'normal':
    stride = (1, 1)
    projection = False
  else:
    raise ValueError('`block_type` must be one of ["projection", "downsample", "normal"]. Given %s' % block_type)
  
  if projection:
    projection_path = _bn_relu_conv_block(init,
                                          filters=pointwise_filters_c + 2 * filter_increment,
                                          kernel=(1, 1),
                                          stride=stride)
    ss = tf.split(projection_path, pointwise_filters_c + 2 * filter_increment, axis=3)
    input_residual_path = tf.concat(ss[:pointwise_filters_c], axis=3)
    input_dense_path = tf.concat(ss[pointwise_filters_c:], axis=3)
  else:
    input_residual_path = input[0]
    input_dense_path = input[1]
  
  x = _bn_relu_conv_block(init, filters=pointwise_filters_a, kernel=(1, 1))
  x = _grouped_convolution_block(x, grouped_channels=grouped_channels, cardinality=cardinality, strides=stride)
  x = _bn_relu_conv_block(x, filters=pointwise_filters_c + filter_increment, kernel=(1, 1))
  
  mm = tf.split(x, pointwise_filters_c + filter_increment, axis=3)
  output_residual_path = tf.concat(mm[: pointwise_filters_c], axis=3)
  output_dense_path = tf.concat(mm[pointwise_filters_c:], axis=3)
  
  residual_path = tf.add_n([input_residual_path, output_residual_path])
  dense_path = tf.concat([input_dense_path, output_dense_path], axis=3)
  
  return [residual_path, dense_path]


def _initial_conv_block_inception(input, initial_conv_filters, weight_decay=5e-4):
  ''' Adds an initial conv block, with batch norm and relu for the DPN
  Args:
      input: input tensor
      initial_conv_filters: number of filters for initial conv block
      weight_decay: weight decay factor
  Returns: a keras tensor
  '''
  channel_axis = -1
  
  x = slim.conv2d(input, initial_conv_filters, (3, 3),
                  padding='SAME',
                  stride=(1, 1),
                  weights_regularizer=slim.l2_regularizer(weight_decay),
                  weights_initializer=tf.contrib.layers.xavier_initializer(),
                  biases_initializer=None)
  x = slim.batch_norm(x)
  x = tf.nn.relu(x)
  # x = slim.max_pool2d(x, (3, 3), stride=(2, 2), padding='SAME')
  
  return x

def _create_dpn(nb_classes,
                img_input,
                initial_conv_filters,
                filter_increment,
                depth,
                cardinality=32,
                width=3,
                weight_decay=5e-4,
                pooling=None):
  ''' Creates a ResNeXt model with specified parameters
  Args:
      initial_conv_filters: number of features for the initial convolution
      include_top: Flag to include the last dense layer
      initial_conv_filters: number of features for the initial convolution
      filter_increment: number of filters incremented per block, defined as a list.
          DPN-92  = [16, 32, 24, 128]
          DON-98  = [16, 32, 32, 128]
          DPN-131 = [16, 32, 32, 128]
          DPN-107 = [20, 64, 64, 128]
      depth: number or layers in the each block, defined as a list.
          DPN-92  = [3, 4, 20, 3]
          DPN-98  = [3, 6, 20, 3]
          DPN-131 = [4, 8, 28, 3]
          DPN-107 = [4, 8, 20, 3]
      width: width multiplier for network
      weight_decay: weight_decay (l2 norm)
      pooling: Optional pooling mode for feature extraction
          when `include_top` is `False`.include_top
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
          - `max-avg` means that both global average and global max
              pooling will be applied to the output of the last
              convolution layer
  Returns: a Keras Model
  '''
  channel_axis = -1
  N = list(depth)
  base_filters = 256
  
  with tf.variable_scope(None, 'dpn', [img_input]):
    # block 1 (initial conv block)
    x = _initial_conv_block_inception(img_input, initial_conv_filters, weight_decay)
  
    # block 2 (projection block)
    filter_inc = filter_increment[0]
    filters = int(cardinality * width)
    
    x = _dual_path_block(x,
                         pointwise_filters_a=filters,
                         grouped_conv_filters_b=filters,
                         pointwise_filters_c=base_filters,
                         filter_increment=filter_inc,
                         cardinality=cardinality,
                         block_type='projection')
  
    for i in range(N[0] - 1):
      x = _dual_path_block(x, pointwise_filters_a=filters,
                           grouped_conv_filters_b=filters,
                           pointwise_filters_c=base_filters,
                           filter_increment=filter_inc,
                           cardinality=cardinality,
                           block_type='normal')
  
    # remaining blocks
    for k in range(1, len(N)):
      print("BLOCK %d" % (k + 1))
      filter_inc = filter_increment[k]
      filters *= 2
      base_filters *= 2
    
      x = _dual_path_block(x, pointwise_filters_a=filters,
                           grouped_conv_filters_b=filters,
                           pointwise_filters_c=base_filters,
                           filter_increment=filter_inc,
                           cardinality=cardinality,
                           block_type='downsample')
    
      for i in range(N[k] - 1):
        x = _dual_path_block(x, pointwise_filters_a=filters,
                             grouped_conv_filters_b=filters,
                             pointwise_filters_c=base_filters,
                             filter_increment=filter_inc,
                             cardinality=cardinality,
                             block_type='normal')
  
    x = tf.concat(x, axis=channel_axis)
    if pooling == 'avg':
      # Global average pooling.
      x = tf.reduce_mean(x, [1, 2], keep_dims=True)
      x = slim.conv2d(x, nb_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
      x = tf.squeeze(x, [1, 2])
    elif pooling == 'max':
      x = tf.reduce_max(x, [1, 2], keep_dims=True)
      x = slim.conv2d(x, nb_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
      x = tf.squeeze(x, [1,2])
    elif pooling == 'max-avg':
      a = tf.reduce_max(x, [1, 2], keep_dims=True)
      b = tf.reduce_mean(x, [1, 2], keep_dims=True)
      x = tf.add_n([a, b])
      x = x * 0.5
      x = slim.conv2d(x, nb_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
      x = tf.squeeze(x, [1,2])

    return x


def DPN(initial_conv_filters=64,
        depth=[3, 4, 20, 3],
        filter_increment=[16, 32, 24, 128],
        cardinality=32,
        width=3,
        weight_decay=5e-4,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=1000):
  """ Instantiate the Dual Path Network architecture for the ImageNet dataset. Note that ,
      when using TensorFlow for best performance you should set
      `image_data_format="channels_last"` in your Keras config
      at ~/.keras/keras.json.
      The model are compatible with both
      TensorFlow and Theano. The dimension ordering
      convention used by the model is the one
      specified in your Keras config file.
      # Arguments
          initial_conv_filters: number of features for the initial convolution
          depth: number or layers in the each block, defined as a list.
              DPN-92  = [3, 4, 20, 3]
              DPN-98  = [3, 6, 20, 3]
              DPN-131 = [4, 8, 28, 3]
              DPN-107 = [4, 8, 20, 3]
          filter_increment: number of filters incremented per block, defined as a list.
              DPN-92  = [16, 32, 24, 128]
              DON-98  = [16, 32, 32, 128]
              DPN-131 = [16, 32, 32, 128]
              DPN-107 = [20, 64, 64, 128]
          cardinality: the size of the set of transformations
          width: width multiplier for the network
          weight_decay: weight decay (l2 norm)
          include_top: whether to include the fully-connected
              layer at the top of the network.
          weights: `None` (random initialization) or `imagenet` (trained
              on ImageNet)
          input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
              to use as image input for the model.
          input_shape: optional shape tuple, only to be specified
              if `include_top` is False (otherwise the input shape
              has to be `(224, 224, 3)` (with `tf` dim ordering)
              or `(3, 224, 224)` (with `th` dim ordering).
              It should have exactly 3 inputs channels,
              and width and height should be no smaller than 8.
              E.g. `(200, 200, 3)` would be one valid value.
          pooling: Optional pooling mode for feature extraction
              when `include_top` is `False`.
              - `None` means that the output of the model will be
                  the 4D tensor output of the
                  last convolutional layer.
              - `avg` means that global average pooling
                  will be applied to the output of the
                  last convolutional layer, and thus
                  the output of the model will be a 2D tensor.
              - `max` means that global max pooling will
                  be applied.
              - `max-avg` means that both global average and global max
                  pooling will be applied to the output of the last
                  convolution layer
          classes: optional number of classes to classify images
              into, only to be specified if `include_top` is True, and
              if no `weights` argument is specified.
      # Returns
          A Keras model instance.
      """

  x = _create_dpn(classes,
                  input_tensor,
                  initial_conv_filters,
                  filter_increment,
                  depth,
                  cardinality,
                  width,
                  weight_decay,
                  pooling)
  return x


def DPN92(input_tensor,
          include_top=True,
          weights=None,
          pooling=None,
          classes=1000):
  return DPN(include_top=include_top, weights=weights, input_tensor=input_tensor,
    pooling=pooling, classes=classes)


def DPN98(input_tensor,
          include_top=True,
          weights=5e-4,
          pooling=None,
          classes=1000):
  return DPN(initial_conv_filters=96,
             depth=[3, 6, 20, 3],
             filter_increment=[16, 32, 32, 128],
             cardinality=40,
             width=4,
             include_top=include_top,
             weights=weights,
             input_tensor=input_tensor,
             pooling=pooling,
             classes=classes)


def DPN137(input_tensor,
           include_top=True,
           weights=None,
           pooling=None,
           classes=1000):
  return DPN(initial_conv_filters=128, depth=[4, 8, 28, 3], filter_increment=[16, 32, 32, 128],
    cardinality=40, width=4, include_top=include_top, weights=weights, input_tensor=input_tensor,
    pooling=pooling, classes=classes)


def DPN107(input_tensor=None,
           include_top=True,
           weights=None,
           pooling=None,
           classes=1000):
  return DPN(initial_conv_filters=128, depth=[4, 8, 20, 3], filter_increment=[20, 64, 64, 128],
    cardinality=50, width=4, include_top=include_top, weights=weights, input_tensor=input_tensor,
    pooling=pooling, classes=classes)


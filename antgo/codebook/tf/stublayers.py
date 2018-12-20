# -*- coding: UTF-8 -*-
# @Time    : 2018/11/21 5:14 PM
# @File    : stublayers.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.stublayer import *
from antgo.automl.basestublayers import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
import numpy as np


class StubDense(BaseStubDense):
  def __init__(self, obj):
    super(StubDense, self).__init__(obj.input_units, obj.units, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.fully_connected,num_outputs=self.units)(*args, **kwargs)


class StubConv2d(BaseStubConv2d):
  def __init__(self, obj):
    super(StubConv2d, self).__init__(obj.input_channel,
                                     obj.filters,
                                     obj.kernel_size_h,
                                     obj.kernel_size_w,
                                     obj.rate_h,
                                     obj.rate_w,
                                     obj.stride,
                                     obj.input,
                                     obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.conv2d,
                             num_outputs=self.filters,
                             kernel_size=[self.kernel_size_h,
                                          self.kernel_size_w],
                             rate=[self.rate_h, self.rate_w],
                             stride=self.stride)(*args, **kwargs)


class StubSeparableConv2d(BaseStubSeparableConv2d):
  def __init__(self, obj):
    super(StubSeparableConv2d, self).__init__(obj.input_channel,
                                              obj.filters,
                                              obj.kernel_size_h,
                                              obj.kernel_size_w,
                                              obj.rate_h,
                                              obj.rate_w,
                                              obj.stride,
                                              obj.input,
                                              obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.separable_conv2d,
                      num_outputs=self.filters,
                      kernel_size=[self.kernel_size_h, self.kernel_size_w],
                      rate=[self.rate_h, self.rate_w],
                      depth_multiplier=1)(*args, **kwargs)


class StubSPP(BaseStubSPP):
  def __init__(self, obj):
    super(StubSPP, self).__init__(obj.grid_h, obj.grid_w, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    # 1.step average pooling
    _, input_h, input_w, input_c = self.input.shape
    grid_h = min(self.grid_h, input_h)
    grid_w = min(self.grid_w, input_w)
    output = functools.partial(slim.avg_pool2d,
                               kernel_size=[grid_h, grid_w],
                               stride=[grid_h, grid_w])(*args, **kwargs)

    # 2.step 1x1 convolution
    output = slim.conv2d(output, input_c, 1)

    # 3.step bilinear resize
    output = tf.image.resize_bilinear(output, [input_h, input_w])

    return output


class StubConcatenate(BaseStubConcatenate):
  def __init__(self, obj):
    super(StubConcatenate, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.concat, axis=-1)(*args, **kwargs)


class StubAdd(BaseStubAdd):
  def __init__(self, obj):
    super(StubAdd, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.add)(*args, **kwargs)


class StubFlatten(BaseStubFlatten):
  def __init__(self, obj):
    super(StubFlatten, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.flatten)(*args, **kwargs)


class StubReLU(BaseStubReLU):
  def __init__(self, obj):
    super(StubReLU, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.relu)(*args, **kwargs)


class StubReLU6(BaseStubReLU6):
  def __init__(self, obj):
    super(StubReLU6, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.relu6)(*args, **kwargs)


class StubSoftmax(BaseStubSoftmax):
  def __init__(self, obj):
    super(StubSoftmax, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.softmax)(*args, **kwargs)


class StubAvgPooling2d(BaseStubAvgPooling2d):
  def __init__(self, obj):
    super(StubAvgPooling2d, self).__init__(obj.kernel_size_h, obj.kernel_size_w, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.avg_pool2d,
                             kernel_size=(self.kernel_size_h,self.kernel_size_w),
                             stride=(self.kernel_size_h,self.kernel_size_w))(*args, **kwargs)


class StubMaxPooling2d(BaseStubMaxPooling2d):
  def __init__(self, obj):
    super(StubMaxPooling2d, self).__init__(obj.kernel_size_h, obj.kernel_size_w, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.max_pool2d,
                             kernel_size=(self.kernel_size_h,self.kernel_size_w),
                             stride=(self.kernel_size_h,self.kernel_size_w))(*args, **kwargs)


class StubGlobalPooling2d(BaseStubGlobalPooling2d):
  def __init__(self, obj):
    super(StubGlobalPooling2d, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.reduce_mean, axis=[1,2])(*args, **kwargs)


class StubDropout2d(BaseStubDropout2d):
  def __init__(self, obj):
    super(StubDropout2d, self).__init__(obj.rate, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.dropout, keep_prob=self.rate)(*args,**kwargs)


class StubInput(BaseStubInput):
  def __init__(self, obj):
    super(StubInput, self).__init__(obj.shape, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.placeholder, shape=self.shape, dtype=tf.float32)(*args, **kwargs)


class StubBatchNormalization2d(BaseStubBatchNormalization2d):
  def __init__(self, obj):
    super(StubBatchNormalization2d, self).__init__(obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.batch_norm)(*args, **kwargs)


class StubBilinearResize(BaseStubBilinearResize):
  def __init__(self, obj):
    super(StubBilinearResize, self).__init__(obj.height, obj.width, obj.input, obj.output)

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.image.resize_bilinear, size=(self.height,self.width))(*args, **kwargs)


class LayerFactory(object):
  def __init__(self):
    pass

  def __getattr__(self, item):
    if item not in ['dense',
                    'conv2d',
                    'separable_conv2d',
                    'concat',
                    'add',
                    'avg_pool2d',
                    'max_pool2d',
                    'global_pool2d',
                    'relu',
                    'flatten',
                    'relu6',
                    'bn2d',
                    'softmax',
                    'dropout_2d',
                    'bilinear_resize',
                    'spp',
                    'input']:
      return getattr(super(LayerFactory, self), item)

    def func(*args, **kwargs):
      if item == 'dense':
        return StubDense(*args, **kwargs)
      elif item == 'conv2d':
        return StubConv2d(*args, **kwargs)
      elif item == 'separable_conv2d':
        return StubSeparableConv2d(*args, **kwargs)
      elif item == 'spp':
        return StubSPP(*args, **kwargs)
      elif item == 'concat':
        return StubConcatenate(*args, **kwargs)
      elif item == 'add':
        return StubAdd(*args, **kwargs)
      elif item == 'avg_pool2d':
        return StubAvgPooling2d(*args, **kwargs)
      elif item == 'max_pool2d':
        return StubMaxPooling2d(*args, **kwargs)
      elif item == 'global_pool2d':
        return StubGlobalPooling2d(*args, **kwargs)
      elif item == 'relu':
        return StubReLU(*args, **kwargs)
      elif item == 'relu6':
        return StubReLU6(*args, **kwargs)
      elif item == 'flatten':
        return StubFlatten(*args, **kwargs)
      elif item == 'bn2d':
        return StubBatchNormalization2d(*args, **kwargs)
      elif item == 'softmax':
        return StubSoftmax(*args, **kwargs)
      elif item == 'dropout_2d':
        return StubDropout2d(*args, **kwargs)
      elif item == 'bilinear_resize':
        return StubBilinearResize(*args, **kwargs)
      elif item == 'input':
        return StubInput(*args, **kwargs)

    return func


if __name__ == '__main__':
  layer_factory = LayerFactory()
  input = layer_factory.input(shape=[1,224,224,3], dtype=tf.float32)
  conv2d_1 = layer_factory.conv2d(3, 16, kernel_size=3)
  conv2d_2 = layer_factory.conv2d(16, 32, kernel_size=3)
  conv2d_3 = layer_factory.conv2d(32, 64, kernel_size=3)
  relu = layer_factory.relu()
  conv2d_4 = layer_factory.conv2d(64,2, kernel_size=3)
  flatten = layer_factory.flatten()
  softmax = layer_factory.softmax()

  tf_input = input()
  tf_conv2d_1 = conv2d_1(tf_input)
  tf_conv2d_2 = conv2d_2(tf_conv2d_1)
  tf_conv2d_3 = conv2d_3(tf_conv2d_2)
  tf_relu = relu(tf_conv2d_3)
  tf_conv2d_4 = conv2d_4(tf_relu)
  tf_flatten = flatten(tf_conv2d_4)
  tf_softmax = softmax(tf_flatten)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    aa = sess.run(tf_softmax, feed_dict={tf_input: np.random.random([1,224,224,3])})
    print(aa)
  pass

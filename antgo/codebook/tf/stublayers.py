# -*- coding: UTF-8 -*-
# @Time    : 2018/11/21 5:14 PM
# @File    : stublayers.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.stublayer import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import functools
import numpy as np


class StubWeightBiasLayer(StubLayer):
  def import_weights(self, layer):
    pass

  def export_weights(self, layer):
    pass


class StubDense(StubWeightBiasLayer):
  def __init__(self, input_units, units, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.input_units = input_units
    self.units = units
    self.layer_type = 'dense'
    self.layer_width = units

  @property
  def output_shape(self):
    return self.units,

  def size(self):
    return self.input_units * self.units + self.units

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.fully_connected,num_outputs=self.units)(*args, **kwargs)

  def flops(self):
    return 0


class StubConv2d(StubWeightBiasLayer):
  def __init__(self, input_channel, filters, kernel_size_h, kernel_size_w, rate_h=1, rate_w=1, stride=1, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.input_channel = input_channel
    self.filters = filters
    self.kernel_size_h = kernel_size_h
    self.kernel_size_w = kernel_size_w
    self.rate_h = rate_h
    self.rate_w = rate_w
    self.stride = stride
    self.layer_type = 'conv2d'
    self.layer_width = filters
    self.n_dim = 2
    assert(stride == 1)

  @property
  def output_shape(self):
    ret = (self.input.shape[0],)
    for dim in self.input.shape[1:-1]:
      ret = ret + (max(int(dim / self.stride), 1),)

    ret = ret + (self.filters,)
    return ret

  def size(self):
    return self.filters * self.kernel_size_h * self.kernel_size_w * self.input_channel + self.filters

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.conv2d,
                             num_outputs=self.filters,
                             kernel_size=[self.kernel_size_h,
                                          self.kernel_size_w],
                             rate=[self.rate_h, self.rate_w],
                             stride=self.stride)(*args, **kwargs)

  def flops(self):
    return 0


class StubSeparableConv2d(StubWeightBiasLayer):
  def __init__(self, input_channel, filters, kernel_size_h, kernel_size_w, rate_h=1, rate_w=1, stride=1, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.input_channel = input_channel
    self.filters = filters
    self.kernel_size_h = kernel_size_h
    self.kernel_size_w = kernel_size_w
    self.rate_h = rate_h
    self.rate_w = rate_w
    self.stride = stride
    self.layer_type = 'conv2d'
    self.layer_width = filters
    self.n_dim = 2
    assert(stride == 1)

  @property
  def output_shape(self):
    ret = (self.input.shape[0],)
    for dim in self.input.shape[1:-1]:
      ret = ret + (max(int(dim / self.stride), 1),)

    ret = ret + (self.filters,)
    return ret

  def size(self):
    return self.filters * self.kernel_size_h * self.kernel_size_w*self.input_channel + self.filters

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.separable_conv2d,
                      num_outputs=self.filters,
                      kernel_size=[self.kernel_size_h, self.kernel_size_w],
                      rate=[self.rate_h, self.rate_w],
                      depth_multiplier=1)(*args, **kwargs)

  def flops(self):
    return 0


class StubSPP(StubWeightBiasLayer):
  def __init__(self, grid_h, grid_w, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.grid_h = grid_h
    self.grid_w = grid_w
    self.layer_type = 'spp'

  @property
  def output_shape(self):
    return self.input.shape

  def size(self):
    return 0

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

  def flops(self):
    return 0


class StubConcatenate(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    if input is None:
      input = []
    super().__init__(input, output)
    self.layer_type = 'concat'

  @property
  def output_shape(self):
    ret = 0
    for current_input in self.input:
      ret += current_input.shape[-1]
    ret = tuple(self.input[0].shape[:-1]) + (ret,)
    return ret

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.concat, axis=-1)(*args, **kwargs)


class StubAdd(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'add'

  @property
  def output_shape(self):
    return self.input[0].shape

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.add)(*args, **kwargs)

  def flops(self):
    return 0


class StubFlatten(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'flatten'

  @property
  def output_shape(self):
    ret = 1
    for dim in self.input.shape:
      ret *= dim
    return ret,

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.flatten)(*args, **kwargs)


class StubReLU(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'relu'

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.relu)(*args, **kwargs)

  def flops(self):
    return 0


class StubReLU6(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'relu6'

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.relu6)(*args, **kwargs)

  def flops(self):
    return 0


class StubSoftmax(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'softmax'

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.softmax)(*args, **kwargs)

  def flops(self):
    return 0


class StubPooling(StubLayer):
  def __init__(self, kernel_size_h=2, kernel_size_w=2, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.kernel_size_h = kernel_size_h
    self.kernel_size_w = kernel_size_w
    self.is_spatial_change = True
    self.layer_type = 'pool2d'

  @property
  def output_shape(self):
    ret = (self.input.shape[0],)
    ret = ret + (max(int(self.input.shape[1] / self.kernel_size_h), 1),)
    ret = ret + (max(int(self.input.shape[2] / self.kernel_size_w), 1),)
    ret = ret + (self.input.shape[-1],)
    return ret


class StubAvgPooling2d(StubPooling):
  def __call__(self, *args, **kwargs):
    return functools.partial(slim.avg_pool2d,
                             kernel_size=(self.kernel_size_h,self.kernel_size_w),
                             stride=(self.kernel_size_h,self.kernel_size_w))(*args, **kwargs)

  def flops(self):
    return 0


class StubMaxPooling2d(StubPooling):
  def __call__(self, *args, **kwargs):
    return functools.partial(slim.max_pool2d,
                             kernel_size=(self.kernel_size_h,self.kernel_size_w),
                             stride=(self.kernel_size_h,self.kernel_size_w))(*args, **kwargs)

  def flops(self):
    return 0


class StubGlobalPooling(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'global_pool2d'

  @property
  def output_shape(self):
    return (self.input.shape[0], 1, 1, self.input.shape[-1])


class StubGlobalPooling2d(StubGlobalPooling):
  def __call__(self, *args, **kwargs):
    return functools.partial(tf.reduce_mean, axis=[1,2])(*args, **kwargs)

  def flops(self):
    return 0


class StubDropout(StubLayer):
  def __init__(self, rate, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.rate = rate
    self.layer_type = 'dropout'


class StubDropout2d(StubDropout):
  def __call__(self, *args, **kwargs):
    return functools.partial(tf.nn.dropout, keep_prob=self.rate)(*args,**kwargs)


class StubInput(StubLayer):
  def __init__(self, shape, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.shape = shape
    self.layer_type = 'placeholder'

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.placeholder, shape=self.shape, dtype=tf.float32)(*args, **kwargs)


class StubBatchNormalization2d(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.layer_type = 'bn'
    self.n_dim = 2

  def __call__(self, *args, **kwargs):
    return functools.partial(slim.batch_norm)(*args, **kwargs)

  def flops(self):
    return 0


class StubBilinearResize(StubLayer):
  def __init__(self, height, width, input=None, output=None, **kwargs):
    super().__init__(input, output)
    self.height = height
    self.width = width
    self.layer_type = 'resize'

  @property
  def output_shape(self):
    return (self.input.shape[0], self.height, self.width, self.input.shape[-1])

  def __call__(self, *args, **kwargs):
    return functools.partial(tf.image.resize_bilinear, size=(self.height,self.width))(*args, **kwargs)

  def flops(self):
    return 0


class LayerFactory():
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

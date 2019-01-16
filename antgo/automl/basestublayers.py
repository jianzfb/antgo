# -*- coding: UTF-8 -*-
# @Time    : 2018/12/18 12:14 PM
# @File    : basestublayers.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.stublayer import *
import numpy as np


class BaseStubWeightBiasLayer(StubLayer):
  def import_weights(self, layer):
    pass

  def export_weights(self, layer):
    pass


class BaseStubDense(BaseStubWeightBiasLayer):
  def __init__(self, input_units, units, input=None, output=None, **kwargs):
    super(BaseStubDense, self).__init__(input, output, **kwargs)
    self.input_units = input_units
    self.units = units
    self.layer_type = 'dense'
    self.layer_name = 'dense'
    self.layer_width = units

  @property
  def output_shape(self):
    return self.units,

  def size(self):
    return self.input_units * self.units + self.units

  def flops(self):
    return self.input_units * self.units + (self.input_units - 1) + self.units

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.dense(self.input_units, self.units, block_name=self.block_name, cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubConv2d(BaseStubWeightBiasLayer):
  def __init__(self, input_channel, filters, kernel_size_h, kernel_size_w, rate_h=1, rate_w=1, stride=1, input=None, output=None, **kwargs):
    super(BaseStubConv2d, self).__init__(input, output, **kwargs)
    self.input_channel = input_channel
    self.filters = filters
    self.kernel_size_h = kernel_size_h
    self.kernel_size_w = kernel_size_w
    self.rate_h = rate_h
    self.rate_w = rate_w
    self.stride = stride
    self.layer_type = 'conv2d'
    self.layer_name = 'conv2d'
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

  def flops(self):
    n = self.input_channel * self.kernel_size_h * self.kernel_size_w  # vector_length
    flops_per_instance = n + (n - 1)  # general defination for number of flops (n: multiplications and n-1: additions)
    num_instances_per_filter = ((self.input.shape[1] - self.kernel_size_h + self.kernel_size_h / 2) / self.stride) + 1  # for rows
    num_instances_per_filter *= ((self.input.shape[2] - self.kernel_size_w + self.kernel_size_w / 2) / self.stride) + 1  # multiplying with cols

    flops_per_filter = num_instances_per_filter * flops_per_instance
    total_flops_per_layer = flops_per_filter * self.filters  # multiply with number of filters

    return total_flops_per_layer

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.conv2d(self.input_channel,
                                        self.filters,
                                        self.kernel_size_h,
                                        self.kernel_size_w,
                                        self.rate_h,
                                        self.rate_w,
                                        self.stride,
                                        block_name=self.block_name,
                                        cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubSeparableConv2d(BaseStubWeightBiasLayer):
  def __init__(self, input_channel, filters, kernel_size_h, kernel_size_w, rate_h=1, rate_w=1, stride=1, input=None, output=None, **kwargs):
    super(BaseStubSeparableConv2d, self).__init__(input, output, **kwargs)
    self.input_channel = input_channel
    self.filters = filters
    self.kernel_size_h = kernel_size_h
    self.kernel_size_w = kernel_size_w
    self.rate_h = rate_h
    self.rate_w = rate_w
    self.stride = stride
    self.layer_type = 'conv2d'
    self.layer_name = 'separable_conv2d'
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

  def flops(self):
    # 1.step depthwise convolution
    n = 1 * self.kernel_size_h * self.kernel_size_w  # vector_length
    flops_per_instance_step_1 = n + (n - 1)  # general defination for number of flops (n: multiplications and n-1: additions)
    num_instances_per_filter = ((self.input.shape[1] - self.kernel_size_h + self.kernel_size_h / 2) / self.stride) + 1    # for rows
    num_instances_per_filter *= ((self.input.shape[2] - self.kernel_size_w + self.kernel_size_w / 2) / self.stride) + 1   # multiplying with cols

    flops_per_filter_step_1 = num_instances_per_filter * flops_per_instance_step_1
    total_flops_per_layer_step_1 = flops_per_filter_step_1 * self.filters     # multiply with number of filters

    # 2.step pointwise convolution
    n = self.input_channel * 1 * 1
    flops_per_instance_step_2 = n + (n - 1)
    flops_per_filter_step_2 = num_instances_per_filter * flops_per_instance_step_2
    total_flops_per_layer_step_2 = flops_per_filter_step_2 * self.filters     # multiply with number of filters

    return total_flops_per_layer_step_1+total_flops_per_layer_step_2

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.separable_conv2d(self.input_channel,
                                                  self.filters,
                                                  self.kernel_size_h,
                                                  self.kernel_size_w,
                                                  self.rate_h,
                                                  self.rate_w,
                                                  self.stride,
                                                  block_name=self.block_name,
                                                  cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubSPP(BaseStubWeightBiasLayer):
  def __init__(self, grid_h, grid_w, input=None, output=None, **kwargs):
    super(BaseStubSPP, self).__init__(input, output, **kwargs)
    self.grid_h = grid_h
    self.grid_w = grid_w
    self.layer_type = 'spp'
    self.layer_name = 'spp'

  @property
  def output_shape(self):
    return self.input.shape

  def size(self):
    return 0

  def flops(self):
    # 1.step average pooling
    flops_per_instance_step_1 = self.grid_h*self.grid_w
    operates = ((self.input.shape[1] - self.grid_h + self.grid_h / 2) / self.grid_h) + 1
    operates *= ((self.input.shape[2] - self.grid_w + self.grid_w / 2) / self.grid_w) + 1
    flops_step_1 = operates * flops_per_instance_step_1 * self.input.shape[3]

    # 2.step 1x1 convolution
    n = self.input.shape[3] * 1 * 1
    flops_step_2 = operates * (n + (n - 1)) * self.input.shape[3]     # multiply with number of filters

    # 3.step bilinear resize
    flops_step_3 = self.input.shape[1] * self.input.shape[2] * self.input.shape[3] * 7

    return flops_step_1 + flops_step_2 + flops_step_3

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.spp(self.grid_h,
                                     self.grid_w,
                                     self.input,
                                     block_name=self.block_name,
                                     cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubConcatenate(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    if input is None:
      input = []
    super(BaseStubConcatenate, self).__init__(input, output, **kwargs)
    self.layer_type = 'concat'
    self.layer_name = 'concat'

  @property
  def output_shape(self):
    ret = 0
    for current_input in self.input:
      ret += current_input.shape[-1]
    ret = tuple(self.input[0].shape[:-1]) + (ret,)
    return ret

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.concat(block_name=self.block_name,
                                        cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubAdd(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubAdd, self).__init__(input, output, **kwargs)
    self.layer_type = 'add'
    self.layer_name = 'add'

  @property
  def output_shape(self):
    return self.input[0].shape

  def flops(self):
    return self.input[0].shape[1] * self.input[0].shape[2] * self.input[0].shape[3] - 1

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.add(block_name=self.block_name,
                                     cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubFlatten(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubFlatten, self).__init__(input, output, **kwargs)
    self.layer_type = 'flatten'
    self.layer_name = 'flatten'

  @property
  def output_shape(self):
    ret = 1
    for dim in self.input.shape:
      ret *= dim
    return ret,

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.flatten(block_name=self.block_name,
                                         cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubReLU(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubReLU, self).__init__(input, output, **kwargs)
    self.layer_type = 'relu'
    self.layer_name = 'relu'

  def flops(self):
    return self.input.shape[1] * self.input.shape[2] * self.input.shape[3]

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.relu(block_name=self.block_name,
                                      cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubReLU6(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubReLU6, self).__init__(input, output, **kwargs)
    self.layer_type = 'relu6'
    self.layer_name = 'relu6'

  def flops(self):
    return self.input.shape[1] * self.input.shape[2] * self.input.shape[3]

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.relu6(block_name=self.block_name,
                                       cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubSoftmax(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubSoftmax, self).__init__(input, output, **kwargs)
    self.layer_type = 'softmax'
    self.layer_name = 'softmax'

  def flops(self):
    return 0

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.softmax(block_name=self.block_name,
                                         cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubPooling(StubLayer):
  def __init__(self, kernel_size_h=2, kernel_size_w=2, input=None, output=None, **kwargs):
    super(BaseStubPooling, self).__init__(input, output, **kwargs)
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


class BaseStubAvgPooling2d(BaseStubPooling):
  def __init__(self, kernel_size_h=2, kernel_size_w=2, input=None, output=None, **kwargs):
    super(BaseStubAvgPooling2d, self).__init__(kernel_size_h, kernel_size_w, input, output, **kwargs)
    self.layer_name = 'avg_pool2d'

  def flops(self):
    flops_per_instance_step_1 = self.kernel_size_h*self.kernel_size_w
    operates = ((self.input.shape[1] - self.kernel_size_h + self.kernel_size_h / 2) / self.kernel_size_h) + 1
    operates *= ((self.input.shape[2] - self.kernel_size_w + self.kernel_size_w / 2) / self.kernel_size_w) + 1
    total_flops = operates * flops_per_instance_step_1 * self.input.shape[3]

    return total_flops

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.avg_pool2d(self.kernel_size_h,
                                            self.kernel_size_w,
                                            block_name=self.block_name,
                                            cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubMaxPooling2d(BaseStubPooling):
  def __init__(self, kernel_size_h=2, kernel_size_w=2, input=None, output=None, **kwargs):
    super(BaseStubMaxPooling2d, self).__init__(kernel_size_h, kernel_size_w, input, output, **kwargs)
    self.layer_name = 'max_pool2d'

  def flops(self):
    flops_per_instance_step_1 = self.kernel_size_h * self.kernel_size_w
    operates = ((self.input.shape[1] - self.kernel_size_h + self.kernel_size_h / 2) / self.kernel_size_h) + 1
    operates *= ((self.input.shape[2] - self.kernel_size_w + self.kernel_size_w / 2) / self.kernel_size_w) + 1
    total_flops = operates * flops_per_instance_step_1 * self.input.shape[3]

    return total_flops

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.max_pool2d(self.kernel_size_h,
                                            self.kernel_size_w,
                                            block_name=self.block_name,
                                            cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubGlobalPooling(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubGlobalPooling, self).__init__(input, output, **kwargs)
    self.layer_type = 'global_pool2d'

  @property
  def output_shape(self):
    return (self.input.shape[0], 1, 1, self.input.shape[-1])


class BaseStubGlobalPooling2d(BaseStubGlobalPooling):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubGlobalPooling2d, self).__init__(input, output, **kwargs)
    self.layer_name = 'global_pool2d'

  def flops(self):
    return self.input.shape[1] * self.input.shape[2] * self.input.shape[3]

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.global_pool2d(block_name=self.block_name, cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubDropout(StubLayer):
  def __init__(self, rate, input=None, output=None, **kwargs):
    super(BaseStubDropout, self).__init__(input, output, **kwargs)
    self.rate = rate
    self.layer_type = 'dropout'


class BaseStubDropout2d(BaseStubDropout):
  def __init__(self, rate, input=None, output=None, **kwargs):
    super(BaseStubDropout2d, self).__init__(rate, input, output, **kwargs)
    self.layer_name = 'dropout_2d'

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.dropout_2d(self.rate,
                                            block_name=self.block_name,
                                            cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubInput(StubLayer):
  def __init__(self, shape, input=None, output=None, **kwargs):
    super(BaseStubInput, self).__init__(input, output, **kwargs)
    self.shape = shape
    self.layer_type = 'placeholder'
    self.layer_name = 'input'

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.input(self.shape)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubBatchNormalization2d(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(BaseStubBatchNormalization2d, self).__init__(input, output, **kwargs)
    self.layer_type = 'bn'
    self.layer_name = 'bn2d'
    self.n_dim = 2

  def flops(self):
    return self.input.shape[1]*self.input.shape[2]*self.input.shape[3]*2

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.bn2d(block_name=self.block_name,
                                      cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseStubBilinearResize(StubLayer):
  def __init__(self, height, width, input=None, output=None, **kwargs):
    super(BaseStubBilinearResize, self).__init__(input, output, **kwargs)
    self.height = height
    self.width = width
    self.layer_type = 'resize'
    self.layer_name = 'bilinear_resize'

  @property
  def output_shape(self):
    return (self.input.shape[0], self.height, self.width, self.input.shape[-1])

  def flops(self):
    return self.input.shape[1] * self.input.shape[2] * self.input.shape[3] * 7

  def __call__(self, *args, **kwargs):
    if self.layer_factory is not None:
      layer = self.layer_factory.bilinear_resize(self.height,
                                                 self.width,
                                                 block_name=self.block_name,
                                                 cell_name=self.cell_name)
      return layer(*args, **kwargs)

    raise NotImplementedError


class BaseLayerFactory(object):
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
      return getattr(super(BaseLayerFactory, self), item)

    def func(*args, **kwargs):
      if item == 'dense':
        return BaseStubDense(*args, **kwargs)
      elif item == 'conv2d':
        return BaseStubConv2d(*args, **kwargs)
      elif item == 'separable_conv2d':
        return BaseStubSeparableConv2d(*args, **kwargs)
      elif item == 'spp':
        return BaseStubSPP(*args, **kwargs)
      elif item == 'concat':
        return BaseStubConcatenate(*args, **kwargs)
      elif item == 'add':
        return BaseStubAdd(*args, **kwargs)
      elif item == 'avg_pool2d':
        return BaseStubAvgPooling2d(*args, **kwargs)
      elif item == 'max_pool2d':
        return BaseStubMaxPooling2d(*args, **kwargs)
      elif item == 'global_pool2d':
        return BaseStubGlobalPooling2d(*args, **kwargs)
      elif item == 'relu':
        return BaseStubReLU(*args, **kwargs)
      elif item == 'relu6':
        return BaseStubReLU6(*args, **kwargs)
      elif item == 'flatten':
        return BaseStubFlatten(*args, **kwargs)
      elif item == 'bn2d':
        return BaseStubBatchNormalization2d(*args, **kwargs)
      elif item == 'softmax':
        return BaseStubSoftmax(*args, **kwargs)
      elif item == 'dropout_2d':
        return BaseStubDropout2d(*args, **kwargs)
      elif item == 'bilinear_resize':
        return BaseStubBilinearResize(*args, **kwargs)
      elif item == 'input':
        return BaseStubInput(*args, **kwargs)

    return func

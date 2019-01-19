# -*- coding: UTF-8 -*-
# @Time    : 2019/1/11 1:47 PM
# @File    : branch.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.automl.basestublayers import *
import random


class Branch(StubLayer):
  def __init__(self, input=None, output=None, **kwargs):
    super(Branch, self).__init__(input, output, **kwargs)
    self.branch_name = kwargs.get('branch_name', '')


class DummyNode(object):
  def __init__(self, shape, id=-1):
    self.shape = shape
    self.id = id


class ConvBnBranch(Branch):
  def __init__(self, output_channel, input=None, output=None, **kwargs):
    super(ConvBnBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'convbn_branch'

    self.output_channel = output_channel
    self.layer_1 = BaseStubConv2d(None, self.output_channel, 1, 1, cell_name=self.cell_name, block_name=self.block_name)
    self.layer_2 = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    self.layer_3 = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)

  @property
  def output_shape(self):
    self.layer_1.input = self.input
    self.layer_2.input = DummyNode(self.layer_1.output_shape)
    self.layer_3.input = DummyNode(self.layer_2.output_shape)
    return self.layer_3.output_shape

  def flops(self):
    self.layer_1.input = self.input
    self.layer_2.input = DummyNode(self.layer_1.output_shape)
    self.layer_3.input = DummyNode(self.layer_2.output_shape)
    return self.layer_1.flops() + self.layer_2.flops() + self.layer_3.flops()

  def __call__(self, *args, **kwargs):
    layer_1_c = self.layer_factory.conv2d(None, self.output_channel, 1, 1, cell_name=self.cell_name, block_name=self.block_name)
    layer_1 = layer_1_c(*args, **kwargs)

    layer_2_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    layer_2 = layer_2_c(layer_1)

    layer_3_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    layer_3 = layer_3_c(layer_2)
    return layer_3

  @property
  def layer_type_encoder(self):
    # 0 ~ 0.1
    return 0.05


class SeperableConvBranch(Branch):
  def __init__(self, output_channel, input=None, output=None, **kwargs):
    super(SeperableConvBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'seperableconv_branch'

    # 3x3 atrous separable convolution
    # rate 1,3,6,9,12,15,18,21
    if 'rate_h' not in kwargs:
      rate_list = [1, 3, 6, 9, 12, 15, 18, 21]
      self.rate_h_index = random.randint(0, len(rate_list) - 1)
      self.rate_h = rate_list[self.rate_h_index]
      self.rate_w_index = random.randint(0, len(rate_list) - 1)
      self.rate_w = rate_list[self.rate_w_index]
    else:
      self.rate_h = kwargs['rate_h']
      self.rate_w = kwargs['rate_w']
      self.rate_h_index = kwargs['rate_h_index']
      self.rate_w_index = kwargs['rate_w_index']

    self.output_channel = output_channel
    self.layer_1 = BaseStubSeparableConv2d(input_channel=None,
                                           filters=self.output_channel,
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           rate_h=self.rate_h,
                                           rate_w=self.rate_w,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)

    self.layer_2 = BaseStubBatchNormalization2d(cell_name=self.cell_name,
                                                block_name=self.block_name)
    self.layer_3 = BaseStubReLU(cell_name=self.cell_name,
                                block_name=self.block_name)

  @property
  def output_shape(self):
    self.layer_1.input = self.input
    self.layer_2.input = DummyNode(self.layer_1.output_shape)
    self.layer_3.input = DummyNode(self.layer_2.output_shape)
    return self.layer_3.output_shape

  def flops(self):
    self.layer_1.input = self.input
    self.layer_2.input = DummyNode(self.layer_1.output_shape)
    self.layer_3.input = DummyNode(self.layer_2.output_shape)
    return self.layer_1.flops() + self.layer_2.flops() + self.layer_3.flops()

  def __call__(self, *args, **kwargs):
    layer_1_c = self.layer_factory.separable_conv2d(input_channel=None,
                                                    filters=self.output_channel,
                                                    kernel_size_h=3,
                                                    kernel_size_w=3,
                                                    rate_h=self.rate_h,
                                                    rate_w=self.rate_w,
                                                    cell_name=self.cell_name,
                                                    block_name=self.block_name)
    layer_1 = layer_1_c(*args, **kwargs)

    layer_2_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    layer_2 = layer_2_c(layer_1)

    layer_3_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    layer_3 = layer_3_c(layer_2)
    return layer_3

  @property
  def layer_type_encoder(self):
    # 0.1 ~ 0.2
    return (self.rate_h_index * 8 + self.rate_w_index) / 720.0 + 0.1


class SPPBranch(Branch):
  def __init__(self, input=None, output=None, **kwargs):
    super(SPPBranch, self).__init__(input, output, **kwargs)
    # spatial pyramid pooling
    # shape = clone_graph.node_list[output_node_id].shape
    # min_hw = min(shape[1], shape[2])
    self.layer_name = 'spp_branch'

    if 'grid_h' not in kwargs:
      gh = [1, 2, 4, 8]
      # gh = [n for n in gh if n < min_hw]
      self.grid_h_index = random.randint(0, len(gh) - 1)
      self.grid_h = gh[self.grid_h_index]

      gw = [1, 2, 4, 8]
      # gw = [n for n in gw if n < min_hw]
      self.grid_w_index = random.randint(0, len(gw) - 1)
      self.grid_w = gw[self.grid_w_index]
    else:
      self.grid_h = kwargs['grid_h']
      self.grid_w = kwargs['grid_w']
      self.grid_h_index = kwargs['grid_h_index']
      self.grid_w_index = kwargs['grid_w_index']

    self.layer_1 = BaseStubSPP(grid_h=self.grid_h,
                               grid_w=self.grid_w,
                               cell_name=self.cell_name,
                               block_name=self.block_name)

  @property
  def output_shape(self):
    self.layer_1.input = self.input
    return self.layer_1.output_shape

  def flops(self):
    self.layer_1.input = self.input
    return self.layer_1.flops()

  def __call__(self, *args, **kwargs):
    layer_1_c = self.layer_factory.spp(grid_h=self.grid_h,
                                       grid_w=self.grid_w,
                                       cell_name=self.cell_name,
                                       block_name=self.block_name)
    layer_1_c.input = self.input
    layer_1 = layer_1_c(*args, **kwargs)
    return layer_1

  @property
  def layer_type_encoder(self):
    # 0.2 ~ 0.3
    return (self.grid_h_index * 4 + self.grid_w_index) / 200.0 + 0.2


class FocusBranch(Branch):
  def __init__(self, output_channel, input=None, output=None, **kwargs):
    super(FocusBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'focus_branch'

    self.output_channel = output_channel
    if 'rate_list' not in kwargs:
      candidate_rate_list = [1, 2, 4, 6, 8]
      self.rate_list = random.sample(candidate_rate_list, 3)
      self.rate_list = sorted(self.rate_list)
    else:
      self.rate_list = kwargs['rate_list']

    self.group_1_conv = BaseStubSeparableConv2d(input_channel=None,
                                                filters=self.output_channel,
                                                kernel_size_h=3,
                                                kernel_size_w=3,
                                                rate_h=self.rate_list[0],
                                                rate_w=self.rate_list[0],
                                                cell_name=self.cell_name,
                                                block_name=self.block_name)

    self.group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name,block_name=self.block_name)
    self.group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)

    self.group_2_conv = BaseStubSeparableConv2d(input_channel=None,
                                                filters=self.output_channel,
                                                kernel_size_h=3,
                                                kernel_size_w=3,
                                                rate_h=self.rate_list[1],
                                                rate_w=self.rate_list[1],
                                                cell_name=self.cell_name,
                                                block_name=self.block_name)
    self.group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name,block_name=self.block_name)
    self.group_2_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)

    self.group_12_add = BaseStubAdd()
    self.group_3_conv = BaseStubSeparableConv2d(input_channel=None,
                                                filters=self.output_channel,
                                                kernel_size_h=3,
                                                kernel_size_w=3,
                                                rate_h=self.rate_list[2],
                                                rate_w=self.rate_list[2],
                                                cell_name=self.cell_name,
                                                block_name=self.block_name)
    self.group_3_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name,block_name=self.block_name)
    self.group_3_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)

    self.group_4_12_add = BaseStubAdd()
    self.group_4_123_add = BaseStubAdd()

  @property
  def output_shape(self):
    return (self.input.shape[0], self.input.shape[1], self.input.shape[2], self.output_channel)

  def flops(self):
    self.group_1_conv.input = self.input
    self.group_1_bn.input = DummyNode(self.group_1_conv.output_shape)
    self.group_1_relu.input = DummyNode(self.group_1_bn.output_shape)

    self.group_2_conv.input = DummyNode(self.group_1_relu.output_shape)
    self.group_2_bn.input = DummyNode(self.group_2_conv.output_shape)
    self.group_2_relu.input = DummyNode(self.group_2_bn.output_shape)

    self.group_12_add.input = [DummyNode(self.group_1_relu.output_shape), DummyNode(self.group_2_relu.output_shape)]
    self.group_3_conv.input = DummyNode(self.group_12_add.output_shape)
    self.group_3_bn.input = DummyNode(self.group_3_conv.output_shape)
    self.group_3_relu.input = DummyNode(self.group_3_bn.output_shape)

    self.group_4_12_add.input = [DummyNode(self.group_1_relu.output_shape), DummyNode(self.group_2_relu.output_shape)]
    self.group_4_123_add.input = [DummyNode(self.group_3_relu.output_shape), DummyNode(self.group_4_12_add.output_shape)]

    return self.group_1_conv.flops() + \
           self.group_1_bn.flops() + \
           self.group_1_relu.flops() +\
           self.group_2_conv.flops() + \
           self.group_2_bn.flops() +\
           self.group_2_relu.flops() +\
           self.group_12_add.flops()+ \
           self.group_3_conv.flops()+\
           self.group_3_bn.flops()+\
           self.group_3_relu.flops()+\
           self.group_4_12_add.flops()+\
           self.group_4_123_add.flops()

  def __call__(self, *args, **kwargs):
    # group 1
    group_1_conv_c = self.layer_factory.separable_conv2d(input_channel=None,
                                                    filters=self.output_channel,
                                                    kernel_size_h=3,
                                                    kernel_size_w=3,
                                                    rate_h=self.rate_list[0],
                                                    rate_w=self.rate_list[0],
                                                    cell_name=self.cell_name,
                                                    block_name=self.block_name)
    group_1_conv = group_1_conv_c(*args, **kwargs)

    group_1_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn = group_1_bn_c(group_1_conv)

    group_1_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu = group_1_relu_c(group_1_bn)

    # group 2
    group_2_conv_c = self.layer_factory.separable_conv2d(input_channel=None,
                                                    filters=self.output_channel,
                                                    kernel_size_h=3,
                                                    kernel_size_w=3,
                                                    rate_h=self.rate_list[1],
                                                    rate_w=self.rate_list[1],
                                                    cell_name=self.cell_name,
                                                    block_name=self.block_name)
    group_2_conv = group_2_conv_c(group_1_relu)

    group_2_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn = group_2_bn_c(group_2_conv)

    group_2_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_2_relu = group_2_relu_c(group_2_bn)

    group_12_add_c = self.layer_factory.add(cell_name=self.cell_name, block_name=self.block_name)
    group_12_add = group_12_add_c(*[[group_1_relu, group_2_relu]])

    # group 3
    group_3_conv_c = self.layer_factory.separable_conv2d(input_channel=None,
                                                    filters=self.output_channel,
                                                    kernel_size_h=3,
                                                    kernel_size_w=3,
                                                    rate_h=self.rate_list[2],
                                                    rate_w=self.rate_list[2],
                                                    cell_name=self.cell_name,
                                                    block_name=self.block_name)
    group_3_conv = group_3_conv_c(group_12_add)

    group_3_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_3_bn = group_3_bn_c(group_3_conv)

    group_3_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_3_relu = group_3_relu_c(group_3_bn)

    group_4_12_add_c = self.layer_factory.add(cell_name=self.cell_name, block_name=self.block_name)
    group_4_12_add = group_4_12_add_c(*[[group_1_relu, group_2_relu]])

    group_4_123_add_c = self.layer_factory.add(cell_name=self.cell_name, block_name=self.block_name)
    group_4_123_add = group_4_123_add_c(*[[group_4_12_add, group_3_relu]])

    return group_4_123_add

  @property
  def layer_type_encoder(self):
    # 0.3 ~ 0.4
    a, b, c = self.rate_list
    a_i = -1
    b_i = -1
    c_i = -1
    for i, m in enumerate(self.rate_list):
      if m == a:
        a_i = i
      if m == b:
        b_i = i
      if m == c:
        c_i = i

    return (a_i * 25 + b_i * 5 + c_i) / 690.0 + 0.3


class SEBranch(Branch):
  def __init__(self, input=None, output=None, **kwargs):
    super(SEBranch, self).__init__(input,output,**kwargs)
    self.layer_name = 'se_branch'

    if 'squeeze_channels' not in kwargs:
      candidate_squeeze_channels = [4, 8, 16]
      self.squeeze_channels = random.choice(candidate_squeeze_channels)
    else:
      self.squeeze_channels = kwargs['squeeze_channels']

  @property
  def output_shape(self):
    group_1 = BaseStubAvgPooling2d(kernel_size_h=self.input.shape[1], kernel_size_w=self.input.shape[2])
    group_1.input = self.input

    group_1_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.squeeze_channels,
                                           kernel_size_h=1,
                                           kernel_size_w=1,
                                           rate_h=1,
                                           rate_w=1,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_1_conv.input = DummyNode(group_1.output_shape)

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=1,
                                           kernel_size_w=1,
                                           rate_h=1,
                                           rate_w=1,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_3_sigmoid = BaseStubSigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid.input = DummyNode(group_2_bn.output_shape)

    group_4_multiply = BaseStubDot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply.input = [self.input, DummyNode(group_3_sigmoid.output_shape)]

    return group_4_multiply.output_shape

  def flops(self):
    group_1 = BaseStubAvgPooling2d(kernel_size_h=self.input.shape[1], kernel_size_w=self.input.shape[2])
    group_1.input = self.input

    group_1_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.squeeze_channels,
                                           kernel_size_h=1,
                                           kernel_size_w=1,
                                           rate_h=1,
                                           rate_w=1,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_1_conv.input = DummyNode(group_1.output_shape)

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=1,
                                           kernel_size_w=1,
                                           rate_h=1,
                                           rate_w=1,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_3_sigmoid = BaseStubSigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid.input = DummyNode(group_2_bn.output_shape)

    group_4_multiply = BaseStubDot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply.input = [self.input, DummyNode(group_3_sigmoid.output_shape)]

    return group_1.flops()+\
           group_1_conv.flops()+\
           group_1_bn.flops()+\
           group_1_relu.flops()+\
           group_2_conv.flops()+\
           group_2_bn.flops()+\
           group_3_sigmoid.flops()+\
           group_4_multiply.flops()

  def __call__(self, *args, **kwargs):
    group_1_layer_c = self.layer_factory.avg_pool2d(kernel_size_h=self.input.shape[1], kernel_size_w=self.input.shape[2])
    group_1_layer = group_1_layer_c(*args, **kwargs)

    group_1_conv_c = self.layer_factory.conv2d(None,
                                               filters=self.squeeze_channels,
                                               kernel_size_h=1,
                                               kernel_size_w=1,
                                               cell_name=self.cell_name,
                                               block_name=self.block_name
                                               )
    group_1_conv = group_1_conv_c(group_1_layer)

    group_1_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn = group_1_bn_c(group_1_conv)

    group_1_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu = group_1_relu_c(group_1_bn)

    group_2_conv_c = self.layer_factory.conv2d(None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=1,
                                           kernel_size_w=1,
                                           rate_h=1,
                                           rate_w=1,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv = group_2_conv_c(group_1_relu)

    group_2_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn = group_2_bn_c(group_2_conv)

    group_3_sigmoid_c = self.layer_factory.sigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid = group_3_sigmoid_c(group_2_bn)

    group_4_multiply_c = self.layer_factory.dot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply = group_4_multiply_c(*[[group_3_sigmoid, args[0]]], **kwargs)

    return group_4_multiply

  @property
  def layer_type_encoder(self):
    # 0.4 ~ 0.5
    if self.squeeze_channels == 4:
      return 0.43
    elif self.squeeze_channels == 8:
      return 0.46
    else:
      return 0.49


class RegionSEBranch(Branch):
  def __init__(self, input=None, output=None, **kwargs):
    super(RegionSEBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'regionse_branch'

    if 'squeeze_channels' not in kwargs:
      candidate_squeeze_channels = [4, 8, 16]
      self.squeeze_channels = random.choice(candidate_squeeze_channels)
    else:
      self.squeeze_channels = kwargs['squeeze_channels']

    if 'region_size' not in kwargs:
      candidate_region_sizes = [2, 4, 6, 8]
      self.region_size = random.choice(candidate_region_sizes)
    else:
      self.region_size = kwargs['region_size']

  @property
  def output_shape(self):
    group_1 = BaseStubAvgPooling2d(kernel_size_h=self.region_size, kernel_size_w=self.region_size)
    group_1.input = self.input

    group_1_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.squeeze_channels,
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_1_conv.input = DummyNode(group_1.output_shape)

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_3_sigmoid = BaseStubSigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid.input = DummyNode(group_2_bn.output_shape)

    group_resize = BaseStubBilinearResize(height=self.input.shape[1], width=self.input.shape[2])
    group_resize.input = DummyNode(group_3_sigmoid.output_shape)

    group_4_multiply = BaseStubDot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply.input = [self.input, DummyNode(group_resize.output_shape)]

    return group_4_multiply.output_shape

  def flops(self):
    group_1 = BaseStubAvgPooling2d(kernel_size_h=self.region_size, kernel_size_w=self.region_size)
    group_1.input = self.input

    group_1_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.squeeze_channels,
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_1_conv.input = DummyNode(group_1.output_shape)

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(input_channel=None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_3_sigmoid = BaseStubSigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid.input = DummyNode(group_2_bn.output_shape)

    group_resize = BaseStubBilinearResize(height=self.input.shape[1], width=self.input.shape[2])
    group_resize.input = DummyNode(group_3_sigmoid.output_shape)

    group_4_multiply = BaseStubDot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply.input = [self.input, DummyNode(group_resize.output_shape)]

    return group_1.flops()+\
           group_1_conv.flops()+\
           group_1_bn.flops()+\
           group_1_relu.flops()+\
           group_2_conv.flops()+\
           group_2_bn.flops()+\
           group_3_sigmoid.flops()+ \
           group_resize.flops()+\
           group_4_multiply.flops()

  def __call__(self, *args, **kwargs):
    group_1_layer_c = self.layer_factory.avg_pool2d(kernel_size_h=self.region_size, kernel_size_w=self.region_size)
    group_1_layer = group_1_layer_c(*args, **kwargs)

    group_1_conv_c = self.layer_factory.conv2d(None,
                                               filters=self.squeeze_channels,
                                               kernel_size_h=3,
                                               kernel_size_w=3,
                                               cell_name=self.cell_name,
                                               block_name=self.block_name
                                               )
    group_1_conv = group_1_conv_c(group_1_layer)

    group_1_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn = group_1_bn_c(group_1_conv)

    group_1_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu = group_1_relu_c(group_1_bn)

    group_2_conv_c = self.layer_factory.conv2d(None,
                                           filters=self.input.shape[-1],
                                           kernel_size_h=3,
                                           kernel_size_w=3,
                                           cell_name=self.cell_name,
                                           block_name=self.block_name)
    group_2_conv = group_2_conv_c(group_1_relu)

    group_2_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn = group_2_bn_c(group_2_conv)

    group_3_sigmoid_c = self.layer_factory.sigmoid(cell_name=self.cell_name, block_name=self.block_name)
    group_3_sigmoid = group_3_sigmoid_c(group_2_bn)

    group_resize_c = self.layer_factory.bilinear_resize(height=self.input.shape[1], width=self.input.shape[2])
    group_resize = group_resize_c(group_3_sigmoid)

    group_4_multiply_c = self.layer_factory.dot(cell_name=self.cell_name, block_name=self.block_name)
    group_4_multiply = group_4_multiply_c(*[[group_resize, args[0]]], **kwargs)

    return group_4_multiply

  @property
  def layer_type_encoder(self):
    # 0.4 ~ 0.5
    # region_size: 2, 4, 6, 8; squeeze_channels: 4, 8, 16
    sc_i = -1
    for i, s in enumerate([4,8,16]):
      if self.squeeze_channels == s:
        sc_i = i

    rs_i = -1
    for j,r in enumerate([2,4,6,8]):
      if self.region_size == r:
        rs_i = j

    return (sc_i * 4 + rs_i) / 160.0 + 0.4


class ResBranch(Branch):
  def __init__(self, output_channel, input=None, output=None, **kwargs):
    super(ResBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'res_branch'
    self.output_channel = output_channel

  @property
  def output_shape(self):
    return (self.input.shape[0], self.input.shape[1], self.input.shape[2], self.output_channel)

  def flops(self):
    group_1_conv = BaseStubConv2d(None, self.output_channel, 3, 3, cell_name=self.cell_name, block_name=self.block_name)
    group_1_conv.input = self.input

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(None, self.output_channel, 3, 3, cell_name=self.cell_name, block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_3 = BaseStubAdd(cell_name=self.cell_name, block_name=self.block_name)
    group_3.input = [self.input, DummyNode(group_2_bn.output_shape)]

    group_4 = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_4.input = DummyNode(group_3.output_shape)

    return group_1_conv.flops() + \
           group_1_bn.flops() +\
           group_1_relu.flops() +\
           group_2_conv.flops() +\
           group_2_bn.flops() + \
           group_3.flops() + \
           group_4.flops()

  def __call__(self, *args, **kwargs):
    group_1_conv_c = self.layer_factory.conv2d(None, self.output_channel, 3, 3, cell_name=self.cell_name, block_name=self.block_name)
    group_1_conv = group_1_conv_c(*args, **kwargs)

    group_1_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn = group_1_bn_c(group_1_conv)

    group_1_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu = group_1_relu_c(group_1_bn)

    group_2_conv_c = self.layer_factory.conv2d(None, self.output_channel, 3, 3, cell_name=self.cell_name, block_name=self.block_name)
    group_2_conv = group_2_conv_c(group_1_relu)

    group_2_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn = group_2_bn_c(group_2_conv)

    group_3_c = self.layer_factory.add(cell_name=self.cell_name, block_name=self.block_name)
    group_3_c = group_3_c(*[[group_2_bn, args[0]]], **kwargs)

    group_4_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_4 = group_4_c(group_3_c)

    return group_4

  @property
  def layer_type_encoder(self):
    # 0.5 ~ 0.6
    return 0.55


class BottleNeckResBranch(Branch):
  def __init__(self, output_channel, input=None, output=None, **kwargs):
    super(BottleNeckResBranch, self).__init__(input, output, **kwargs)
    self.layer_name = 'bottleneck_res_branch'
    self.output_channel = output_channel

    self.candidate_bottleneck = [8, 16, 32, 64]
    if 'bottleneck' not in kwargs:
      self.bottleneck = random.choice(self.candidate_bottleneck)
    else:
      self.bottleneck = kwargs['bottleneck']

  @property
  def output_shape(self):
    return (self.input.shape[0], self.input.shape[1], self.input.shape[2], self.output_channel)

  def flops(self):
    group_1_conv = BaseStubConv2d(None, self.bottleneck, 1, 1, cell_name=self.cell_name, block_name=self.block_name)
    group_1_conv.input = self.input

    group_1_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn.input = DummyNode(group_1_conv.output_shape)

    group_1_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu.input = DummyNode(group_1_bn.output_shape)

    group_2_conv = BaseStubConv2d(None, self.bottleneck, 3, 3, cell_name=self.cell_name,block_name=self.block_name)
    group_2_conv.input = DummyNode(group_1_relu.output_shape)

    group_2_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn.input = DummyNode(group_2_conv.output_shape)

    group_2_relu = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_2_relu.input = DummyNode(group_2_bn.output_shape)

    group_3_conv = BaseStubConv2d(None, self.output_channel, 1, 1, cell_name=self.cell_name, block_name=self.block_name)
    group_3_conv.input = DummyNode(group_2_relu.output_shape)

    group_3_bn = BaseStubBatchNormalization2d(cell_name=self.cell_name, block_name=self.block_name)
    group_3_bn.input = DummyNode(group_3_conv.output_shape)

    group_4 = BaseStubAdd(cell_name=self.cell_name, block_name=self.block_name)
    group_4.input = [self.input, DummyNode(group_3_bn.output_shape)]

    group_5 = BaseStubReLU(cell_name=self.cell_name, block_name=self.block_name)
    group_5.input = DummyNode(group_4.output_shape)

    return group_1_conv.flops() + \
           group_1_bn.flops() + \
           group_1_relu.flops() + \
           group_2_conv.flops() + \
           group_2_bn.flops() + \
           group_2_relu.flops() + \
           group_3_conv.flops() + \
           group_3_bn.flops() +\
           group_4.flops() + \
           group_5.flops()

  def __call__(self, *args, **kwargs):
    group_1_conv_c = self.layer_factory.conv2d(None, self.bottleneck, 1, 1, cell_name=self.cell_name,
                                               block_name=self.block_name)
    group_1_conv = group_1_conv_c(*args, **kwargs)

    group_1_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_1_bn = group_1_bn_c(group_1_conv)

    group_1_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_1_relu = group_1_relu_c(group_1_bn)

    group_2_conv_c = self.layer_factory.conv2d(None, self.bottleneck, 3, 3, cell_name=self.cell_name,
                                               block_name=self.block_name)
    group_2_conv = group_2_conv_c(group_1_relu)

    group_2_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_2_bn = group_2_bn_c(group_2_conv)

    group_2_relu_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_2_relu = group_2_relu_c(group_2_bn)

    group_3_conv_c = self.layer_factory.conv2d(None, self.output_channel, 1, 1, cell_name=self.cell_name,
                                               block_name=self.block_name)
    group_3_conv = group_3_conv_c(group_2_relu)

    group_3_bn_c = self.layer_factory.bn2d(cell_name=self.cell_name, block_name=self.block_name)
    group_3_bn = group_3_bn_c(group_3_conv)

    group_3_c = self.layer_factory.add(cell_name=self.cell_name, block_name=self.block_name)
    group_3_c = group_3_c(*[[group_3_bn, args[0]]], **kwargs)

    group_4_c = self.layer_factory.relu(cell_name=self.cell_name, block_name=self.block_name)
    group_4 = group_4_c(group_3_c)
    return group_4

  @property
  def layer_type_encoder(self):
    # 0.5 ~ 0.6
    mi = -1
    for i, m in enumerate(self.candidate_bottleneck):
      if self.bottleneck == m:
        mi = i

    return 0.5 + 0.1 / len(self.candidate_bottleneck) * mi


class PoolBranch(Branch):
  def __init__(self, input=None, output=None, **kwargs):
    super(PoolBranch, self).__init__(input, output, **kwargs)
    # spatial pyramid pooling
    # shape = clone_graph.node_list[output_node_id].shape
    # min_hw = min(shape[1], shape[2])
    self.layer_name = 'avg_branch'
    self.is_avg_pool = False

    if 'is_avg_pool' in kwargs:
      self.is_avg_pool = kwargs['is_avg_pool']
      if self.is_avg_pool:
        self.layer_1 = BaseStubAvgPooling2d(kernel_size_h=3,
                                            kernel_size_w=3,

                                            cell_name=self.cell_name,
                                            block_name=self.block_name)
      else:
        self.layer_1 = BaseStubMaxPooling2d(kernel_size_h=3,
                                            kernel_size_w=3,
                                            cell_name=self.cell_name,
                                            block_name=self.block_name)

    else:
      if random.random() < 0.5:
        self.layer_1 = BaseStubAvgPooling2d(kernel_size_h=3,
                                            kernel_size_w=3,
                                            cell_name=self.cell_name,
                                            block_name=self.block_name)
        self.is_avg_pool = True
      else:
        self.layer_1 = BaseStubMaxPooling2d(kernel_size_h=3,
                                            kernel_size_w=3,
                                            cell_name=self.cell_name,
                                            block_name=self.block_name)
        self.is_avg_pool = False

  @property
  def output_shape(self):
    self.layer_1.input = self.input
    return self.layer_1.output_shape

  def flops(self):
    self.layer_1.input = self.input
    return self.layer_1.flops()

  def __call__(self, *args, **kwargs):
    if self.is_avg_pool:
      layer_1_c = self.layer_factory.avg_pool2d(kernel_size_h=3,
                                                kernel_size_w=3,
                                                cell_name=self.cell_name,
                                                block_name=self.block_name)
      layer_1_c.input = self.input
      layer_1 = layer_1_c(*args, **kwargs)
      return layer_1
    else:
      layer_1_c = self.layer_factory.max_pool2d(kernel_size_h=3,
                                                kernel_size_w=3,
                                                cell_name=self.cell_name,
                                                block_name=self.block_name)
      layer_1_c.input = self.input
      layer_1 = layer_1_c(*args, **kwargs)
      return layer_1

  @property
  def layer_type_encoder(self):
    if self.is_avg_pool:
      return 1 / 300.0
    else:
      return 2 / 300.0
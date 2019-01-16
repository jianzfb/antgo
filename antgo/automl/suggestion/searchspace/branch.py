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
    return 1 / 10.0


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
    return (self.rate_h_index * 8 + self.rate_w_index) / 100.0


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
    return (self.grid_h_index * 4 + self.grid_w_index) / 200.0


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
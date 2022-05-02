# -*- coding: UTF-8 -*-
# @Time    : 2022/5/1 21:31
# @File    : nn.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch


class AvgPool1D(torch.nn.AvgPool1d):
  def __init__(self, kernel_size, stride=None, padding=0, exclusive=True, ceil_mode=False, name=None):
    super().__init__(kernel_size=kernel_size,
                     stride=stride,
                     padding=padding,
                     ceil_mode=ceil_mode,
                     count_include_pad=exclusive)


class AvgPool2D(torch.nn.AvgPool2d):
  def __init__(self,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               exclusive=True,
               divisor_override=None,
               data_format='NCHW',
               name=None):
    assert(data_format == 'NCHW')
    super().__init__(
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      ceil_mode=ceil_mode,
      count_include_pad=exclusive,
      divisor_override=divisor_override
    )


class AvgPool3D(torch.nn.AvgPool3d):
  def __init__(self,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               exclusive=True,
               divisor_override=None,
               data_format='NCDHW',
               name=None):
    assert(data_format == 'NCDHW')
    super().__init__(
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      ceil_mode=ceil_mode,
      count_include_pad=exclusive,
      divisor_override=divisor_override
    )


class BatchNorm(torch.nn.Module):
  def __init__(self,
               num_channels,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-05,
               param_attr=None,
               bias_attr=None,
               dtype='float32',
               data_layout='NCHW',
               in_place=False,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=False,
               use_global_stats=False,
               trainable_statistics=False):
    super().__init__()
    self.bn = torch.nn.BatchNorm2d(num_channels,
                                   eps=epsilon,
                                   momentum=1-momentum,
                                   affine=True,
                                   track_running_stats=use_global_stats,
                                   )
    self.act = act
    self.param_attr = param_attr
    self.bias_attr = bias_attr

    if self.act is not None:
      assert(self.act in ['relu', 'relu6'])

    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)

  def forward(self, x):
    x = self.bn(x)
    if self.act is not None:
      if self.act == 'relu':
        x = torch.relu(x)
      elif self.act == 'relu6':
        x = torch.nn.functional.relu6(x)

    return x


class BatchNorm1D(torch.nn.BatchNorm1d):
  def __init__(self,
               num_features,
               momentum=0.9,
               epsilon=1e-05,
               weight_attr=None,
               bias_attr=None,
               data_format='NCL',
               name=None,
               use_global_stats=True):
    momentum = 1-momentum
    super(BatchNorm1D, self).__init__(
      num_features=num_features,
      eps=epsilon,
      momentum=momentum,
      affine=True,
      track_running_stats=use_global_stats,
    )
    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class BatchNorm2D(torch.nn.BatchNorm2d):
  def __init__(self,
               num_features,
               momentum=0.9,
               epsilon=1e-05,
               weight_attr=None,
               bias_attr=None,
               data_format='NCHW',
               name=None,
               use_global_stats=True):
    momentum = 1 - momentum
    super(BatchNorm2D, self).__init__(
      num_features=num_features,
      eps=epsilon,
      momentum=momentum,
      affine=True,
      track_running_stats=use_global_stats,
    )
    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class BatchNorm3D(torch.nn.BatchNorm3d):
  def __init__(self,
               num_features,
               momentum=0.9,
               epsilon=1e-05,
               weight_attr=None,
               bias_attr=None,
               data_format='NCDHW',
               name=None,
               use_global_stats=True):
    momentum = 1 - momentum
    super(BatchNorm3D, self).__init__(
      num_features=num_features,
      eps=epsilon,
      momentum=momentum,
      affine=True,
      track_running_stats=use_global_stats,
    )
    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class BCEWithLogitsLoss(torch.nn.BCEWithLogitsLoss):
  def __init__(self,
               weight=None,
               reduction='mean',
               pos_weight=None,
               name=None):
    super(BCEWithLogitsLoss, self).__init__(
      weight=weight,
      size_average=None,
      reduction=reduction,
      pos_weight=pos_weight
    )


class Pad2D(torch.nn.ConstantPad2d):
  def __init__(self,
               padding,
               mode='constant',
               value=0.0,
               data_format='NCHW',
               name=None):
    assert(mode == 'constant')
    assert(data_format == 'NCHW')
    super(Pad2D, self).__init__(padding=padding, value=value)


class Conv1D(torch.nn.Conv1d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               padding_mode='zeros',
               weight_attr=None,
               bias_attr=None,
               data_format='NCL'):
    assert (data_format == 'NCL')
    super(Conv1D, self).__init__(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=True if bias_attr is not None else False,
                                 padding_mode=padding_mode,
                                 device=None,
                                 dtype=None)
    self.weight_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bias)


class Conv2D(torch.nn.Conv2d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               padding_mode='zeros',
               weight_attr=None,
               bias_attr=None,
               data_format='NCHW'):
    assert(data_format == 'NCHW')
    super(Conv2D, self).__init__(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=True if bias_attr is not None else False,
                                 padding_mode=padding_mode,
                                 device=None,
                                 dtype=None)
    self.weight_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bias)


class Conv3D(torch.nn.Conv3d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               padding_mode='zeros',
               weight_attr=None,
               bias_attr=None,
               data_format='NCDHW'):
    assert(data_format == 'NCDHW')
    super(Conv3D, self).__init__(in_channels,
                                 out_channels,
                                 kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=True if bias_attr is not None else False,
                                 padding_mode=padding_mode,
                                 device=None,
                                 dtype=None)
    self.weight_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bias)


class Conv2DTranspose(torch.nn.ConvTranspose2d):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=0,
               output_padding=0,
               groups=1,
               dilation=1,
               weight_attr=None,
               bias_attr=None,
               data_format='NCHW'):
    super().__init__(in_channels,
                     out_channels,
                     kernel_size,
                     stride=stride,
                     padding=padding,
                     output_padding=output_padding,
                     groups=groups,
                     bias=True if bias_attr is not None else False,
                     dilation=dilation,
                     padding_mode='zeros',
                     device=None,
                     dtype=None)
    self.weight_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bias)


class CrossEntropyLoss(torch.nn.CrossEntropyLoss):
  def __init__(self,
               weight=None,
               ignore_index=- 100,
               reduction='mean',
               soft_label=False,
               axis=- 1,
               name=None):
    super().__init__(
      weight=weight,
      size_average=None,
      ignore_index=ignore_index,
      reduce=None,
      reduction=reduction,
      label_smoothing=0.0)


class Dropout(torch.nn.Dropout):
  def __init__(self,
               p=0.5,
               axis=None,
               mode="upscale_in_train",
               name=None):
    super().__init__(
      p=p,
      inplace=False
    )


class Embedding(torch.nn.Embedding):
  def __init__(self,
               num_embeddings,
               embedding_dim,
               padding_idx=None,
               sparse=False,
               weight_attr=None,
               name=None):
    super().__init__(num_embeddings,
                     embedding_dim,
                     padding_idx=padding_idx,
                     max_norm=None,
                     norm_type=2.0,
                     scale_grad_by_freq=False,
                     sparse=sparse,
                     _weight=None,
                     device=None,
                     dtype=None)
    self.weight_attr = weight_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)


class GroupNorm(torch.nn.GroupNorm):
  def __init__(self,
               num_groups,
               num_channels,
               epsilon=1e-05,
               weight_attr=None,
               bias_attr=None,
               data_format='NCHW',
               name=None):
    assert(data_format == 'NCHW')
    super().__init__(num_groups,
                     num_channels,
                     eps=epsilon,
                     affine=True,
                     device=None,
                     dtype=None)
    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class InstanceNorm2D(torch.nn.InstanceNorm2d):
  def __init__(self,
               num_features,
               epsilon=1e-05,
               momentum=0.9,
               weight_attr=None,
               bias_attr=None,
               data_format="NCHW",
               track_running_stats=False,
               name=None):
    super().__init__(num_features,
                     eps=epsilon,
                     momentum=1.0-momentum,
                     affine=True,
                     track_running_stats=track_running_stats,
                     device=None, dtype=None)

    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class LayerNorm(torch.nn.LayerNorm):
  def __init__(self,
               normalized_shape,
               epsilon=1e-05,
               weight_attr=None,
               bias_attr=None,
               name=None):
    super().__init__(normalized_shape,
                     eps=epsilon,
                     elementwise_affine=True,
                     device=None, dtype=None)

    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class Linear(torch.nn.Linear):
  def __init__(self,
               in_features,
               out_features,
               weight_attr=None,
               bias_attr=None,
               name=None):
    super().__init__(in_features,
                     out_features,
                     bias=True if bias_attr is not None else False,
                     device=None, dtype=None)
    self.weight_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.weight_attr is not None and self.weight_attr.initializer is not None:
      self.weight_attr.initializer(self.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bias)


class L1Loss(torch.nn.L1Loss):
  def __init__(self, reduction='mean', name=None):
    super().__init__(size_average=None,
                     reduce=None,
                     reduction=reduction)


class MaxPool1D(torch.nn.MaxPool1d):
  def __init__(self,
               kernel_size,
               stride=None,
               padding=0,
               return_mask=False,
               ceil_mode=False,
               name=None):
    super().__init__(kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=1,
                     return_indices=return_mask,
                     ceil_mode=ceil_mode)


class MaxPool2D(torch.nn.MaxPool2d):
  def __init__(self,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               return_mask=False,
               data_format='NCHW',
               name=None):
    super().__init__(kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=1,
                     return_indices=return_mask,
                     ceil_mode=ceil_mode)


class MaxPool3D(torch.nn.MaxPool3d):
  def __init__(self,
               kernel_size,
               stride=None,
               padding=0,
               ceil_mode=False,
               return_mask=False,
               data_format='NCDHW',
               name=None):
    super().__init__(kernel_size,
                     stride=stride,
                     padding=padding,
                     dilation=1,
                     return_indices=return_mask,
                     ceil_mode=ceil_mode)


class ReLU(torch.nn.ReLU):
  def __init__(self, name=None):
    super().__init__(inplace=False)


class ReLU6(torch.nn.ReLU6):
  def __init__(self, name=None):
    super().__init__(inplace=False)


class Softmax(torch.nn.Softmax):
  def __init__(self, axis=- 1, name=None):
    super().__init__(dim=axis)


class SyncBatchNorm(torch.nn.SyncBatchNorm):
  def __init__(self,
               num_features,
               epsilon=1e-5,
               momentum=0.9,
               weight_attr=None,
               bias_attr=None,
               data_format='NCHW',
               use_global_stats=True,
               name=None):
    super().__init__(num_features,
                     eps=epsilon,
                     momentum=1.0-momentum,
                     affine=True,
                     track_running_stats=use_global_stats,
                     process_group=None,
                     device=None,
                     dtype=None)
    self.param_attr = weight_attr
    self.bias_attr = bias_attr
    self.init_weight()

  def init_weight(self):
    if self.param_attr is not None and self.param_attr.initializer is not None:
      self.param_attr.initializer(self.bn.weight)

    if self.bias_attr is not None and self.bias_attr.initializer is not None:
      self.bias_attr.initializer(self.bn.bias)


class Sequential(torch.nn.Sequential):
  def __init__(self, *args):
    super().__init__(*args)


class Layer(torch.nn.Module):
  def __init__(self):
    super().__init__()


class Functional(object):
  pass

functional = Functional()

def add_functional_function(func):
  setattr(functional, func.__name__, func)

@add_functional_function
def softmax(x, axis=- 1, dtype=None, name=None):
  return torch.softmax(x, dim=axis)

@add_functional_function
def smooth_l1_loss(input, label, reduction='mean', delta=1.0, name=None):
  return torch.nn.functional.smooth_l1_loss(input,
                                            label,
                                            size_average=None,
                                            reduce=None,
                                            reduction=reduction,
                                            beta=delta)

@add_functional_function
def relu(x, name=None):
  return torch.nn.functional.relu(x)

@add_functional_function
def mse_loss(input, label, reduction='mean', name=None):
  return torch.nn.functional.mse_loss(
    input,
    label,
    size_average=None, reduce=None, reduction=reduction
  )

@add_functional_function
def log_softmax(x, axis=- 1, dtype=None, name=None):
  return torch.nn.functional.log_softmax(x, dim=axis)

@add_functional_function
def leaky_relu(x, negative_slope=0.01, name=None):
  return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope)

@add_functional_function
def interpolate(x, size=None, scale_factor=None, mode='nearest', align_corners=False, align_mode=0, data_format='NCHW', name=None):
  assert(data_format == 'NCHW')
  return torch.nn.functional.interpolate(x,
                                         size=size,
                                         scale_factor=scale_factor,
                                         mode=mode,
                                         align_corners=align_corners,
                                         recompute_scale_factor=None)
@add_functional_function
def dropout(x, p=0.5, axis=None, training=True, mode="upscale_in_train", name=None):
  return torch.nn.functional.dropout(x, p=p, training=training)


class Initializer(object):
  pass

initializer = Initializer()

def add_initializer_function(func):
  setattr(initializer, func.__name__, func)

@add_initializer_function
def XavierUniform(fan_in=None, fan_out=None, name=None):
  return lambda x: torch.nn.init.xavier_uniform_(x)

@add_initializer_function
def XavierNormal(fan_in=None, fan_out=None, name=None):
  return lambda x:torch.nn.init.xavier_normal_(x)


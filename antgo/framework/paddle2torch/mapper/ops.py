# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 12:33
# @File    : ops.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch
from antgo.framework.paddle2torch.utils import *


class ParamAttr(object):
  def __init__(self,
               name=None,
               initializer=None,
               learning_rate=1.0,
               regularizer=None,
               trainable=True,
               do_model_average=False,
               need_clip=True):
    self.name = name
    self.initializer = initializer
    self.learning_rate = learning_rate
    self.regularizer = regularizer
    self.trainable = trainable
    self.do_model_average = do_model_average
    self.need_clip = need_clip


def reshape(x, shape):
  return torch.reshape(x, shape)


def add(input, other):
  return torch.add(input, other)

def clip(input, min, max):
  return torch.clip(input, min, max)

def concat(tensors, axis=0):
  return torch.concat(tensors, dim=axis)

def exp(input):
  return torch.exp(input)

def log(input):
  return torch.log(input)

def logical_and(input, other):
  return torch.logical_and(input, other)

def logical_not(input):
  return torch.logical_not(input)

def logical_or(input, other):
  return torch.logical_or(input, other)

def logical_xor(input, other):
  return torch.logical_xor(input, other)


def matmul(input, other):
  return torch.matmul(input, other)


def multiply(input, other):
  return torch.mul(input, other)

def maximum(input, other):
  return torch.max(input, other)

def max(x, axis=None, keepdim=False, name=None):
  return torch.max(x)

def mean(x, axis=-1, keepdim=False, name=None):
  return torch.mean(x)

def min(x, axis=-1, keepdim=False, name=None):
  return torch.min(x)

def ones(shape, dtype):
  return torch.ones(shape, dtype)

def ones_like(input):
  return torch.ones_like(input)

def split(x, num_or_sections, axis=0, name=None):
  return torch.split(x,num_or_sections,axis)

def sqrt(x, name=None):
  return torch.sqrt(x)


def stack(x, axis=0, name=None):
  return torch.stack(x,dim=axis)

def sum(x, axis=None, dtype=None, keepdim=False, name=None):
  return torch.sum(x,dim=axis,keepdim=keepdim)

def squeeze(x, axis=None, name=None):
  return torch.unsqueeze(x,dim=axis)

def zeros(shape, dtype=None, name=None):
  return torch.zeros(shape, dtype=dtype)

def zeros_like(x, dtype=None, name=None):
  return torch.zeros_like(x)

def uniform(shape, dtype='float32', min=-1.0, max=1.0, seed=0, name=None):
  x = torch.rand(shape, dtype=PADDLE_TORCH_TYPE_MAPPER[dtype])
  x = x * (max - min) + min
  return x

def rand(shape, dtype='float32'):
  return torch.rand(shape, dtype=PADDLE_TORCH_TYPE_MAPPER[dtype])


def to_tensor(data, dtype=None, place=None, stop_gradient=True):
  return torch.tensor(data,
                      PADDLE_TORCH_TYPE_MAPPER[dtype],
                      device=place if place is None else place.device(),
                      requires_grad=not stop_gradient)

def flatten(x, start_axis=0, stop_axis=- 1, name=None):
  return torch.flatten(x,start_axis,stop_axis)

def save(obj, f):
  return torch.save(obj, f)


def load(path, **configs):
  return torch.load(path,map_location=configs.get('map_location', None))


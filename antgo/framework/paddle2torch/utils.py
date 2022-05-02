# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 19:49
# @File    : utils.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch

PADDLE_TORCH_TYPE_MAPPER = {
  "float32": torch.float32,
  "float64": torch.float64,
  "int64": torch.int64,
  "int32": torch.int32,
}


class CPUPlace(object):
  def __init__(self):
    pass

  def device(self):
    return torch.device('cpu')

class CUDAPlace(object):
  def __init__(self, cuda_id=0):
    self.cuda_id = cuda_id

  def device(self):
    return torch.device(f'cuda:{self.cuda_id}')
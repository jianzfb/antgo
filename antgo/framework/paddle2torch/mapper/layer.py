# -*- coding: UTF-8 -*-
# @Time    : 2022/5/1 21:31
# @File    : layer.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch

def add_module_function(func):
  setattr(torch.nn.Module, func.__name__, func)

@add_module_function
def add_sublayer(self, name, module):
    self.add_module(name, module)



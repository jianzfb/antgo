# -*- coding: UTF-8 -*-
# @Time    : 2022/5/2 12:38
# @File    : nn.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch


class AvgPool1d(torch.nn.AvgPool1d):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def flops(self):
    return 0



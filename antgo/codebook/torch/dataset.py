# -*- coding: UTF-8 -*-
# @Time    : 2021/11/4 10:51 下午
# @File    : dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torch


class TorchDataset(torch.utils.data.Dataset):
  def __init__(self, dataset):
    self.dataset = dataset

  def __getitem__(self, index):
    return self.dataset.at(index)

  def __len__(self):
    return self.dataset.size
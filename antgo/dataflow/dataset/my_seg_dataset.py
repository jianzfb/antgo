# -*- coding: UTF-8 -*-
# @Time    : 2022/4/27 20:38
# @File    : my_seg_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import os
import gzip
import random
import numpy as np
from ...utils import logger
from .dataset import Dataset
import time
import copy
import json


class CustomSegDataset(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(CustomSegDataset, self).__init__(train_or_test, dir, params)
    self.file_list = list()
    anno_file = getattr(self, 'anno_file', f"{self.train_or_test}.txt")
    self.prefix = getattr(self, 'prefix', "")
    with open(os.path.join(self.dir, anno_file)) as f:
      for line in f:
        items = line.strip().split()

        full_path_im = os.path.join(self.dir, self.prefix, items[0])
        full_path_label = os.path.join(self.dir, self.prefix, items[1])
        if not os.path.exists(full_path_im):
          raise IOError(
            'The image file {} is not exist!'.format(full_path_im))
        if not os.path.exists(full_path_label):
          raise IOError('The image file {} is not exist!'.format(
            full_path_label))
        self.file_list.append([full_path_im, full_path_label])
    self.num_samples = len(self.file_list)

    logger.info(f"Image num = {self.num_samples}")

  @property
  def size(self):
    return self.num_samples

  def at(self, id):
    image_file, label_file = self.file_list[id]
    annotation = {
      'im_file': image_file,
      'semantic_file': label_file
    }

    return None, annotation

  def data_pool(self):
    for index in range(self.size):
      yield self.at(index)

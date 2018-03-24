# -*- coding: UTF-8 -*-
# File: omniglot.py
# Author: jian(jian@mltalker.com)
from __future__ import division
from __future__ import unicode_literals

import numpy as np
import os
import random
import copy
import time
from antgo.dataflow.dataset import Dataset
from antgo.utils.fs import download
from antgo.utils.fs import maybe_here_fixed_3_hierarchy

__all__ = ['Omniglot']
OmniglotURL = 'https://github.com/brendenlake/omniglot/raw/master/python/'
class Omniglot(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(Omniglot, self).__init__(train_or_test, dir)
    assert(train_or_test in ['train', 'val', 'sample'])
    self.train_or_test = train_or_test
    self.data_folder = dir       # data folder

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    maybe_data_dir = maybe_here_fixed_3_hierarchy(os.path.join(self.data_folder, train_or_test), '.png')
    if maybe_data_dir is None:
      if train_or_test == 'train':
        self.download(os.path.join(self.data_folder, train_or_test), ['images_background.zip'], OmniglotURL, auto_unzip=True)
        maybe_data_dir = os.path.join(self.data_folder, train_or_test, 'images_background')
      else:
        self.download(os.path.join(self.data_folder, train_or_test), ['images_evaluation.zip'], OmniglotURL, auto_unzip=True)
        maybe_data_dir = os.path.join(self.data_folder, train_or_test, 'images_evaluation')

    self.character_folders = [os.path.join(maybe_data_dir, family, character) \
                              for family in os.listdir(maybe_data_dir) \
                              if os.path.isdir(os.path.join(maybe_data_dir, family)) \
                              for character in os.listdir(os.path.join(maybe_data_dir, family))]

    self.character_labels = range(len(self.character_folders))
    self.labels_num = len(self.character_labels)

    self.character_image_paths = [
      [os.path.join(self.character_folders[i], p) for p in os.listdir(self.character_folders[i])] for i in
      range(len(self.character_folders))]

    self.samples_label = []
    self.samples_index = []
    for label in range(self.labels_num):
      for i in range(len(self.character_image_paths[label])):
        self.samples_label.append(label)
        self.samples_index.append(i)

    self.samples_num = 0
    for ll in self.character_image_paths:
      self.samples_num += len(ll)

    self.ids = list(range(self.samples_num))
    self.samples = self.make_data(self.character_image_paths, self.character_labels)

    # fixed seed
    self.seed = time.time()

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert (split_method == 'holdout')

    val_omniglot = Omniglot('val', self.dir)
    return self, val_omniglot

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      idxs = copy.deepcopy(self.ids)
      if self.rng:
          self.rng.shuffle(idxs)

      # generate single image
      for k in idxs:
          yield [self.load_image(self.samples[k][2]), self.samples[k][1]]

  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    return [self.load_image(self.samples[self.ids[id]][2]), self.samples[self.ids[id]][1]]

  @property
  def size(self):
      return len(self.ids)
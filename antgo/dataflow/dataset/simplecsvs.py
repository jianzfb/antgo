# -*- coding: UTF-8 -*-
# Time: 11/5/17
# File: simplecsvs.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import csv
from antgo.dataflow.dataset.dataset import *
import numpy as np
import os
import copy
import time

__all__ = ['CSV']


class CSV(Dataset):
  def __init__(self, train_or_test, dataset_dir=None, ext_params=None):
    super(CSV, self).__init__(train_or_test, dataset_dir, ext_params, 'CSV')
    dataset_path = os.path.join(dataset_dir, train_or_test)
    # make dirs
    if not os.path.exists(dataset_path):
      os.makedirs(dataset_path)

    # maybe download dataset
    all_csvs = self.all_csvs(dataset_path)
    if len(all_csvs) == 0:
      self.download(dataset_path, auto_untar=True)
      all_csvs = self.all_csvs(dataset_path)

    delimiter = getattr(self, 'delimiter', ',')
    is_skip_first = getattr(self, 'skip_first', 0)

    self.csv_data = []
    for csv_fp in all_csvs:
      with open(csv_fp,'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter.encode('utf-8'))
        samples = list(csv_reader)
        if is_skip_first:
          self.csv_data.extend(samples[1:])
        else:
          self.csv_data.extend(samples)

    self.ids = np.arange(0, len(self.csv_data)).tolist()
    self.seed = time.time()

  def all_csvs(self, dataset_path):
    all_files = os.listdir(dataset_path)
    all_csvs = []
    for file_name in all_files:
      if file_name[-3:].lower() == 'csv':
        all_csvs.append(os.path.join(dataset_path, file_name))

    return all_csvs

  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      ids = copy.copy(self.ids)
      if self.rng:
        self.rng.shuffle(ids)

      # filter by ids
      filter_ids = getattr(self, 'filter', None)
      if filter_ids is not None:
        ids = [i for i in ids if i in filter_ids]

      # label col
      label_col = getattr(self, 'label', None)
      if label_col is not None:
        label_col = int(label_col)

      for id in ids:
        if label_col is None:
          yield [self.csv_data[id]]
        else:
          cols = len(self.csv_data[id][0])
          data = [self.csv_data[id][i] for i in range(0, cols) if i != label_col]
          label = int(self.csv_data[id][label_col])
          yield [data, label]

  def at(self, id):
    label_col = getattr(self, 'label', None)
    if label_col is not None:
      label_col = int(label_col)

    if label_col is None:
      return [self.csv_data[id]]
    else:
      cols = len(self.csv_data[id][0])
      data = [self.csv_data[id][i] for i in range(0, cols) if i != label_col]
      label = int(self.csv_data[id][label_col])
      return [data, label]

  def split(self, split_params={}, split_method="holdout"):
    assert(self.train_or_test in ['train', 'sample'])

    category_ids = copy.copy(self.ids)
    if 'is_stratified' in split_params and split_params['is_stratified'] and \
        (split_method == 'repeated-holdout' or split_method == 'holdout'):

      # label col
      label_col = getattr(self, 'label', None)
      if label_col is not None:
        label_col = int(label_col)

      if label_col is None:
        for id in self.ids:
          category_ids[id] = 0 if random.random() > 0.5 else 1
      else:
        for id in self.ids:
          category_ids[id] = int(self.csv_data[id][label_col])

    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      np.random.shuffle(category_ids)

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = CSV(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = CSV(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids

    return train_dataset, val_dataset

  @property
  def size(self):
    return len(self.ids)
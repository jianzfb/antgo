# -*- coding: UTF-8 -*-
# @Time    : 17-8-29
# @File    : heart.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import *
import numpy as np
import copy
import time

__all__ = ['Heart']

class Heart(Dataset):
  def __init__(self, train_or_test, dataset_dir=None, ext_params=None):
    assert(train_or_test in ['train', 'val', 'test'])
    if train_or_test == 'test':
      train_or_test = 'val'
    super(Heart, self).__init__(train_or_test, dataset_dir, ext_params, 'heart')
    assert(os.path.exists(os.path.join(dataset_dir, 'train', 'heart_scale')))
    
    dataset_path = os.path.join(dataset_dir,'train','heart_scale')
    self.heart_data = np.zeros((270, 13))
    self.heart_label = []
    with open(dataset_path, 'r') as fp:
      data = fp.readline()
      count = 0
      while data:
        data = data.replace('\n', '')
        data = data.strip()
        items = data.split(' ')
        label = 1
        if items[0][0] == '+':
          # positive
          label = 1
        else:
          # negative
          label = 0
        
        for value in items[1:]:
          k, v = value.split(':')
          self.heart_data[count, int(k)-1] = float(v)
        
        self.heart_label.append({'category': label, 'category_id': label})
        count += 1
        data = fp.readline()
    
    # dataset
    self.ids = np.arange(0, 270).tolist()

    self.seed = time.time()
  
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
      
      for id in ids:
        data = self.heart_data[id]
        label = self.heart_label[id]
        
        label = self.filter_by_condition(label)
        if label is None:
          continue
        
        yield [data, label]
  
  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')

    category_ids = copy.copy(self.ids)
    if 'is_stratified' in split_params and split_params['is_stratified'] and \
        (split_method == 'repeated-holdout' or split_method == 'holdout'):

      # traverse dataset
      for id in self.ids:
          category_ids[id] = self.heart_label[id]['category']

    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      np.random.shuffle(category_ids)

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = Heart(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = Heart(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids
    return train_dataset, val_dataset
  
  @property
  def size(self):
    return 270
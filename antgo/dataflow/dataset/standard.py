# encoding=utf-8
# @Time    : 17-8-2
# @File    : standard.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import *
from antgo.dataflow.core import *
from antgo.utils.fs import maybe_here_match_format
from antgo.dataflow.basic import *
import numpy as np
import copy
import time


class Standard(Dataset):
  def __init__(self, train_or_test, dataset_dir=None, ext_params=None):
    dataset_name = dataset_dir.split('/')[-1]
    super(Standard, self).__init__(train_or_test, dataset_dir, ext_params, dataset_name)
    assert(os.path.exists(dataset_dir))

    self._record_reader = RecordReader(os.path.join(dataset_dir, train_or_test))
    for k, v in self._record_reader.record_attrs().items():
      setattr(self, k, v)

    # dataset
    self.ids = np.arange(0, int(self.count)).tolist()

    # fixed seed
    self.seed = time.time()

  def close(self):
    self._record_reader.close()
    self._record_reader = None

  def data_pool(self):
    self.epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if self.epoch >= max_epoches:
        break
      self.epoch += 1

      ids = copy.copy(self.ids)
      if self.rng:
        self.rng.shuffle(ids)
      
      # filter by ids
      filter_ids = getattr(self, 'filter', None)
      if filter_ids is not None:
        ids = [i for i in ids if i in filter_ids]

      for id in ids:
        data, label = self._record_reader.read(id, 'data', 'label')
        # filter by condition
        if type(label) == dict:
          if 'category' in label and 'category_id' in label:
            label = self.filter_by_condition(label)
            if label is None:
              continue
          
          label['id'] = id
        yield [data, label]

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')

    category_ids = copy.copy(self.ids)
    if 'is_stratified' in split_params and split_params['is_stratified'] and \
        (split_method == 'repeated-holdout' or split_method == 'holdout'):

      # traverse dataset
      for id in self.ids:
        _, label = self._record_reader.read(id, 'data', 'label')
        if type(label) == dict and 'category' in label:
          category_ids[id] = label['category']
        else:
          category_ids[id] = 0 if random.random() > 0.5 else 1

    if split_method == 'holdout':
      if 'ratio' not in split_params:
        train_dataset = Standard(self.train_or_test, self.dir, self.ext_params)
        val_dataset = Standard('val', self.dir, self.ext_params)
        return train_dataset, val_dataset

    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      np.random.shuffle(category_ids)

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = Standard(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = Standard(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids
    return train_dataset, val_dataset

  @property
  def size(self):
    return len(self.ids)
  
  def at(self, id):
    data, label = self._record_reader.read(id, 'data', 'label')
    return data, label
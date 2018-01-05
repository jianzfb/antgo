# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : lfw.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np

__all__ = ['LFW']

LFW_URL = 'http://vis-www.cs.umass.edu/lfw/index.html#download'
class LFW(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(LFW, self).__init__(train_or_test, dir)
    assert(train_or_test == 'train')
    
    image_flag = getattr(params, 'image', 'align')
    # 0.step maybe download
    if image_flag == 'align':
      if not os.path.exists(os.path.join(self.dir, 'lfw-deepfunneled')):
        self.download(self.dir,
                      file_names=[],
                      default_url='http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz',
                      auto_untar=True,
                      is_gz=True)
    else:
      if not os.path.exists(os.path.join(self.dir, 'lfw')):
        self.download(self.dir,
                      file_names=[],
                      default_url='http://vis-www.cs.umass.edu/lfw/lfw.tgz',
                      auto_untar=True,
                      is_gz=True)
    
    # 1.step data folder (wild or align)
    if image_flag == 'align':
      self._data_folder = os.path.join(self.dir, 'lfw-deepfunneled')
    else:
      self._data_folder = os.path.join(self.dir, 'lfw')
    
    # 2.step data files
    self._persons_list = []
    self._persons_id_str = []
    self._persons_id = []
    for person_id_str in os.listdir(self._data_folder):
      if person_id_str[0] == '.':
        continue
      for person_image in os.listdir(os.path.join(self._data_folder, person_id_str)):
        self._persons_list.append(os.path.join(self._data_folder, person_id_str, person_image))
        self._persons_id_str.append(person_id_str)
    
    id_set = set(self._persons_id_str)
    person_id_map = {}
    for s_i, s in enumerate(id_set):
      person_id_map[s] = s_i
    
    for person_id_str in self._persons_id_str:
      person_id = person_id_map[person_id_str]
      self._persons_id.append(person_id)
    
    self.ids = list(range(len(self._persons_list)))

    # fixed seed
    self.seed = time.time()
    
  @property
  def size(self):
    return len(self.ids)
  
  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1
    
      # idxs = np.arange(len(self.ids))
      idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(idxs)
        
      for k in idxs:
        person_file = self._persons_list[k]
        person_image = imread(person_file)
        person_id_str = self._persons_id_str[k]
        person_id = self._persons_id[k]
        
        yield [person_image, {'category_id': person_id,
                              'category': person_id_str,
                              'id': k,
                              'info': [person_image.shape[0], person_image.shape[1], person_image.shape[2]]}]
  
  def at(self, id):
    person_file = self._persons_list[id]
    person_image = imread(person_file)
    person_id_str = self._persons_id_str[id]
    person_id = self._persons_id[id]
  
    return person_image, {'category_id': person_id,
                          'category': person_id_str,
                          'id': id,
                          'info': [person_image.shape[0], person_image.shape[1], person_image.shape[2]]}
  
  def split(self, split_params={}, split_method=''):
    assert (self.train_or_test == 'train')
    assert (split_method in ['repeated-holdout', 'bootstrap', 'kfold'])
    
    category_ids = None
    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      category_ids = [i for i in range(len(self.ids))]
      np.random.shuffle(category_ids)
    else:
      category_ids = [self._persons_id[i] for i in range(len(self.ids))]
    
    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = LFW(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids
    
    val_dataset = LFW(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids
    
    return train_dataset, val_dataset
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
    super(LFW, self).__init__(train_or_test, dir, params)
    assert(train_or_test in ['train', 'test','sample'])

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    self.image_flag = getattr(self, 'image', 'align')
    # 0.step maybe download
    if self.image_flag == 'align':
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
    if self.image_flag == 'align':
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
    self.person_id_map = {}
    for s_i, s in enumerate(id_set):
      self.person_id_map[s] = s_i
    
    for person_id_str in self._persons_id_str:
      person_id = self.person_id_map[person_id_str]
      self._persons_id.append(person_id)
    
    self.ids = list(range(len(self._persons_list)))

    # fixed seed
    self.seed = time.time()

    self.view = int(getattr(self, 'view', 0))
    self.pairs = []
    if self.view == 1:
      if self.train_or_test == 'train':
        with open(os.path.join(self.dir, 'pairsDevTrain.txt')) as fp:
          same_pair_num = fp.readline()
          same_pair_num = int(same_pair_num.strip())
          for pair_i in range(same_pair_num):
            content = fp.readline()
            person_name, image_a, image_b = content.strip().split('\t')
            person_a_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_a))
            person_b_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_b))

            self.pairs.append((person_a_image, person_b_image))

          content = fp.readline()
          while content:
            content = content.strip()
            if content == '':
              break
            diff_person_a_name, diff_person_a_index, diff_person_b_name, diff_person_b_index = \
              content.split('\t')

            diff_person_a_image = '%s/%s_%04d.jpg'%(diff_person_a_name, diff_person_a_name, int(diff_person_a_index))
            diff_person_b_image = '%s/%s_%04d.jpg'%(diff_person_b_name, diff_person_b_name, int(diff_person_b_index))
            self.pairs.append((diff_person_a_image, diff_person_b_image))

            content = fp.readline()
      elif self.train_or_test == 'test':
        with open(os.path.join(self.dir, 'pairsDevTest.txt')) as fp:
          same_pair_num = fp.readline()
          same_pair_num = int(same_pair_num.strip())
          for pair_i in range(same_pair_num):
            content = fp.readline()
            person_name, image_a, image_b = content.strip().split('\t')
            person_a_image = '%s/%s_%04d.jpg' % (person_name, person_name, int(image_a))
            person_b_image = '%s/%s_%04d.jpg' % (person_name, person_name, int(image_b))

            self.pairs.append((person_a_image, person_b_image))

          content = fp.readline()
          while content:
            content = content.strip()
            if content == '':
              break
            diff_person_a_name, diff_person_a_index, diff_person_b_name, diff_person_b_index = \
              content.split('\t')

            diff_person_a_image = '%s/%s_%04d.jpg' % (diff_person_a_name, diff_person_a_name, int(diff_person_a_index))
            diff_person_b_image = '%s/%s_%04d.jpg' % (diff_person_b_name, diff_person_b_name, int(diff_person_b_index))
            self.pairs.append((diff_person_a_image, diff_person_b_image))

            content = fp.readline()
    else:
      if os.path.exists(os.path.join(self.dir, 'pairs.txt')):
        pass
      if os.path.exists(os.path.join(self.dir, 'people.txt')):
        pass
    
  @property
  def size(self):
    if self.view == 0:
      return len(self.ids)
    elif self.view == 1:
      return len(self.pairs)
    else:
      return 0
  
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
    
      if self.view == 0:
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
      elif self.view == 1:
        pair_ids = list(range(len(self.pairs)))
        if self.rng:
          self.rng.shuffle(pair_ids)
        for pair_i in pair_ids:
          person_a, person_b = self.pairs[pair_i]
          person_a_name = person_a.split('/')[0]
          person_b_name = person_b.split('/')[0]
          person_a_path = os.path.join(self._data_folder, person_a)
          person_a_image = imread(person_a_path)
          person_b_path = os.path.join(self._data_folder, person_b)
          person_b_image = imread(person_b_path)

          data = np.stack([person_a_image, person_b_image],0)
          yield data, {'a_category': person_a_name,
                       'a_category_id': self.person_id_map[person_a_name],
                       'b_category': person_b_name,
                       'b_category_id': self.person_id_map[person_b_name],
                       'id': pair_i}
      else:
        pass

  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    if self.view == 0:
      person_file = self._persons_list[id]
      person_image = imread(person_file)
      person_id_str = self._persons_id_str[id]
      person_id = self._persons_id[id]

      return person_image, {'category_id': person_id,
                            'category': person_id_str,
                            'id': id,
                            'info': [person_image.shape[0], person_image.shape[1], person_image.shape[2]]}
    elif self.view == 1:
      pair_i = id
      person_a, person_b = self.pairs[pair_i]
      person_a_name = person_a.split('/')[0]
      person_b_name = person_b.split('/')[0]
      person_a_path = os.path.join(self.dir, person_a)
      person_a_image = imread(person_a_path)
      person_b_path = os.path.join(self.dir, person_b)
      person_b_image = imread(person_b_path)

      data = np.stack([person_a_image, person_b_image], 0)
      return data, {'a_category': person_a_name,
                   'a_category_id': self.person_id_map[person_a_name],
                   'b_category': person_b_name,
                   'b_category_id': self.person_id_map[person_b_name],
                   'id': pair_i}
    else:
      pass
  
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
# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File    : lfw.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import sys
from antgo.dataflow.dataset import *
import os
import numpy as np
import time
import copy
import cv2
from antgo.framework.helper.fileio.file_client import *


__all__ = ['LFW']

class LFW(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(LFW, self).__init__(train_or_test, dir, ext_params)
    assert(train_or_test in ['train', 'val', 'test'])
    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

    if not os.path.exists(os.path.join(self.dir, 'lfw-deepfunneled')):
      ali = AliBackend()
      ali.download('ali:///dataset/lfw/lfw-deepfunneled.tgz', self.dir)
      ali.download('ali:///dataset/lfw/pairsDevTest.txt', self.dir)
      ali.download('ali:///dataset/lfw/pairsDevTrain.txt', self.dir)
      ali.download('ali:///dataset/lfw/peopleDevTest.txt', self.dir)
      ali.download('ali:///dataset/lfw/peopleDevTrain.txt', self.dir)

      os.system(f'cd {self.dir} && tar -xf lfw-deepfunneled.tgz')

    self.task_type = getattr(self, 'task_type', 'CLASSIFICATION')
    assert(self.task_type in ['CLASSIFICATION', 'DISTANCE'])
    
    # 1.step data folder (wild or align)
    self._data_folder = os.path.join(self.dir, 'lfw-deepfunneled')

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
    self.same_pairs = []
    self.diff_pairs = []
    if self.train_or_test == 'train':
      with open(os.path.join(self.dir, 'pairsDevTrain.txt')) as fp:
        same_pair_num = fp.readline()
        same_pair_num = int(same_pair_num.strip())
        for pair_i in range(same_pair_num):
          content = fp.readline()
          person_name, image_a, image_b = content.strip().split('\t')
          person_a_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_a))
          person_b_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_b))

          self.same_pairs.append((person_a_image, person_b_image))

        content = fp.readline()
        while content:
          content = content.strip()
          if content == '':
            break
          diff_person_a_name, diff_person_a_index, diff_person_b_name, diff_person_b_index = \
            content.split('\t')

          diff_person_a_image = '%s/%s_%04d.jpg'%(diff_person_a_name, diff_person_a_name, int(diff_person_a_index))
          diff_person_b_image = '%s/%s_%04d.jpg'%(diff_person_b_name, diff_person_b_name, int(diff_person_b_index))
          self.diff_pairs.append((diff_person_a_image, diff_person_b_image))

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

          self.same_pairs.append((person_a_image, person_b_image))

        content = fp.readline()
        while content:
          content = content.strip()
          if content == '':
            break
          diff_person_a_name, diff_person_a_index, diff_person_b_name, diff_person_b_index = \
            content.split('\t')

          diff_person_a_image = '%s/%s_%04d.jpg' % (diff_person_a_name, diff_person_a_name, int(diff_person_a_index))
          diff_person_b_image = '%s/%s_%04d.jpg' % (diff_person_b_name, diff_person_b_name, int(diff_person_b_index))
          self.diff_pairs.append((diff_person_a_image, diff_person_b_image))

          content = fp.readline()

  @property
  def size(self):
    if self.task_type == 'CLASSIFICATION':
      return len(self.ids)
    else:
      return len(self.pairs)

  def at(self, id):
    if self.task_type == 'CLASSIFICATION':
      person_file = self._persons_list[id]
      person_image = cv2.imread(person_file)
      person_id = self._persons_id[id]

      return person_image, {
        'label': person_id,
        'image_meta': {
          'image_shape': (person_image.shape[0], person_image.shape[1]),
          'image_file': person_file
        }
      }
    else:
      pair_i = id
      if pair_i < len(self.same_pairs):
        same_person_a, same_person_b = self.same_pairs[pair_i]
        same_person_a_path = os.path.join(self.dir, same_person_a)
        same_person_a_image = cv2.imread(same_person_a_path)
        same_person_b_path = os.path.join(self.dir, same_person_b)
        same_person_b_image = cv2.imread(same_person_b_path)
        data = np.stack([same_person_a_image, same_person_b_image], 0)
        return data, {
          'label': 0,
          'image_meta': {
            'image_shape': (same_person_a_image.shape[0], same_person_a_image.shape[1]),
            'image_file': same_person_a_path
          }
        }
      else:
        diff_person_a, diff_person_b = self.diff_pairs[pair_i]
        diff_person_a_path = os.path.join(self.dir, diff_person_a)
        diff_person_a_image = cv2.imread(diff_person_a_path)
        diff_person_b_path = os.path.join(self.dir, diff_person_b)
        diff_person_b_image = cv2.imread(diff_person_b_path)
        data = np.stack([diff_person_a_image, diff_person_b_image], 0)        
        return data, {
          'label': 1,
          'image_meta': {
            'image_shape': (diff_person_a_image.shape[0], diff_person_a_image.shape[1]),
            'image_file': diff_person_a_path
          }
        }

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

# lfw = LFW('train', '/root/workspace/dataset/lfw')
# num = lfw.size
# for i in range(num):
#   data = lfw.sample(i)
#   print(data.keys())
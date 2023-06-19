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
from filelock import FileLock

__all__ = ['LFW']

class LFW(Dataset):
  def __init__(self, train_or_test, dir=None, ext_params=None):
    super(LFW, self).__init__(train_or_test, dir, ext_params)
    assert(train_or_test in ['train', 'val', 'test'])
    lock = FileLock('DATASET.lock')
    with lock:
      if not os.path.exists(os.path.join(self.dir, 'lfw-deepfunneled')):  
        # 数据集不存在，需要重新下载，并创建标记
        if not os.path.exists(self.dir):
          os.makedirs(self.dir)

        ali = AliBackend()
        ali.download('ali:///dataset/lfw/lfw-deepfunneled.tgz', self.dir)
        ali.download('ali:///dataset/lfw/pairsDevTest.txt', self.dir)
        ali.download('ali:///dataset/lfw/pairsDevTrain.txt', self.dir)
        ali.download('ali:///dataset/lfw/peopleDevTest.txt', self.dir)
        ali.download('ali:///dataset/lfw/peopleDevTrain.txt', self.dir)
        ali.download('ali:///dataset/lfw/pairs.txt', self.dir)
        ali.download('ali:///dataset/lfw/people.txt', self.dir)

        os.system(f'cd {self.dir} && tar -xf lfw-deepfunneled.tgz')

    self.purpose = getattr(self, 'purpose', 'development')
    assert(self.purpose in ['development', 'benchmark'])

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

    self.same_pairs = []
    self.diff_pairs = []
    if self.purpose == 'development':
      # 区分训练和测试
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
    else:
      # 不区分训练和测试
      with open(os.path.join(self.dir, 'pairs.txt')) as fp:
          for line in fp.readlines()[1:]:
              pair = line.strip().split()
              if len(pair) == 3:
                # same
                person_name, image_a, image_b = pair
                person_a_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_a))
                person_b_image = '%s/%s_%04d.jpg'%(person_name, person_name, int(image_b))
                self.same_pairs.append((person_a_image, person_b_image))
              else:
                # diff
                diff_person_a_name, diff_person_a_index, diff_person_b_name, diff_person_b_index = pair
                diff_person_a_image = '%s/%s_%04d.jpg' % (diff_person_a_name, diff_person_a_name, int(diff_person_a_index))
                diff_person_b_image = '%s/%s_%04d.jpg' % (diff_person_b_name, diff_person_b_name, int(diff_person_b_index))
                self.diff_pairs.append((diff_person_a_image, diff_person_b_image))

  @property
  def size(self):
    return len(self.same_pairs) + len(self.diff_pairs)

  def at(self, id):
    pair_i = id
    if pair_i < len(self.same_pairs):
      same_person_a, same_person_b = self.same_pairs[pair_i]
      same_person_a_path = os.path.join(self._data_folder, same_person_a)
      same_person_a_image = cv2.imread(same_person_a_path)
      same_person_b_path = os.path.join(self._data_folder, same_person_b)
      same_person_b_image = cv2.imread(same_person_b_path)
      
      cx = same_person_a_image.shape[1]/2
      cy = same_person_a_image.shape[0]/2
      left_t_x = int(cx - 64)
      left_t_y = int(cy - 64)
      
      right_b_x = int(cx + 64)
      right_b_y = int(cy + 64)
      
      crop_a_image = same_person_a_image[left_t_y:right_b_y, left_t_x:right_b_x]
      crop_b_image = same_person_b_image[left_t_y:right_b_y, left_t_x:right_b_x]
      return crop_a_image, {
        'issame': 1,
        'image_2': crop_b_image,
        'image_meta': {
          'image_shape': (crop_a_image.shape[0], crop_a_image.shape[1]),
          'image_file': same_person_a_path
        }
      }
    else:
      diff_person_a, diff_person_b = self.diff_pairs[pair_i-len(self.same_pairs)]
      diff_person_a_path = os.path.join(self._data_folder, diff_person_a)
      diff_person_a_image = cv2.imread(diff_person_a_path)
      diff_person_b_path = os.path.join(self._data_folder, diff_person_b)
      diff_person_b_image = cv2.imread(diff_person_b_path)
      
      cx = diff_person_a_image.shape[1]/2
      cy = diff_person_a_image.shape[0]/2
      left_t_x = int(cx - 64)
      left_t_y = int(cy - 64)
      
      right_b_x = int(cx + 64)
      right_b_y = int(cy + 64)
      
      crop_a_image = diff_person_a_image[left_t_y:right_b_y, left_t_x:right_b_x]
      crop_b_image = diff_person_b_image[left_t_y:right_b_y, left_t_x:right_b_x]      
      return crop_a_image, {
        'issame': 0,
        'image_2': crop_b_image,
        'image_meta': {
          'image_shape': (crop_a_image.shape[0], crop_a_image.shape[1]),
          'image_file': diff_person_a_path
        }
      }

# lfw = LFW('train', '/root/workspace/dataset/lfw', ext_params={'purpose': 'benchmark'})
# num = lfw.size
# for i in range(num):
#   data = lfw.sample(i)
#   # print(data.keys())
#   cv2.imwrite('./a1.png', data['image'][0])
#   cv2.imwrite('./b1.png', data['image'][1])
#   print(i)

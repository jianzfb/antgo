# -*- coding: UTF-8 -*-
# @Time : 2018/8/13
# @File : ade20k.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import *
import os
import numpy as np

__all__ = ['ADE20K']
class ADE20K(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(ADE20K, self).__init__(train_or_test, dir)

    assert(train_or_test in ['sample', 'train', 'val', 'test'])
    if train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    self._image_file_list = []
    self._annotation_file_list = []
    if train_or_test in ['train', 'val']:
      image_target_path = os.path.join(dir,
                                       'ADEChallengeData2016',
                                       'images',
                                       'training' if train_or_test == 'train' else 'validation')
      annotation_target_path = os.path.join(dir,
                                            'ADEChallengeData2016',
                                            'annotations',
                                            'training' if train_or_test == 'train' else 'validation')
      if not os.path.exists(image_target_path):
        logger.error('ADE dtaset must download from \n http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip')
        sys.exit(0)

      for file in os.listdir(image_target_path):
        if file[0] == '.':
          continue

        self._image_file_list.append(os.path.join(image_target_path, file))
        self._annotation_file_list.append(os.path.join(annotation_target_path, '%s.png'%file.split('.')[0]))
    else:
      parse_file = os.path.join(dir, 'release_test', 'list.txt')
      if not os.path.exists(parse_file):
        logger.error(
          'ADE dtaset must download from \n http://data.csail.mit.edu/places/ADEchallenge/release_test.zip')
        sys.exit(0)

      with open(parse_file) as fp:
        for file in fp.readlines():
          if len(file) == 0:
            continue

          self._image_file_list.append(os.path.join(dir, 'release_test', 'testing', file.replace('\n','')))

    self.ids = list(range(len(self._image_file_list)))

  @property
  def size(self):
    return len(self.ids)

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.copy(self.ids)
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

      for k in idxs:
        image_file = self._image_file_list[k]
        image = imread(image_file)

        if self.train_or_test in ['train', 'val']:
          label_file = self._annotation_file_list[k]
          label = imread(label_file)

          if len(label.shape) == 3:
            label = label[:,:,0]

          yield [image, {'segmentation_map': label, 'file_id': image_file.split('/')[-1]}]
        else:
          yield [image, {'file_id': image_file.split('/')[-1]}]

  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    image_file = self._image_file_list[id]
    image = imread(image_file)

    if self.train_or_test in ['train', 'val']:
      label_file = self._annotation_file_list[id]
      label = imread(label_file)

      if len(label.shape) == 3:
        label = label[:, :, 0]

      return [image, {'segmentation_map':label, 'file_id': image_file.split('/')[-1]}]
    else:
      return [image, {'file_id': image_file.split('/')[-1]}]

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert(split_method == 'holdout')

    validation_dataet = ADE20K('val', self.dir)
    return self, validation_dataet

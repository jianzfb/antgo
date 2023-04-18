# -*- coding: UTF-8 -*-
# @Time : 2018/8/13
# @File : ade20k.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import cv2
from antgo.dataflow.dataset import *
from antgo.framework.helper.fileio.file_client import *


__all__ = ['ADE20K']
class ADE20K(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(ADE20K, self).__init__(train_or_test, dir)
    assert(train_or_test in ['train', 'val', 'test'])

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

  def at(self, id):
    image_file = self._image_file_list[id]
    image = cv2.imread(image_file)

    if self.train_or_test in ['train', 'val']:
      label_file = self._annotation_file_list[id]
      label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)

      return (
        image, 
        {
          'segments':label, 
          'image_meta': {
            'image_shape': (image.shape[0], image.shape[1]),
            'image_file': image_file
          }
         }
      )
    else:
      return (
        image,
        {
          'image_meta': {
            'image_shape': (image.shape[0], image.shape[1]),
            'image_file': image_file
          }
         }
      )

  def split(self, split_params={}, split_method='holdout'):
    assert(self.train_or_test == 'train')
    assert(split_method == 'holdout')

    validation_dataet = ADE20K('val', self.dir)
    return self, validation_dataet

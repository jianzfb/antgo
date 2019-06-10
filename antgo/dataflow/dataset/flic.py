# -*- coding: UTF-8 -*-
# @Time    : 2019-06-08 15:51
# @File    : flic.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import scipy.io as io
from antgo.dataflow.dataset import *
import os
import numpy as np

__all__ = ['FLIC']
class FLIC(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(FLIC, self).__init__(train_or_test, dir)
    assert(train_or_test in ['train', 'test'])
    matr = io.loadmat(os.path.join(dir, 'examples.mat'))
    dataset = matr['examples'][0]

    self.ids = []
    self.files = []
    self.coords = []
    self.image_size = []
    self.torso = []

    self.valid_coords = [0,1,2,3,4,5,6,9,12,13,16]
    self.coords_name = ['Nose',
                        'Right Shoulder',
                        'Right Elbow',
                        'Right Wrist',
                        'Right Hip',
                        'Left Shoulder',
                        'Left Elbow',
                        'Left Wrist',
                        'Left Hip',
                        'Left Eye',
                        'Right Eye']
    count = 0
    for data in dataset:
      if train_or_test == 'train' and data[7][0][0]:
        # add train set
        self.ids.append(count)
        self.files.append(data[3][0])
        self.coords.append(data[2]) # 2 x 20
        self.image_size.append(data[4])
        self.torso.append(data[6])

        count += 1
      elif train_or_test == 'test' and data[8][0][0]:
        # add test set
        self.ids.append(count)
        self.files.append(data[3][0])
        self.coords.append(data[2]) # 2 x 20
        self.image_size.append(data[4])
        self.torso.append(data[6])

        count += 1

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
        img = imread(os.path.join(self.dir, 'images', self.files[k]))
        yield img, {'coords': self.coords[k][:,self.valid_coords],
                    'coords_name': self.coords_name,
                    'id': k,
                    'torso': self.torso[k]}

  def at(self, id):
    img = imread(os.path.join(self.dir, 'images', self.files[id]))
    return img, {'coords': self.coords[id][:,self.valid_coords],
                 'coords_name': self.coords_name,
                'id': id,
                'torso': self.torso[id]}

  def split(self, split_params={}, split_method=''):
    raise NotImplementedError
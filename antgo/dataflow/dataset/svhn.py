# -*- coding: UTF-8 -*-
# File: svhn.py
# Author: jian(jian@mltalker.com)
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import random
import numpy as np
from six.moves import range
import copy
from antgo.utils import logger, get_rng
from antgo.dataflow.dataset import Dataset
from antgo.utils.fs import download
from antgo.utils.fs import maybe_here
import scipy.io
__all__ = ['SVHNDigit']

SVHN_URL = "http://ufldl.stanford.edu/housenumbers/"
class SVHNDigit(Dataset):
  def __init__(self, name, dir=None,params=None):
    """
    :param name: 'train', 'test'
    :param data_dir: a directory containing the original {train,test}_32x32.mat
    """
    super(SVHNDigit, self).__init__(name, dir, params)

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    if name == "val":
        super(SVHNDigit,self).__init__("val",dir)
    else:
        super(SVHNDigit, self).__init__(name, dir)

    filename = ""
    if self.train_or_test == 'val':
      if not os.path.exists(os.path.join(self.dir, 'extra_32x32.mat')):
        self.download(self.dir, ['extra_32x32.mat'], default_url=SVHN_URL)

      filename = os.path.join(self.dir, 'extra' + '_32x32.mat')
    else:
      if not os.path.exists(os.path.join(self.dir, self.train_or_test + '_32x32.mat')):
        self.download(self.dir, [self.train_or_test + '_32x32.mat'], default_url=SVHN_URL)

      filename = os.path.join(self.dir, self.train_or_test + '_32x32.mat')
    assert os.path.isfile(filename), \
      "File {} not found! Please download it from {}.".format(filename, SVHN_URL)

    logger.info("Loading {} ...".format(filename))
    data = scipy.io.loadmat(filename)
    self.X = data['X'].transpose(3, 0, 1, 2)
    self.Y = data['y'].reshape((-1))
    self.Y[np.where(self.Y == 10)] = 0
    self.Y = self.Y.astype(np.uint8)

    self.ids = list(range(self.Y.shape[0]))

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')

    svhn_val = SVHNDigit('val', self.dir)
    return self, svhn_val

  @property
  def size(self):
    return len(self.ids)

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

      idxs = copy.deepcopy(self.ids)
      if self.rng:
          self.rng.shuffle(idxs)

      for k in idxs:
          yield [self.X[k], self.Y[k]]

  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    return [self.X[self.ids[id]], self.Y[self.ids[id]]]

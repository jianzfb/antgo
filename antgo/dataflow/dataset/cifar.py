# -*- coding: UTF-8 -*-
# @Time    : 17-12-27
# @File: cifar.py
# @Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals

import os, sys
import pickle
import numpy as np
import random
import six
from six.moves import range
from ...utils import logger
from ...utils.fs import download
from ...utils.fs import maybe_here
from .dataset import Dataset
import copy
import time
from filelock import FileLock

__all__ = ['Cifar10', 'Cifar100']
CIFAR_10_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
CIFAR_100_URL = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'


def read_cifar(filenames, cifar_classnum):
  assert cifar_classnum == 10 or cifar_classnum == 100
  ret = []
  for fname in filenames:
    fo = open(fname, 'rb')
    if six.PY3:
      dic = pickle.load(fo, encoding='bytes')
    else:
      dic = pickle.load(fo)
    data = dic[b'data']
    if cifar_classnum == 10:
      label = dic[b'labels']
      IMG_NUM = 10000 # cifar10 data are split into blocks of 10000
    elif cifar_classnum == 100:
      label = dic[b'fine_labels']
      IMG_NUM = 50000 if 'train' in fname else 10000
    fo.close()
    for k in range(IMG_NUM):
      img = data[k].reshape(3, 32, 32)
      img = np.transpose(img, [1, 2, 0])
      ret.append([img, label[k]])
  return ret


def get_filenames(dir, cifar_classnum):
  assert cifar_classnum == 10 or cifar_classnum == 100
  filenames = []
  if cifar_classnum == 10:
    filenames = [os.path.join(dir, 'cifar-10-batches-py', 'data_batch_%d' % i) for i in range(1, 6)]
    filenames.append(os.path.join(dir, 'cifar-10-batches-py', 'test_batch'))
  elif cifar_classnum == 100:
    filenames = [os.path.join(dir, 'cifar-100-python', 'train'),
                 os.path.join(dir, 'cifar-100-python', 'test')]
  return filenames


class CifarBase(Dataset):
  """
  Return [image, label],
      image is 32x32x3 in the range [0,255]
  """
  def __init__(self, train_or_test, dir=None, params=None, cifar_classnum=10):
    assert cifar_classnum == 10 or cifar_classnum == 100
    super(CifarBase,self).__init__(train_or_test,dir)
    self.train_or_test = train_or_test
    self.cifar_classnum = cifar_classnum
    self.dir = dir

    if not os.path.exists(self.dir):
      os.makedirs(self.dir)

    # fixed seed
    self.seed = time.time()

    """Download and extract the tarball from Alex's website."""
    cifar_foldername = ''
    if cifar_classnum == 10:
      cifar_foldername = 'cifar-10-batches-py'
    else:
      cifar_foldername = 'cifar-100-python'

    data_url = CIFAR_10_URL if cifar_classnum == 10 else CIFAR_100_URL
    lock = FileLock('DATASET.lock')
    with lock:
      if not os.path.exists(os.path.join(self.dir, cifar_foldername)):
        # 数据集不存在，需要重新下载，并创建标记
        self.download(self.dir, default_url=data_url, auto_untar=True, is_gz=True)

    fnames = get_filenames(self.dir, self.cifar_classnum)
    if self.train_or_test == 'train':
      self.fs = fnames[:-1]
    else:
      self.fs = [fnames[-1]]
    for f in self.fs:
      if not os.path.isfile(f):
        raise ValueError('Failed to Find File: ' + f)
    self.train_or_test = self.train_or_test
    self.data = read_cifar(self.fs, self.cifar_classnum)

    # unwarp
    self.image, self.label = zip(*self.data)

    # ids
    self.ids = [i for i in range(len(self.image))]

  @property
  def size(self):
    return len(self.ids)

  def get_cat_ids(self, idx):
    return self.label[idx]
  
  def get_ann_info(self, idx):
    return {'label': self.label[idx]}
  
  def at(self, id):
    return self.image[id], self.label[id]


class Cifar10(CifarBase):
  def __init__(self, train_or_test, dir=None, params=None):
    super(Cifar10, self).__init__(train_or_test, dir, params,10)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method in ['repeated-holdout', 'bootstrap', 'kfold'])

    category_ids = None
    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      category_ids = [i for i in range(len(self.ids))]
      np.random.shuffle(category_ids)
    else:
      category_ids = [self.label[i] for i in range(len(self.ids))]

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = Cifar10(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = Cifar10(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids

    return train_dataset, val_dataset  # split data by their label


class Cifar100(CifarBase):
  def __init__(self, train_or_test, dir=None, params=None):
    super(Cifar100, self).__init__(train_or_test, dir, params, 100)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method in ['repeated-holdout', 'bootstrap', 'kfold'])

    category_ids = None
    if split_method == 'kfold':
      np.random.seed(np.int64(self.seed))
      category_ids = [i for i in range(len(self.ids))]
      np.random.shuffle(category_ids)
    else:
      category_ids = [self.label[i] for i in range(len(self.ids))]

    train_ids, val_ids = self._split(category_ids, split_params, split_method)
    train_dataset = Cifar100(self.train_or_test, self.dir, self.ext_params)
    train_dataset.ids = train_ids

    val_dataset = Cifar100(self.train_or_test, self.dir, self.ext_params)
    val_dataset.ids = val_ids

    return train_dataset, val_dataset  # split data by their label

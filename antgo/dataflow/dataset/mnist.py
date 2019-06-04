# -*- coding: UTF-8 -*-
# File: mnist.py
# Author: jian(jian@mltalker.com)
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import gzip
import random
import numpy as np
from ...utils import logger
from .dataset import Dataset
import time
import copy

__all__ = ['Mnist']
MINIST_URL = 'http://yann.lecun.com/exdb/mnist/'

def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
         'Invalid magic number %d in MNIST image file: %s' %
         (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data

def extract_labels(filename):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  with gzip.open(filename) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
        raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels

class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    """Construct a DataSet. """
    assert images.shape[0] == labels.shape[0], (
        'images.shape: %s labels.shape: %s' % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    images = images.astype(np.float32)
    images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples


class Mnist(Dataset):
  """
  Return [image, label],
      image is 28x28 in the range [0,1]
  """
  def __init__(self, train_or_test, dir=None, params=None):
    """
    Args:
        train_or_test: string either 'train' or 'test'
    """
    super(Mnist,self).__init__(train_or_test, dir, params)
    assert(train_or_test != 'val')
    self.train_or_test = train_or_test

    # read sample data
    if self.train_or_test == 'sample':
      self.data_samples, self.ids = self.load_samples()
      return

    # fixed seed
    self.seed = time.time()

    if self.train_or_test == "train":
      self.download(self.dir, file_names=['train-images-idx3-ubyte.gz'], default_url=MINIST_URL)
      self.train_images = extract_images(os.path.join(self.dir, 'train-images-idx3-ubyte.gz'))

      self.download(self.dir, file_names=['train-labels-idx1-ubyte.gz'])
      self.train_labels = extract_labels(os.path.join(self.dir, 'train-labels-idx1-ubyte.gz'))
      self.train = DataSet(self.train_images, self.train_labels)

      self.ids = [i for i in range(self.train.num_examples)]
    else:
      self.download(self.dir, file_names=['t10k-images-idx3-ubyte.gz'], default_url=MINIST_URL)
      test_images = extract_images(os.path.join(self.dir,'t10k-images-idx3-ubyte.gz'))

      self.download(self.dir, file_names=['t10k-labels-idx1-ubyte.gz'])
      test_labels = extract_labels(os.path.join(self.dir, 't10k-labels-idx1-ubyte.gz'))
      self.test = DataSet(test_images, test_labels)

      self.ids = [i for i in range(self.test.num_examples)]

  def split(self, split_params={}, split_method='holdout'):
      assert(self.train_or_test == 'train')
      assert(split_method == 'holdout')
      val_dataset = Mnist('test', self.dir, self.ext_params)
      return self, val_dataset

  @property
  def size(self):
    return len(self.ids)

  def at(self, id):
    if self.train_or_test == 'sample':
      return self.data_samples[id]

    if self.train_or_test == 'train':
      img = self.train.images[id].reshape((28, 28))
      label = self.train.labels[id]

      return img, {'id': id, 'category_id': label}
    else:
      img = self.test.images[id].reshape((28, 28))
      label = self.test.labels[id]

      return img, {'id': id, 'category_id': label}

  def data_pool(self):
    if self.train_or_test == 'sample':
      sample_idxs = copy.deepcopy(self.ids)
      if self.rng:
        self.rng.shuffle(sample_idxs)

      for index in sample_idxs:
        yield self.data_samples[index]
      return

    ds = None
    if self.train_or_test == 'train':
      ds = self.train
    else:
      ds = self.test

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
        img = ds.images[id].reshape((28, 28, 1))
        label = ds.labels[id]

        yield img, {'id': id, 'category_id': label}

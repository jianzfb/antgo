# -*- coding: UTF-8 -*-
# Time: 12/2/17
# File: tfdataset.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import Dataset
import tensorflow as tf
import os
import sys


class TFRecordsReader(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(TFRecordsReader, self).__init__(train_or_test, dir, params)
    self._batch_size = getattr(self, '_batch_size', 1)
    self._capacity = getattr(self, '_capacity', 2)
    self._min_after_dequeue = getattr(self, '_min_after_dequeue', 1)
    
    self._data_size = None
    self._data_type = None
    self._label_size = None
    self._label_type = None
    self._num_threads = getattr(self, '_num_threads', 2)
    self._num_samples = getattr(self, '_num_samples', 100)
    
    self._pattern = getattr(self, '_pattern', '*.tfrecords')
    self._sess = None
  
  @property
  def batch_size(self):
    return self._batch_size
  @batch_size.setter
  def batch_size(self, val):
    self._batch_size = val
  
  @property
  def capacity(self):
    return self._capacity
  @capacity.setter
  def capacity(self, val):
    self._capacity = val
  
  @property
  def min_after_dequeue(self):
    return self._min_after_dequeue
  @min_after_dequeue.setter
  def min_after_dequeue(self, val):
    self._min_after_dequeue = val
  
  @property
  def data_size(self):
    return self._data_size
  @data_size.setter
  def data_size(self, val):
    self._data_size = val
  
  @property
  def label_size(self):
    return self._label_size
  @label_size.setter
  def label_size(self, val):
    self._label_size = val
  
  @property
  def data_type(self):
    return self._data_type
  @data_type.setter
  def data_type(self, val):
    self._data_type = val
  
  @property
  def label_type(self):
    return self._label_type
  @label_type.setter
  def label_type(self, val):
    self._label_type = val
  
  @property
  def num_threads(self):
    return self._num_threads
  @num_threads.setter
  def num_threads(self, val):
    self._num_threads = val
  
  @property
  def num_samples(self):
    return self._num_samples
  @num_samples.setter
  def num_samples(self, val):
    self._num_samples = val
  
  @property
  def file_pattern(self):
    return self._pattern
  @file_pattern.setter
  def file_pattern(self,val):
    self._pattern = val

  def data_pool(self):
    self.epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if self.epoch >= max_epoches:
        break
      self.epoch += 1
      
      try:
        for _ in range(self.num_samples):
          a, b = self._sess.run([self.data_tensor, self.label_tensor])
          yield a, b
      except:
        error_info = sys.exc_info()
        print(error_info)
        
  def at(self, id):
    raise NotImplementedError

  def split(self, split_params={}, split_method=""):
    raise NotImplementedError
  
  @property
  def size(self):
    return self.num_samples
    
  def init(self, *args, **kwargs):
    if 'sess' in kwargs:
      self._sess = kwargs['sess']

    file_names = tf.train.match_filenames_once(os.path.join(self.dir, self.train_or_test, self.file_pattern))
    filename_queue = tf.train.string_input_producer(file_names)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                          'image': tf.FixedLenFeature([], tf.string),
                                          'label': tf.FixedLenFeature([], tf.string),
                                       })

    image = tf.decode_raw(features['image'], self.data_type)
    image = tf.reshape(image, self.data_size)

    label = tf.decode_raw(features['label'], self.label_type)
    label = tf.reshape(label, self.label_size)

    self.data_tensor = image
    self.label_tensor = label

    if self.batch_size > 0:
      image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                        batch_size=self.batch_size,
                                                        capacity=self.capacity,
                                                        min_after_dequeue=self.min_after_dequeue,
                                                        num_threads=self.num_threads)
  
      self.data_tensor = image_batch
      self.label_tensor = label_batch

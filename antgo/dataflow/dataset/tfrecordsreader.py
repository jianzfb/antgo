# -*- coding: UTF-8 -*-
# Time: 12/2/17
# File: tfdataset.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset import Dataset
import tensorflow as tf
slim = tf.contrib.slim
import os
import sys


class TFRecordsReader(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(TFRecordsReader, self).__init__(train_or_test, dir, params)
    assert(train_or_test in ['train', 'val', 'test'])
    temp = getattr(self, '_data_size', '700,700,3')
    self._data_size = [int(s) for s in temp.split(',')]
    self._data_type = tf.as_dtype(getattr(self, '_data_type', 'uint8'))
    temp = getattr(self, '_label_size', '700,700,1')
    self._label_size = [int(s) for s in temp.split(',')]
    self._label_type = tf.as_dtype(getattr(self, '_label_type', 'uint8'))
    self._num_samples = int(getattr(self, '_num_samples', 199600))
    self._pattern = getattr(self, '_pattern', '*.tfrecord?')
    self._has_format = bool(getattr(self, '_format', False))
  
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
  
  @property
  def has_format(self):
    return self._has_format
  @has_format.setter
  def has_format(self, val):
    self._has_format = val
  
  def data_pool(self):
    raise NotImplementedError
  
  def at(self, id):
    raise NotImplementedError

  def split(self, split_params={}, split_method=""):
    raise NotImplementedError
  
  @property
  def size(self):
    return self.num_samples
    
  def model_fn(self, *args, **kwargs):
    # 1.step candidate data file list
    file_names = tf.train.match_filenames_once(os.path.join(self.dir, self.train_or_test, self.file_pattern))
    # dd= tf.get_default_session().run(file_names)
    
    # 2.step shuffle data file list
    filename_queue = tf.train.string_input_producer(file_names, shuffle= self.train_or_test == "train")

    # 3.step read from data file
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    
    # 4.step parse data
    if self.has_format:
      keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format': tf.FixedLenFeature((), tf.string, default_value='png'),
      }
      items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format', channels=self.data_size[-1]),
        'label': slim.tfexample_decoder.Image('label/encoded', 'label/format', channels=self.label_size[-1])
      }
  
      decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
      serialized_example = tf.reshape(serialized_example, shape=[])
      image, label = decoder.decode(serialized_example, ['image', 'label'])
      return image, label
    else:
      features = tf.parse_single_example(serialized_example,
                                         features={
                                            'image': tf.FixedLenFeature([], tf.string),
                                            'label': tf.FixedLenFeature([], tf.string),
                                         })
  
      image = tf.decode_raw(features['image'], self.data_type)
      if self.data_size is not None:
        image = tf.reshape(image, self.data_size)
  
      label = tf.decode_raw(features['label'], self.label_type)
      if self.label_size is not None:
        label = tf.reshape(label, self.label_size)
      
      return image, label
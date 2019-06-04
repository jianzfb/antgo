# -*- coding: UTF-8 -*-
# @Time    : 18-1-3
# @File    : dataset.py
# @Author  : Jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import threading
import time
import numpy as np
import tensorflow as tf
from antgo.utils import logger


class TFQueueDataset(object):
  def __init__(self,
               datasource,
               dtype=[tf.uint8],
               shape=[[None, None, None]],
               max_queue_size=32,
               wait_time=0.01):
    # Change the shape of the input data here with the parameter shapes.
    self.wait_time = wait_time
    self.max_queue_size = max_queue_size
    self.dtype_list = dtype
    self.shape_list = shape

    self.threads = []
    self._coord = None
    self.sample_placeholder = []
    self.datasource = datasource
    self._size = None
  
  @property
  def size(self):
    if self._size is None:
      return self.datasource.size
    else:
      return self._size

  @size.setter
  def size(self, val):
    self._size = val

  @property
  def coord(self):
    return self._coord

  @coord.setter
  def coord(self, val):
    self._coord = val
    
  def dequeue(self):
    return self.model_fn()

  def model_fn(self):
    self.queue = tf.PaddingFIFOQueue(self.max_queue_size, self.dtype_list, shapes=self.shape_list)
    self.queue_size = self.queue.size()
    for i in range(len(self.dtype_list)):
      self.sample_placeholder.append(tf.placeholder(dtype=self.dtype_list[i], shape=None))
    self.enqueue = self.queue.enqueue(self.sample_placeholder)
    tf.add_to_collection('CUSTOM_DATASET_QUEUE', self)
    
    return self.queue.dequeue()
  
  def thread_main(self, sess):
    stop = False
    while not stop:
      self.datasource._reset_iteration_state()
      iterator = self.datasource.iterator_value()
      
      for data in iterator:
        try:
          while self.queue_size.eval(session=sess) == self.max_queue_size:
            if self.coord.should_stop():
              self.queue.close()
              break
            time.sleep(self.wait_time)

          if self.coord.should_stop():
            stop = True
            self.queue.close()
            break
          feed_dict = {}
          if type(data) == list or type(data) == tuple:
            for i in range(len(data)):
              feed_dict.update({self.sample_placeholder[i]: data[i]})
          else:
            feed_dict = {self.sample_placeholder[0]: data}
          sess.run(self.enqueue, feed_dict=feed_dict)
        except:
          logger.error('couldnt feed data into tensorflow pipeline')
          pass
        
  def start_threads(self, sess):
    for _ in range(1):
      thread = threading.Thread(target=self.thread_main, args=(sess,))
      thread.daemon = True  # Thread will close when parent quits.
      thread.start()
      self.threads.append(thread)
    return self.threads
# -*- coding: UTF-8 -*-
# @Time : 2018/4/28
# @File : queue_dataset.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

from antgo.dataflow.dataset.dataset import Dataset
import os
import sys
from antgo.utils.serialize import *
from PIL import Image
import imageio
import numpy as np


class QueueDataset(Dataset):
  def __init__(self, queue=None):
    super(QueueDataset, self).__init__('test', '', None)
    self.queue = queue

  @property
  def size(self):
    return sys.maxsize

  def at(self, id):
    raise NotImplementedError

  def _video_iterator(self, video_path, request_param):
    reader = imageio.get_reader(video_path)
    for im in reader:
      img_data = np.fromstring(im.tobytes(), dtype=np.uint8)
      img_data = img_data.reshape((im.shape[0], im.shape[1], im.shape[2]))
      yield img_data if request_param is None else (img_data, request_param)

  def _parse_data(self, data, data_type, request_param):
    if data_type == 'VIDEO':
      return self._video_iterator(data, request_param)
    elif data_type == 'IMAGE':
      image_data = Image.open(data)
      img_data = np.fromstring(image_data.tobytes(), dtype=np.uint8)
      img_data = img_data.reshape((image_data.size[1], image_data.size[0], len(image_data.getbands())))
      return img_data if request_param is None else (img_data, request_param)
    elif data_type == 'FILE':
      with open(data, 'r') as fp:
        return fp.read() if request_param is None else (fp.read(), request_param)
    elif data_type == 'STRING':
      return data if request_param is None else (data, request_param)
    elif data_type == 'JSON':
      return data if request_param is None else (data, request_param)
    else:
      return None

  def data_pool(self):
    while True:
      data_pack = self.queue.get()
      if type(data_pack) == list and \
              (type(data_pack[0]) == list or type(data_pack[0]) == tuple):
        # multi-data
        storage = []
        for dd in data_pack:
          data, data_type, request_param = dd
          response_data = self._parse_data(data,data_type,request_param)
          storage.append(response_data)
        yield storage
      else:
        # single-data
        data, data_type, request_param = data_pack
        yield self._parse_data(data, data_type, request_param)
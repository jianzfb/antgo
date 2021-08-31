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
from io import BytesIO
import numpy as np
import cv2

class QueueDataset(Dataset):
  def __init__(self, queue=None):
    super(QueueDataset, self).__init__('test', '', None)
    self.queue = queue
    self.annotation = {}

  @property
  def size(self):
    return sys.maxsize

  def at(self, id):
    raise NotImplementedError

  def now(self):
    return self.annotation

  def _parse_data(self, data, data_type, request_param):
    assert(data_type not in ['VIDEO'])
    try:
      if data_type == 'IMAGE':
        img_data = cv2.imread(data)
        return (img_data,{}) if request_param is None else (img_data, request_param)
      if data_type == 'IMAGE_MEMORY':
        img_data = cv2.imread(BytesIO(data))
        return (img_data,{}) if request_param is None else (img_data, request_param)
      elif data_type == 'FILE':
        with open(data, 'r') as fp:
          return (fp.read(),{}) if request_param is None else (fp.read(), request_param)
      elif data_type == 'STRING':
        return (data,{}) if request_param is None else (data, request_param)
      elif data_type == 'JSON':
        return (data,{}) if request_param is None else (data, request_param)
      else:
        return (None,{})
    except:
      return None

  def data_pool(self):
    while True:
      data_pack = self.queue.get()
      if type(data_pack) == list and \
              (type(data_pack[0]) == list or type(data_pack[0]) == tuple):
        # multi-data
        for single_pack in data_pack:
          data, data_type, request_param = single_pack
          response_data = self._parse_data(data, data_type, request_param)
          if response_data is None:
            # 解析数据有误，不进行处理
            continue

          # annotation
          self.annotation = response_data[1]
          yield response_data
      else:
        # single-data
        data, data_type, request_param = data_pack
        if data_type == 'VIDEO':
          # 解析视频文件
          try:
            cap = cv2.VideoCapture(data)

            if request_param is None:
              request_param = {}

            # 视频fps
            fps = cap.get(5)
            request_param.update({'fps': fps})

            # 视频帧数
            frame_num=cap.get(7)
            request_param.update({'frame_num': (int)(frame_num)})

            frame_index = 0       
            while True:
              ret, frame = cap.read()
              if not ret:
                break
              
              request_param.update({'frame_index': frame_index})
              self.annotation = request_param
              yield (frame, {}) if request_param is None else (frame, request_param)
              frame_index += 1              
              if frame_index == (int)(frame_num):
                break
          except:
            # 解析视频有误，不进行处理
            continue
        else:
          response_data = self._parse_data(data, data_type, request_param)
          if response_data is None:
            # 解析数据有误，不进行处理
            continue

          # annotation
          self.annotation = response_data[1]
          yield response_data
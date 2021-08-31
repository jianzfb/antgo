# -*- coding: UTF-8 -*-
# @Time    : 2019-04-30 18:32
# @File    : simplevideos.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from antgo.dataflow.dataset.dataset import *
import os
import copy
from PIL import Image
import cv2

__all__ = ['SimpleVideos']


class SimpleVideos(Dataset):
  def __init__(self, train_or_test, dir=None, params=None):
    super(SimpleVideos, self).__init__(train_or_test, dir, params)

    self.data_files = []
    for data_file in os.listdir(os.path.join(self.dir, self.train_or_test)):
      if data_file.lower().split('.')[-1] in ['mp4', 'mov', 'avi']:
        file_path = os.path.join(self.dir, self.train_or_test, data_file)
        self.data_files.append(file_path)

    assert (len(self.data_files) > 0)
    self.ids = [i for i in range(len(self.data_files))]

  def data_pool(self):
    epoch = 0
    while True:
      max_epoches = self.epochs if self.epochs is not None else 1
      if epoch >= max_epoches:
        break
      epoch += 1

      ids = copy.copy(self.ids)
      if self.rng:
        self.rng.shuffle(ids)

      for id in ids:
        video_path = self.data_files[id]

        # read
        cap = cv2.VideoCapture(video_path)
        params = {}
        # 加入fps
        fps = cap.get(5)
        params.update({'fps': fps})
        # 加入帧数
        frame_num = (int)(cap.get(7))
        params.update({'frame_num': frame_num})

        frame_index = 0       
        while True:
          ret, frame = cap.read()
          if not ret:
            break
          
          params.update({'frame_index': frame_index})
          yield frame, params
          frame_index += 1              
          if frame_index == (int)(frame_num):
            break

  def at(self, id):
    raise NotImplementedError

  @property
  def size(self):
    return len(self.ids)

  def split(self, split_params={}, split_method='holdout'):
    assert (self.train_or_test == 'train')
    assert (split_method == 'holdout')

    return self, SimpleVideos('val', self.dir, self.ext_params)

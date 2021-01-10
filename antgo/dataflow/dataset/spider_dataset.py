# -*- coding: UTF-8 -*-
# @Time    : 2020/10/26 10:26 上午
# @File    : spider_dataset.py
# @Author  : jian<jian@mltalker.com>
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
from antgo.ant.utils import *


class SpiderDataset(Dataset):
  def __init__(self, queue=None):
    super(SpiderDataset, self).__init__('test', '', None)
    self.queue = queue

  @property
  def size(self):
    return sys.maxsize

  def at(self, id):
    raise NotImplementedError

  def __spider_data_source(self, config):
    datasource_address = config['datasource_address']
    datasource_keywards = config['datasource_keywards']

    # 调用爬虫获取图像列表

    return []

  def data_pool(self):
    while True:
      try:
        data_pack = self.queue.get()
        data_source, config = data_pack
        data = []
        if data_source == 'spider':
          data = self.__spider_data_source(config)

        for image_file in data:
          image_data = Image.open(image_file)
          img_data = np.fromstring(image_data.tobytes(), dtype=np.uint8)
          img_data = img_data.reshape((image_data.size[1], image_data.size[0], len(image_data.getbands())))
          yield image_data, {}
      except:
        logger.error('fail receive data info')

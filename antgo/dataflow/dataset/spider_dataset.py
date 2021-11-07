# -*- coding: UTF-8 -*-
# @Time    : 2020/10/26 10:26 上午
# @File    : spider_dataset.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from uuid import uuid4
from antgo.dataflow.dataset.dataset import Dataset
from antgo.ant.utils import *
from antgo.utils import logger
from bs4 import BeautifulSoup
import os
import sys
import cv2
import numpy as nps
import threading
import requests
import re
import uuid
from antgo.ant.download import *
try:
    import queue
except:
    import Queue as queue


class SpiderDataset(Dataset):
  def __init__(self, command_queue=None, dir=None, params=None):
    super(SpiderDataset, self).__init__('test', dir, params)
    self.command_queue = command_queue
    self.waiting_process_queue = queue.Queue()
    self.count = 0
    self.keywords = {}


  def __spider_data_source(self, config):
    datasource_address = config['datasource_address']
    datasource_keywards = config['datasource_keywards']
    dsatasource_params = {}
    if 'datasource_params' in config:
      dsatasource_params = config['datasource_params']
    dsatasource_params.update({
      'download_data_type': 'image'
    })

    if datasource_address not in self.keywords:
      self.keywords[datasource_address] = []
    
    if datasource_keywards in self.keywords[datasource_address]:
      logger.error("Duplicate keywords.")
      return
    self.keywords[datasource_address].append(datasource_keywards)

    if datasource_address == 'baidu':
      baidu_download(datasource_keywards, dsatasource_params, self.dir, self.waiting_process_queue)
    elif datasource_address == 'bing':
      bing_download(datasource_keywards, dsatasource_params, self.dir, self.waiting_process_queue)
    elif datasource_address == 'google':
      google_download(datasource_keywards, dsatasource_params, self.dir, self.waiting_process_queue)
    elif datasource_address == 'vcg':
      vcg_download(datasource_keywards, dsatasource_params, self.dir, self.waiting_process_queue)

  def data_pool(self):
    self.count = 0
    while True:
      try:
        # 1.step 接受爬虫目标指令
        data_pack = self.command_queue.get()
        _, config = data_pack
        self.__spider_data_source(config)

        # 2.step 读取等待处理文件
        image_file = self.waiting_process_queue.get()
        if image_file is None:
          continue

        while True:
          try:
            # read image
            image = cv2.imread(image_file)
            if image is None:
              logger.error("Fail to parse %s"%image_file)
              # get next image file
              image_file = self.waiting_process_queue.get()
              if image_file is None:
                break

              continue
          except:
            logger.error("Fail to parse %s"%image_file)
            # get next image file
            image_file = self.waiting_process_queue.get()
            if image_file is None:
              break

            continue
          
          # increment 1
          self.count += 1

          # return data
          yield image, {}

          # get next image file
          image_file = self.waiting_process_queue.get()
          if image_file is None:
            break
      except:
        logger.error('Fail receive data info.')

  def finish_process_num(self):
    return self.count

  def waiting_process_num(self):
    return self.waiting_process_queue.qsize()

  def at(self, id):
    # 忽略 id
    self.count = 0
    while True:
      try:
        # 1.step 接受爬虫目标指令
        data_pack = self.command_queue.get()
        _, config = data_pack
        self.__spider_data_source(config)

        # 2.step 读取等待处理文件
        image_file = self.waiting_process_queue.get()
        if image_file is None:
          continue

        while True:
          try:
            # read image
            image = cv2.imread(image_file)
            if image is None:
              logger.error("Fail to parse %s" % image_file)
              # get next image file
              image_file = self.waiting_process_queue.get()
              if image_file is None:
                break

              continue
          except:
            logger.error("Fail to parse %s" % image_file)
            # get next image file
            image_file = self.waiting_process_queue.get()
            if image_file is None:
              break

            continue

          # increment 1
          self.count += 1

          # return data
          return image, {}
      except:
        logger.error('Fail receive data info.')

  def split(self, split_params={}, split_method='holdout'):
    raise NotImplemented

  @property
  def size(self):
    return None

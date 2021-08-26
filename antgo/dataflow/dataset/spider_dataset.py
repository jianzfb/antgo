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
import os
import sys
import cv2
import numpy as nps
import threading
import requests
import re
import uuid
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

  def __baidu_download(self, keyward):
    headers = {
        'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
        'Connection': 'keep-alive',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
        'Upgrade-Insecure-Requests': '1'
    }
    A = requests.Session()
    A.headers = headers
    url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + keyward + '&pn='

    def __find_and_download(waiting_process_queue, search_url, session, dir, max_page_num=100):
        t = 0
        num = 0
        while t < max_page_num:
            Url = search_url + str(t)
            t = t+1
            try:
                Result = session.get(Url, timeout=7, allow_redirects=False)
            except BaseException:
                t = t + 60
                continue
            else:
                pic_url = re.findall('"objURL":"(.*?)",', Result.text, re.S)  # 先利用正则表达式找到图片url
                for each in pic_url:
                  print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
                  try:
                      if each is not None:
                          pic = requests.get(each, timeout=7)
                      else:
                          continue
                  except BaseException:
                      print('错误，当前图片无法下载')
                      continue
                  else:
                      # 分配唯一文件标识
                      file_folder = os.path.join(dir, 'test')
                      if not os.path.exists(file_folder):
                        os.makedirs(file_folder)
                      
                      file_path = os.path.join(file_folder, '%s.jpg'%str(uuid.uuid4()))
                      with open(file_path, 'wb') as fp:
                        fp.write(pic.content)
                      num += 1

                      # 加入等待处理队列
                      waiting_process_queue.put(file_path)

    # 搜索和下载
    t = threading.Thread(target=__find_and_download, args=(self.waiting_process_queue, url, A, self.dir))
    t.start()

  def __bing_download(self, keyward):
    pass

  def __google_download(self, keyward):
    pass

  def __spider_data_source(self, config):
    datasource_address = config['datasource_address']
    datasource_keywards = config['datasource_keywards']

    if datasource_address == 'baidu':
      self.__baidu_download(datasource_keywards)
    elif datasource_address == 'bing':
      self.__bing_download(datasource_keywards)
    elif datasource_address == 'google':
      self.__google_download(datasource_keywards)

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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image is None:
              logger.error("Fail to parse %s"%image_file)
              # get next image file
              image_file = self.waiting_process_queue.get()
              continue

          except:
            logger.error("Fail to parse %s"%image_file)
            # get next image file
            image_file = self.waiting_process_queue.get()
            continue
          
          # return data
          yield image, {}

          # increment 1
          self.count += 1

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
    raise NotImplemented
  
  def split(self, split_params={}, split_method='holdout'):
    raise NotImplemented

  @property
  def size(self):
    return 10000000

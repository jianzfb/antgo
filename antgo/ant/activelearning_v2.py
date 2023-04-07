# -*- coding: UTF-8 -*-
# @Time    : 2022/9/3 00:54
# @File    : activelearning_v2.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import time
import cv2
from antgo.resource.html import *
from antgo.ant.base import *
from antgo.ant.base import _pick_idle_port
from antgo.dataflow.common import *
from antgo.dataflow.recorder import *
from antvis.client.httprpc import *
from multiprocessing import Process
from antgo.task.task import *
from antgo.crowdsource.label_server import *
import traceback
import subprocess
import os
import socket
import requests
import json
import zipfile
import imagesize


class ActivelearningDataRecorder(object):
  def __init__(self, rpc, dump_dir):
    self.rpc = rpc
    self.activelearning_static_dir = os.path.join(dump_dir, 'static')

  def record(self, sample):
    if not os.path.exists(os.path.join(self.activelearning_static_dir, 'data')):
      os.makedirs(os.path.join(self.activelearning_static_dir, 'data'))

    assert('image_file' in sample or 'id' in sample)
    image_file = ''
    image_height = 0
    image_width = 0
    if 'image_file' in sample:
      # 将文件上传至目录
      shutil.copy(sample['image_file'], os.path.join(self.activelearning_static_dir, 'data'))
      image_file = sample['image_file']
      image_file = f'/static/data/{image_file}'
      image_width, image_height = imagesize.get(sample['image_file'])
    else:
      data = sample['image']
      if len(data.shape) == 3:
        assert (data.shape[2] == 3 or data.shape[2] == 4)

      assert (len(data.shape) == 2 or len(data.shape) == 3)

      if data.dtype != np.uint8:
        data_min = np.min(data)
        data_max = np.max(data)
        data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)

      sample.pop('image')
      sample_id = sample['id']
      cv2.imwrite(os.path.join(self.activelearning_static_dir, 'data', f'{sample_id}.png'), data)
      image_file = f'/static/data/{sample_id}.png'
      image_height, image_width = data.shape[:2]

    data = {}
    data['image_file'] = image_file
    data['label_info'] = sample.get('label_info', [])
    data['width'] = image_width
    data['height'] = image_height
    samples = json.dumps([data])
    # 更新待标注样本
    self.rpc.label.sample.fresh.post(samples=samples)

  def close(self):
    pass


class AntActiveLearningV2(AntBase):
  def __init__(self,
               ant_context,
               ant_name,
               data_factory,
               ant_dump_dir,
               ant_token,
               ant_task_config=None,
               **kwargs):
    super(AntActiveLearningV2, self).__init__(ant_name, ant_context, ant_token, **kwargs)
    self.ant_data_source = data_factory
    self.ant_dump_dir = ant_dump_dir
    self.ant_context.ant = self
    self.ant_task_config = ant_task_config
    self.host_ip = self.context.params.system.get('ip', '127.0.0.1')
    self.host_port = self.context.params.system.get('port', -1)
    self.rpc = None
    self._running_dataset = None
    self._running_task = None
    self.p = None
    self._round = -1

  @property
  def running_dataset(self):
    return self._running_dataset

  @property
  def running_task(self):
      return self._running_task

  def ping_until_ok(self):
    while True:
      content = self.rpc.ping.get()
      if content['status'] != 'ERROR':
        break
      # 暂停5秒钟，再进行尝试
      time.sleep(5)

  def wait_until_stop(self):
    self.p.join()

  def labeling(self):
    # 启动新一轮标注
    response = self.rpc.info.get()
    if response['content']['project_state']['stage'] == 'labeling':
      logger.error(f'Label Round {self._round} is not finish.')
      return

    self._round += 1
    self.rpc.info.post(running_state='running',  running_stage='labeling', running_round=self._round)

  def waiting(self):
    # 完成本轮标注，后台会清空标注信息
    self.rpc.info.post(running_state='running',  running_stage='waiting', running_round=self._round)
    
  def state(self):
    response = self.rpc.info.get()
    return response['content']['project_state']   

  def download(self, waiting_finish=True):
    if waiting_finish:
      # 需要等待本轮标准完成
      while True:
        response = self.rpc.info.get()
        if response['status'] == 'ERROR':
          print('rpc error...')
          time.sleep(5)
          continue

        if response['content']['project_state']['stage'] == 'finish':
          break
        # 等待10分钟后检查
        time.sleep(10)
        
    if not os.path.exists(os.path.join(self.ant_dump_dir, 'label', f'{self._round}')):
      os.makedirs(os.path.join(self.ant_dump_dir, 'label', f'{self._round}'))

    folder = os.path.join(self.ant_dump_dir, 'label', f'{self._round}')
    file_name = f'label.json'
    self.rpc.label.export.download(
      file_folder=folder,
      file_name=file_name
    )    
    with open(os.path.join(folder, file_name), 'r') as fp:
      content = json.load(fp)

      return content

  def start(self, **kwargs):
    # 0.step loading challenge task
    running_ant_task = None
    if self.token is not None:
      # 0.step load challenge task
      response = self.context.dashboard.challenge.get(command=type(self).__name__)
      if response['status'] is None:
        # invalid token
        logger.error('Couldnt load challenge task.')
        self.token = None
      elif response['status'] == 'SUSPEND':
        # prohibit submit challenge task frequently
        # submit only one in one week
        logger.error('Prohibit submit challenge task frequently.')
        exit(-1)
      elif response['status'] == 'OK':
        # maybe user token or task token
        content = response['content']
        if 'task' in content:
          challenge_task = create_task_from_json(content)
          if challenge_task is None:
            logger.error('Couldnt load challenge task.')
            exit(-1)
          running_ant_task = challenge_task
      else:
        # unknow error
        logger.error('Unknow error.')
        exit(-1)

    if running_ant_task is None:
      # 0.step load custom task
      custom_task = create_task_from_xml(self.ant_task_config, self.context)
      if custom_task is None:
        logger.error('Couldnt load custom task.')
        exit(0)
      running_ant_task = custom_task

    assert (running_ant_task is not None)
    self._running_task = running_ant_task

    #获得实验ID
    experiment_uuid = self.context.experiment_uuid
    logger.info('Build experiment folder.')
    if not os.path.exists(os.path.join(self.ant_dump_dir, experiment_uuid)):
      os.makedirs(os.path.join(self.ant_dump_dir, experiment_uuid))

    is_launch_web_server = True
    if self.host_port < 0:
      is_launch_web_server = False

    self.rpc = HttpRpc("v1", "antgo/api", self.host_ip, self.host_port)
    self.context.recorder = \
      ActivelearningDataRecorder(
        self.rpc,
        os.path.join(self.ant_dump_dir, 'activelearning')
      )

    if is_launch_web_server:
      # 准备web服务资源
      static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
      activelearning_static_dir = os.path.join(self.ant_dump_dir, 'activelearning', 'static')
      if os.path.exists(activelearning_static_dir):
        shutil.rmtree(activelearning_static_dir)

      shutil.copytree(os.path.join(static_folder, 'resource', 'app'), activelearning_static_dir)

      # 准备有效端口
      self.host_port = _pick_idle_port(self.host_port)

      # 在独立进程中启动webserver
      sample_metas = {'filters': []}
      metas = self.context.params.activelearning.metas.get()
      white_users = self.context.params.activelearning.white_users.get() \
        if self.context.params.activelearning.white_users is not None else None
      task_metas = {
        'task_type': self.running_task.task_type,
        'task_name': self.running_task.task_name,
        'label_type': self.context.params.activelearning.label_type
      }
      
      data_json_file = kwargs.get('json_file', None)
      sample_folder = None
      sample_list = []
      if data_json_file is not None:
        # 直接使用来自于data_json_file中的样本
        with open(data_json_file, 'r') as fp:
          sample_list = json.load(fp)      
        sample_folder = os.path.dirname(data_json_file)
        
      self.p = \
        multiprocessing.Process(
          target=label_server_start,
          args=(os.path.join(self.ant_dump_dir, 'activelearning'),
                self.host_port,
                task_metas,
                sample_metas,
                metas,
                sample_folder,
                sample_list,
                {
                  'state': 'running', 
                  'stage': 'waiting'    # 标注服务启动后，处在等待状态。之后数据推送好后，重制状态使其处在标注状态
                },
                white_users)
        )
      self.p.daemon = True  # 主进程结束后，自进程也将结束
      self.p.start()

      # 等待直到http服务开启
      self.ping_until_ok()

    if self.context.is_interact_mode:
      logger.info('Running on interact mode.')
      return

    # TODO，
    # 对于非交互模式，设置训练->测试->采样->标注，全局调度
    self.p.join()

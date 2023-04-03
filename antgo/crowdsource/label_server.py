# -*- coding: UTF-8 -*-
# @Time    : 2022/8/21 21:46
# @File    : label_server.py.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import sys
import copy

import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from tornado import web, gen
from tornado import httpclient
import tornado.web
import os
import shutil
import signal
from antgo.crowdsource.base_server import *
from antgo.utils import logger
import sys
import uuid
import json
import time
import math
import numpy as np
from io import BytesIO
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import threading
import base64
import imagesize
from antgo.utils import colormap


MB = 1024 * 1024
GB = 1024 * MB
MAX_STREAMED_SIZE = 1*GB

class LiveApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class LabelTaskHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    task_type = self.db['task_metas']['task_type']
    task_name = self.db['task_metas']['task_name']
    label_type = self.db['task_metas']['label_type']

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'task_type': task_type,
      'task_name': task_name,
      'label_type': label_type,
      'operator': {
        'full_name': user['full_name'],
        'short_name': user['short_name'],
      },
      'label_metas': self.db['label_metas']
    })


class LabelStateHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    if 'running' not in self.db:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    info = {
      'project_type': 'LABEL',
      'project_state': {
        'state': self.db['running']['state'],
        'stage': self.db['running']['stage'],
        'round': self.db['running']['round'],
        'waiting_time_to_next_round': self.db['running']['waiting_time_to_next_round']
      }
    }

    if self.db['running']['state'] == 'running' and self.db['running']['stage'] == 'labeling':
      sample_num_completed = 0
      for sample in self.db['samples']:
        if sample['completed_time'] != 0:
          sample_num_completed += 1
      info['project_state'].update({
        'sample_num_completed': sample_num_completed,
        'sample_num': len(self.db['samples'])
      })

    # state: 'running'/'abnormal'/'overtime'/'pending'
    # stage: 'waiting'/'training'/'labeling'/'finish'
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=info)

  @gen.coroutine
  def post(self):
    running_state = self.get_argument('running_state', None)
    running_stage = self.get_argument('running_stage', None)
    running_round = self.get_argument("running_round", '-1')
    if running_state is None or running_stage is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    assert(running_state in ['running', 'abnormal', 'overtime', 'pending'])
    assert(running_stage in ['waiting', 'labeling', 'finish'])
    # running_stage的finish状态需要由前端触发，标记本轮标注已经结束

    if running_stage == 'labeling':
      running_round = int(running_round)
      if running_round == -1:
        self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
        return 

      self.db['running']['state'] = running_state
      self.db['running']['stage'] = running_stage
      self.db['running']['round'] = (int)(running_round)
      self.response(RESPONSE_STATUS_CODE.SUCCESS)
      return

    # 强制完成标记 （如果是true，则即使是未完成所有样本标注，也强制设置完成标记）
    is_force = self.get_argument('force', False)
    if running_stage == 'finish':
      # 检查是否每个数据已经完成审核或标注
      sample_num = len(self.db['samples'])
      completed_num = 0
      for sample in self.db['samples']:
        if sample['state'] == 'completed':
          completed_num += 1
        
      if sample_num != completed_num and not is_force:
        self.response(
          RESPONSE_STATUS_CODE.SUCCESS, 
          content={
            'finish': False,
            'completed_num': completed_num,
            'sample_num': sample_num
          })
        return 
      else:
        self.response(
          RESPONSE_STATUS_CODE.SUCCESS, 
          content={
            'finish': True
          })
  
    if running_stage == 'waiting':
      # 切换运行状态为等待, 清空标注样本
      self.db['samples'] = []

    self.db['running']['state'] = running_state
    self.db['running']['stage'] = running_stage
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class LabelSamplesHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 获得所有等待标注数据
    page_index = self.get_argument('page_index', None)
    page_size = self.get_argument('page_size', None)
    index_offset = self.get_argument('index_offset', None)

    samples = []
    if page_size is None or page_index is None:
      samples = self.db['samples']
    elif index_offset is not None:
      index_offset = (int)(index_offset)
      page_size = (int)(page_size)
      start_page_index = (int)(page_index)
      end_page_index = index_offset // page_size + 1

      start_index = start_page_index * page_size
      end_index = end_page_index * page_size

      samples = self.db['samples'][start_index:end_index]
      page_index = end_page_index
    else:
      page_index = (int)(page_index)
      page_size = (int)(page_size)
      start_index = page_index * page_size
      end_index = (page_index + 1) * page_size

      samples = self.db['samples'][start_index:end_index]
      page_index = page_index + 1

    # copy
    samples = copy.deepcopy(samples)
    for sample in samples:
      # 转换时间到可显示
      for key in sample.keys():
        if key.endswith('time'):
          sample[key] = self.timestamp_2_str(sample[key])

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'page_samples': samples,
      'page_index': page_index,
      'columns': self.db['sample_metas']['filters'],
      'total_sample_num': len(self.db['samples'])
    })


class LabelFreshSampleHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 样本例子
    # {
    # 'image_file': '../assets/1.jpeg',
    # 'height': 800,
    # 'width': 1200,
    # 'sample_id': 2,
    # 'completed_time': 0,
    # 'created_time': 0,
    # 'update_time': 0,
    # 'state': 'waiting',
    # 'assigner': '',
    # 'operators': [],
    # 'label_info': [],
    # }
    #
    # 添加样本记录
    samples = self.get_argument('samples', None)
    if samples is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return
    samples = json.loads(samples)

    offset = len(self.db['samples'])
    for sample_i, sample in enumerate(samples):
      image_file = sample['image_file']
      # image_path = os.path.join(self.settings.get('static_path'), image_file)
      # width, height = imagesize.get(image_path)
      # sample['width'] = width
      # sample['height'] = height
      sample['sample_id'] = sample_i + offset
      sample['completed_time'] = 0
      sample['created_time'] = 0
      sample['update_time'] = 0
      sample['state'] = 'waiting'
      sample['assigner'] = ''
      sample['operators'] = []
      if 'label_info' not in sample:
        sample['label_info'] = []

    # 更新到数据库中
    self.db['samples'].extend(samples)
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class LabelUpdateSampleHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 样本标注结果
    update_sample = self.get_argument('sample', None)
    if update_sample is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # # 当前操作者
    # user_name = self.get_argument('user_name', 'default')
    # if user_name is None:
    #   self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
    #   return

    update_sample = json.loads(update_sample)
    sample_id = update_sample['sample_id']
    if sample_id < 0 or sample_id >= len(self.db['samples']):
      # 异常
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if self.db['samples'][sample_id]['image_file'] != update_sample['image_file']:
      # 异常
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if user['labeling_sample'] == -1 or user['start_time'] == -1:
      # 异常
      self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
      return

    if self.db['samples'][sample_id]['assigner'] != user['full_name']:
      # 说明，此样本已经重新分配给了其他人，当前操作者不可修改
      self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
      return

    now_time = time.time()
    if now_time - user['start_time'] > 10*60:
      # 当前用户标注超时，不能更新结果
      user['labeling_sample'] = -1
      user['start_time'] = -1
      self.response(RESPONSE_STATUS_CODE.EXECUTE_FORBIDDEN)
      return

    #
    label_info = update_sample['label_info']
    self.db['samples'][sample_id]['label_info'] = label_info  # 标注信息记录(仅保留最新一次的标注更新)
    self.db['samples'][sample_id]['state'] = 'completed'      # 设置当前样本处在完成状态（可能当前样本已经处在完成状态）

    # 如果当前样本首次提交，则更新完成时间
    # 其余情况仅更新更新时间
    self.db['samples'][sample_id]['update_time'] = now_time
    if self.db['samples'][sample_id]['completed_time'] == 0:
      self.db['samples'][sample_id]['completed_time'] = now_time

    # 当更新后，每个样本需要记录操作者信息（名字，更新时间）
    pi = -1
    for i, p in enumerate(self.db['samples'][sample_id]['operators']):
      if p['full_name'] == user['full_name']:
        pi = i
    if pi == -1:
      # 为此样本，创建新的操作者
      self.db['samples'][sample_id]['operators'].append({
        'full_name': user['full_name'],
        'short_name': user['short_name'],
        'update_time': now_time,
        'create_time': now_time,
        'label_info': label_info
      })
    else:
      self.db['samples'][sample_id]['operators'][pi]['update_time'] = now_time
      self.db['samples'][sample_id]['operators'][pi]['label_info'] = label_info

    # finish
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'sample_id': sample_id,
      'sample_state': self.db['samples'][sample_id]['state'],
      'completed_time': self.timestamp_2_str(self.db['samples'][sample_id]['completed_time']),
      'update_time': self.timestamp_2_str(self.db['samples'][sample_id]['update_time']),
    })


class LabelGetSampleHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 调用此接口时，返回的样本是有可能已经存在标注结果
    sample_i = self.get_argument('sample_i', None)
    if sample_i is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    sample_i = (int)(sample_i)
    if sample_i < 0 or sample_i >= len(self.db['samples']):
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    now_time = time.time()
    if user['labeling_sample'] != -1:
      # 当前用户存在正在标注的样本
      now_labeling_sample = user['labeling_sample']
      if self.db['samples'][now_labeling_sample]['assigner'] == user['full_name'] and \
          self.db['samples'][now_labeling_sample]['state'] != 'completed':
        # 如果之前的那个样本还没有处在完成状态(包括标注状态)，则重新切换成等待状态
        self.db['samples'][now_labeling_sample]['state'] = 'waiting'
        self.db['samples'][now_labeling_sample]['created_time'] = 0
        self.db['samples'][now_labeling_sample]['assigner'] = ''

      # 重置
      user['labeling_sample'] = -1
      user['start_time'] = -1

    # 更改当前操作者的状态
    user['labeling_sample'] = sample_i
    user['start_time'] = now_time

    # 更改当前样本的状态
    if self.db['samples'][sample_i]['state'] != 'waiting':
      # 进入此处，说明处在标注状态或完成状态
      if self.db['samples'][sample_i]['state'] != 'completed':
        if now_time - self.db['samples'][sample_i]['created_time'] > 10 * 60:
          # 当前样本处在超时状态 (十分钟)
          # 将当前样本状态切换为等待状态
          self.db['samples'][sample_i]['state'] = 'waiting'     # 重置状态
          self.db['samples'][sample_i]['assigner'] = ''         # 清空已经分配的操作者
          self.db['samples'][sample_i]['created_time'] = 0      # 清空创建时间

    # 仅有当前样本处在未锁定状态才允许进行重新分配标注人员
    if self.db['samples'][sample_i]['state'] != 'labeling':
      self.db['samples'][sample_i]['assigner'] = user['full_name']

    if self.db['samples'][sample_i]['state'] == 'waiting':
      # 仅当当前样本状态处在等待状态时，需要改变当前状态为标注状态，并设置创建时间
      # 否则保持状态不变
      self.db['samples'][sample_i]['state'] = 'labeling'
      self.db['samples'][sample_i]['created_time'] = now_time

    sample = copy.deepcopy(self.db['samples'][sample_i])
    # 转换时间到可显示
    for key in sample.keys():
      if key.endswith('time'):
        sample[key] = self.timestamp_2_str(sample[key])

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'sample': sample
    })


class LabelNextSampleHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # TODO, 需要根据当前标注样本索引，继续向下寻找
    # update_sample = self.get_argument('sample', None)
  
    # 发现下一个还没有标注的样本(没有被分配的样本)
    waiting_label_sample_i = -1
    # state: labeling/completed/waiting
    # state: 标注状态/完成状态/等待状态/超时状态
    for sample_i in range(len(self.db['samples'])):
      if self.db['samples'][sample_i]['state'] == 'waiting':
        waiting_label_sample_i = sample_i
        break

      if self.db['samples'][sample_i]['state'] != 'completed':
        if time.time() - self.db['samples'][sample_i]['created_time'] > 10 * 60:
          # 当前样本处在超时状态 (十分钟)
          # 将当前样本状态切换为等待状态
          self.db['samples'][sample_i]['state'] = 'waiting'     # 重置状态
          self.db['samples'][sample_i]['assigner'] = ''         # 清空已经分配的操作者
          self.db['samples'][sample_i]['created_time'] = 0      # 清空创建时间
          waiting_label_sample_i = sample_i
          break

    if waiting_label_sample_i == -1:
      # 没有发现需要标注的样本
      # 说明当前轮标注结束，修改系统状态

      # self.db['running']['state'] = 'running'
      # self.db['running']['stage'] = 'finish'
      # self.db['running']['round'] += self.db['running']['round']
      # self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      #   'state': self.db['running']['state'],
      #   'stage': self.db['running']['stage'],
      #   'round': self.db['running']['round']
      # })
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if waiting_label_sample_i < 0 or waiting_label_sample_i >= len(self.db['samples']):
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    # 绑定当前用户的正在进行的标注样本
    if user['labeling_sample'] != -1:
      # 当前用户存在正在标注的样本
      now_labeling_sample = user['labeling_sample']
      if self.db['samples'][now_labeling_sample]['assigner'] == user['full_name'] and \
          self.db['samples'][now_labeling_sample]['state'] != 'completed':
        # 如果之前的那个样本还没有处在完成状态(包括标注状态,或等待状态)，则重新切换成等待状态
        self.db['samples'][now_labeling_sample]['state'] = 'waiting'
        self.db['samples'][now_labeling_sample]['created_time'] = 0
        self.db['samples'][now_labeling_sample]['assigner'] = ''

      # 重置
      user['labeling_sample'] = -1
      user['start_time'] = -1

    # 挑选出来的样本必须处在等待状态
    assert(self.db['samples'][waiting_label_sample_i]['state'] == 'waiting')

    now_time = time.time()
    # 更改当前操作者的状态
    user['labeling_sample'] = waiting_label_sample_i
    user['start_time'] = now_time

    # 更改当前样本的状态
    self.db['samples'][waiting_label_sample_i]['state'] = 'labeling'
    self.db['samples'][waiting_label_sample_i]['created_time'] = now_time
    self.db['samples'][waiting_label_sample_i]['assigner'] = user['full_name']

    # copy
    sample = copy.deepcopy(self.db['samples'][waiting_label_sample_i])

    # 转换时间到可显示
    for key in sample.keys():
      if key.endswith('time'):
        sample[key] = self.timestamp_2_str(sample[key])

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'sample': sample,
      'state': self.db['running']['state'],
      'stage': self.db['running']['stage'],
      'round': self.db['running']['round']
    })


class LoginHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user_name = self.get_argument('user_name', None)
    user_password = self.get_argument('user_password', None)
    if user_name is None or user_password is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    white_users = self.db['white_users']
    if white_users is None:
      # 无需登录
      self.response(RESPONSE_STATUS_CODE.SUCCESS)
      return

    if user_name not in white_users:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    if user_password != self.db['white_users'][user_name]['password']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    cookie_id = str(uuid.uuid4())
    self.db['users'].update({
      cookie_id: {
        "full_name": user_name,
        'short_name': user_name[0].upper(),
        'labeling_sample': -1,
        'start_time': -1
      }
    })
    self.set_login_cookie({'cookie_id': cookie_id})
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class LogoutHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    self.clear_cookie('antgo')


class UserInfoHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 获取当前用户名
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 遍历所有样本获得本轮，当前用户的标注记录
    statistic_info = {
      'sample_num': 0,
      'average_time': 0
    }

    for sample in self.db['samples']:
      for operator in sample['operators']:
        if operator['full_name'] == user['full_name']:
          statistic_info['sample_num'] += 1
          statistic_info['average_time'] += (sample['completed_time'] - sample['completed_time'])

    if statistic_info['sample_num'] != 0:
      statistic_info['average_time'] /= statistic_info['sample_num']

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'user_name': user['full_name'],
      'short_name': user['short_name'],
      'task_name': self.db['task_metas']['task_name'],
      'task_type': self.db['task_metas']['task_type'],
      'project_type': 'LABEL',
      'statistic_info': statistic_info
    })


class LabelExportHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    export_samples = []
    for sample_id, sample in enumerate(self.db['samples']):
      export_sample = {
        'id': sample_id,
        'file_upload': sample['image_file'] if 'file_upload' not in sample else sample['file_upload'],
        'data': {'image': sample['image_file']},
        'created_at': self.timestamp_2_str(sample['created_time']),
        'updated_at': self.timestamp_2_str(sample['update_time']),
        'project': self.db['task_metas']['task_name'],
        'annotations': []
      }

      for sample_operator_i, sample_operator in enumerate(sample['operators']):
        temp = {
          'id': sample_operator_i,
          'completed_by': sample_operator['full_name'],
          'updated_at': self.timestamp_2_str(sample_operator['update_time']),
          'created_at': self.timestamp_2_str(sample_operator['create_time']),
          'result': []
        }

        for label_result in sample_operator['label_info']:
          if label_result['type'] in ['RECT', 'POINT', 'POLYGON']:
            v = copy.deepcopy(label_result['shape'])
            v['labels'] = label_result['label']
            temp['result'].append({
              'value': v,
              'width': sample['width'],
              'height': sample['height'],           
              'type': label_result['type']
            })
          elif label_result['type'] == 'CHOICES':
            v = {}
            v['labels'] = label_result['label']
            temp['result'].append({
              'value': v,
              'width': sample['width'],
              'height': sample['height'],
              'type': label_result['type']
            })

        export_sample['annotations'].append(temp)

      export_samples.append(export_sample)

    # 导出数据下载
    project_name = self.db['task_metas']['task_name']
    now_time = time.strftime("%Y-%m-%dx%H:%M:%S", time.localtime(time.time()))
    random_id = str(uuid.uuid4())
    export_file_name = f'{project_name}-{now_time}-{random_id}.json'
    with open(os.path.join(self.settings.get('static_path'), export_file_name), 'w') as fp:
      json.dump(export_samples, fp)

    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', f'attachment; filename={export_file_name}')
    with open(os.path.join(self.settings.get('static_path'), export_file_name), 'rb') as f:
      while True:
        data = f.read(MB)
        if not data:
            break
        self.write(data)

    self.finish()


class LabelImportHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    pass

class PingApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def label_server_start(
    dump_dir,
    server_port,
    task_metas,
    sample_metas,
    label_metas,
    sample_folder=None,
    sample_list=[],
    running_metas=None,
    white_users=None):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    static_dir = os.path.join(dump_dir, 'static')
    if not os.path.exists(static_dir):
      os.makedirs(static_dir)

    # 2.step launch show server
    db = {}
    # 添加默认样本列表
    db['samples'] = []

    if len(sample_list) > 0 and sample_folder is not None:
      sample_folder = os.path.abspath(sample_folder)  # 将传入的路径转换为绝对路径    
      # 为样本所在目录建立软连接到static下面
      os.system(f'cd {static_dir}; ln -s {sample_folder} dataset;')
                  
      # 将默认导入的样本直转换成新格式,并写入本地数据库
      for sample_i, sample_info in enumerate(sample_list):
        label_info = []
        # 添加目标框标注信息
        if len(sample_info['bboxes']) > 0:
          box_num = len(sample_info['bboxes'])
          for bi in range(box_num):
            box = sample_info['bboxes'][bi]
            label = sample_info['labels'][bi]
            label_info.append({
                  'type': 'RECT',
                  'shape': {
                    'x': box[0],
                    'y': box[1],
                    'width': box[2]-box[0],
                    'height': box[3]-box[1]
                  },
                  'label': label
            })
        elif 'bboxes' in sample_info['predictions']:
          box_num = len(sample_info['predictions']['bboxes'])
          for bi in range(box_num):
            box = sample_info['predictions']['bboxes'][bi]
            label = sample_info['predictions']['labels'][bi]
            label_info.append({
                  'type': 'RECT',
                  'shape': {
                    'x': box[0],
                    'y': box[1],
                    'width': box[2]-box[0],
                    'height': box[3]-box[1]
                  },
                  'label': label
            })
        
        # 添加多边形标注信息
        if len(sample_info['segments']) > 0:
          segment_num = len(sample_info['segments'])
          for bi in range(segment_num):
            points = sample_info['segments'][bi]
            label = sample_info['labels'][bi]
            label_info.append({
                  'type': 'POLYGON',
                  'shape': {
                    'points': points
                  },
                  'label': label
            })
        elif 'segments' in sample_info['predictions']:
          box_num = len(sample_info['predictions']['segments'])
          for bi in range(box_num):
            points = sample_info['predictions']['segments'][bi]
            label = sample_info['predictions']['segments'][bi]
            label_info.append({
                  'type': 'POLYGON',
                  'shape': {
                    'points': points
                  },
                  'label': label
            })
        
        # 添加关键点标注信息
        # pass
        
        # 添加图片类别
        if sample_info['image_label_name'] != '':
          label_info.append({
              'type': 'CHOICES',
              'choices': [sample_info['image_label_name']]
          })     
        elif 'image_label_name' in sample_info['predictions']:
          label_info.append({
              'type': 'CHOICES',
              'choices': [sample_info['predictions']['image_label_name']]
          })          
        
        # 加入数据库中
        db['samples'].append({
              'image_file': f'/static/dataset/{sample_info["image_file"]}' if sample_info['image_file'] != '' else sample_info['image_url'],
              # 'image_file': 'https://www.ssfiction.com/wp-content/uploads/2020/08/20200805_5f2b1669e9a24.jpg',
              'height': sample_info['height'],
              'width': sample_info['width'],
              'sample_id': sample_i,
              'completed_time': 0,
              'created_time': 0,
              'update_time': 0,
              'operators': [],
              'assigner': '',
              'state': 'waiting',
              'label_info': label_info,
        })
    
    # 设置样本基本信息
    db['sample_metas'] = sample_metas

    # 设置任务基本信息
    assert('task_type' in task_metas and 'task_name' in task_metas and 'label_type' in task_metas)
    db['task_metas'] = task_metas

    # 设置标注基本信息
    assert('category' in label_metas)
    label_category = label_metas['category']
    label_metas['label_category'] = label_category
    for i, label_config in enumerate(label_metas['label_category']):      
      if 'color' not in label_config or 'background_color' not in label_config:
        # 自动选择配色
        color = colormap.highlight[i%len(colormap.highlight)]['color']
        background_color = colormap.dark[i%len(colormap.dark)]['color']
        
        label_config['color'] = color
        label_config['background_color'] = background_color

    db['label_metas'] = label_metas

    # 设置白盒用户
    db['white_users'] = white_users

    # 其他初始化
    db['users'] = {}

    # 设置运行初始状态
    # state: 'running'/'abnormal'/'overtime'/'pending'
    # stage: 'waiting'/'training'/'labeling'/'finish'
    db['running'] = {}
    db['running']['state'] = 'running' if running_metas is None else running_metas['state']
    db['running']['stage'] = 'waiting' if running_metas is None else running_metas['stage']
    db['running']['round'] = 0
    db['running']['waiting_time_to_next_round'] = 0

    # cookie
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)

    settings = {
      'static_path': static_dir,
      'db': db,
      'cookie_secret': cookie_secret,
      'cookie_max_age_days': 30,
      'Content-Security-Policy': "frame-ancestors 'self' {}".format('http://localhost:8080/')
    }

    app = tornado.web.Application(
      handlers=[
        (r"/antgo/api/label/task/", LabelTaskHandler),                    # 获得系统任务
        (r"/antgo/api/label/sample/", LabelSamplesHandler),               # 获得所有样本（TODO, 支持lazyload）
        (r"/antgo/api/label/sample/fresh/", LabelFreshSampleHandler),     # 更新所有样本
        (r"/antgo/api/label/sample/update/", LabelUpdateSampleHandler),   # 更新样本标注结果
        (r"/antgo/api/label/sample/get/", LabelGetSampleHandler),         # 得到指定样本信息
        (r"/antgo/api/label/sample/next/", LabelNextSampleHandler),       # 得到下一个需要进行标注的样本信息
        (r"/antgo/api/label/export/", LabelExportHandler),                # 标注导出
        (r"/antgo/api/label/export/download/", LabelExportHandler),       # 标注导出
        (r"/antgo/api/label/import/", LabelImportHandler),                # 标注导入
        (r"/antgo/api/user/info/", UserInfoHandler),                  # 获得用户信息
        (r"/antgo/api/user/login/", LoginHandler),  # 登录，仅支持预先指定用户
        (r"/antgo/api/user/logout/", LogoutHandler),  # 退出
        (r"/antgo/api/ping/", PingApiHandler),  # ping 服务
        (r"/antgo/api/info/", LabelStateHandler),  # 获得当前系统状态
        (r'/(.*)', tornado.web.StaticFileHandler, {"path": static_dir, "default_filename": "index.html"})
      ],
      **settings
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('ANTGO-LABEL is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('ANTGO-LABEL stop server')
  except GracefulExitException:
    logger.info('ANTGO-LABEL server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  task_metas = {
      'task_type': 'OBJECT-DET',      #
      'task_name': 'HELLO',
      # 'label_type': 'RECT',
      # 'label_type': 'POINT',
      'label_type': 'POLYGON'
    }
  sample_metas = {
    'filters': ['id', 'time']
  }

  running_metas = {
    'state': 'running',
    'stage': 'labeling'
  }
  label_metas = {
        'category': [
          {
            'class_name': 'A',
            'class_index': 0,
            'color': 'green',
            'background_color': '#00800026'
          },
          {
            'class_name': 'B',
            'class_index': 1,
            'color': 'blue',
            'background_color': '#0000ff26'
          },
        ],
      }
  white_users = {
    'jian@baidu.com': {
      'password': '112233'
    }
  }
  samples = [
            {
              'image_file': '/static/data/1.png',
              'height': 800,
              'width': 1200,
              'sample_id': 0,
              'completed_time': 0,
              'created_time': 0,
              'update_time': 0,
              'operators': [],
              'assigner': '',
              'state': 'waiting',
              'label_info': [
                {
                  'type': 'RECT',
                  'shape': {
                    'x': 0,
                    'y': 0,
                    'width': 100,
                    'height': 100
                  },
                  'label': 0
                },
                {
                  'type': 'RECT',
                  'shape': {
                    'x': 100,
                    'y': 100,
                    'width': 100,
                    'height': 100
                  },
                  'label': 1
                }
              ],
            },
            {
              'image_file': '/static/data/1.png',
              'height': 800,
              'width': 1200,
              'sample_id': 1,
              'completed_time': 0,
              'created_time': 0,
              'update_time': 0,
              'operators': [],
              'state': 'waiting',
              'assigner': '',
              'label_info': [],
            },
            {
              'image_file': '/static/data/1.png',
              'height': 800,
              'width': 1200,
              'sample_id': 2,
              'completed_time': 0,
              'created_time': 0,
              'update_time': 0,
              'state': 'waiting',
              'assigner': '',
              'operators': [],
              'label_info': [],
            }
          ] * 100
  for i in range(len(samples)):
    samples[i] = copy.deepcopy(samples[i])
    samples[i]['sample_id'] = i

  label_server_start('/Users/bytedance/Downloads/workspace/my/A', 9000, task_metas, sample_metas, label_metas, samples, running_metas, white_users)

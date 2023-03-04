# encoding=utf-8
# @Time    : 17-3-3
# @File    : common.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from ..utils.serialize import loads,dumps
from ..utils import logger
import uuid
import json
import sys
import tarfile
import tempfile
import re
import requests
from antgo.ant import flags
from antgo.utils.fs import *
from antgo import config
from antgo.ant.utils import *
import yaml
from antgo.utils.utils import *
from datetime import datetime
from antgo.context import *
import socket


if sys.version > '3':
  PY3 = True
else:
  PY3 = False

FLAGS = flags.AntFLAGS
Config = config.AntConfig


def _is_open(check_ip, port):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  try:
    s.connect((check_ip, int(port)))
    s.shutdown(2)
    return True
  except:
    return False


def _pick_idle_port(from_port=40000, check_count=100):
  check_port = from_port
  while check_count:
    if not _is_open('127.0.0.1', check_port):
      break

    logger.warn('port %d is occupied, try to use %d port'%(int(check_port), int(check_port + 1)))

    check_port += 1
    check_count -= 1

    if check_count == 0:
      check_port = None

  if check_port is None:
    logger.warn('couldnt find valid free port')
    exit(-1)

  return check_port


class AntBase(object):
  def __init__(self, ant_name, ant_context=None, ant_token=None, **kwargs):
    self.user_token = getattr(Config, 'server_user_token', '')
    self.http_prefix = 'http'
    self.ant_name = ant_name
    self.app_token = ant_token

    # three key info
    if 'main_file' in kwargs:
      self.main_file = kwargs['main_file']
    if 'main_folder' in kwargs:
      self.main_folder = kwargs['main_folder']
    if 'main_param' in kwargs:
      self.main_param = kwargs['main_param']
    if 'time_stamp' in kwargs:
      self._time_stamp = kwargs['time_stamp']
    else:
      self._time_stamp = timestamp()

    self._proxy = None
    if 'proxy' in kwargs:
      self._proxy = kwargs['proxy']

    self._signature = None
    if 'signature' in kwargs:
      self._signature = kwargs['signature']

    # current pid
    self._pid = str(os.getpid())

    # server flag
    self.app_server = self.__class__.__name__
    if not PY3:
      self.app_server = unicode(self.app_server)

    # parse hardware resource config
    self._running_config = {'GPU_MODEL': '',
                            'GPU_NUM': 0,
                            'GPU_MEM': 0,
                            'CPU_MODEL': '',
                            'CPU_NUM': 0,
                            'CPU_MEM': 0,
                            'OS_PLATFORM': '',
                            'OS_VERSION': '',
                            'SOFTWARE_FRAMEWORK': '',
                            'DATASET': ''}

    self._description_config = {'SHORT_DESCRIPTION': '',
                                'LONG_DESCRIPTION': '',
                                'VERSION': '',
                                'INPUT_NUM': 1,
                                'INPUT_TYPE':[]}

    if ant_context is not None and \
            ant_context.params is not None and \
            ant_context.params._params is not None:
      config_params = ant_context.params._params
      if 'RUNNING_CONFIG' in config_params:
        if 'GPU_MODEL' in config_params['RUNNING_CONFIG']:
          self._running_config['GPU_MODEL'] = config_params['RUNNING_CONFIG']['GPU_MODEL']

        if 'GPU_NUM' in config_params['RUNNING_CONFIG']:
          self._running_config['GPU_NUM'] = config_params['RUNNING_CONFIG']['GPU_NUM']

        if 'GPU_MEM' in config_params['RUNNING_CONFIG']:
          self._running_config['GPU_MEM'] = config_params['RUNNING_CONFIG']['GPU_MEM']

        if 'CPU_MODEL' in config_params['RUNNING_CONFIG']:
          self._running_config['CPU_MODEL'] = config_params['RUNNING_CONFIG']['CPU_MODEL']

        if 'CPU_NUM' in config_params['RUNNING_CONFIG']:
          self._running_config['CPU_NUM'] = config_params['RUNNING_CONFIG']['CPU_NUM']

        if 'CPU_MEM' in config_params['RUNNING_CONFIG']:
          self._running_config['CPU_MEM'] = config_params['RUNNING_CONFIG']['CPU_MEM']

        if 'OS_PLATFORM' in config_params['RUNNING_CONFIG']:
          self._running_config['OS_PLATFORM'] = config_params['RUNNING_CONFIG']['OS_PLATFORM']

        if 'OS_VERSION' in config_params['RUNNING_CONFIG']:
          self._running_config['OS_VERSION'] = config_params['RUNNING_CONFIG']['OS_VERSION']

        if 'SOFTWARE_FRAMEWORK' in config_params['RUNNING_CONFIG']:
          self._running_config['SOFTWARE_FRAMEWORK'] = config_params['RUNNING_CONFIG']['SOFTWARE_FRAMEWORK']

      if 'DESCRIPTION_CONFIG' in config_params:
        if 'SHORT_DESCRIPTION' in config_params['DESCRIPTION_CONFIG']:
          self._description_config['SHORT_DESCRIPTION'] = config_params['DESCRIPTION_CONFIG']['SHORT_DESCRIPTION']

        if 'LONG_DESCRIPTION' in config_params['DESCRIPTION_CONFIG']:
          self._description_config['LONG_DESCRIPTION'] = config_params['DESCRIPTION_CONFIG']['LONG_DESCRIPTION']

        if 'VERSION' in config_params['DESCRIPTION_CONFIG']:
          self._description_config['VERSION'] = config_params['DESCRIPTION_CONFIG']['VERSION']

        if 'INPUT_NUM' in config_params['DESCRIPTION_CONFIG']:
          self._description_config['INPUT_NUM'] = config_params['DESCRIPTION_CONFIG']['INPUT_NUM']

        if 'INPUT_TYPE' in config_params['DESCRIPTION_CONFIG']:
          self._description_config['INPUT_TYPE'] = config_params['DESCRIPTION_CONFIG']['INPUT_TYPE']

    # global context
    self.ant_context = None
    if ant_context is not None:
      self.ant_context = ant_context
      self.ant_context.ant = self

    self.experiment_uuid = \
      'antgo-%s-%s' % (str(uuid.uuid4()),
                    datetime.fromtimestamp(timestamp()).strftime('%Y%m%d-%H%M%S-%f'))

    if self.app_server in ["AntChallenge", "AntTrain", "AntBatch", "AntBrowser", "AntDemo", "AntWatch", "AntEnsemble"]:
      if self.app_token is not None:
        # 任务模式，在dashboard上创建实验记录
        mlogger.config(project=self.ant_name,
                       experiment=self.ant_name,
                       token=self.app_token,
                       server=self.app_server)

        if mlogger.info.experiment_uuid is not None:
          self.experiment_uuid = mlogger.info.experiment_uuid
      elif self.user_token is not None and self.user_token != '':
        # 非任务模式，基于user token与dashboard进行通信
        mlogger.config(project=self.ant_name,
                       experiment=self.ant_name,
                       token=self.user_token,
                       server=self.app_server)
        if mlogger.info.experiment_uuid is not None:
          self.experiment_uuid = mlogger.info.experiment_uuid
      else:
        # 本地模式（用户没有配置token）
        logger.warn("Now is in local mode. Please set user token in config file, enjoy experiment manage!")

  @property
  def running_dataset(self):
    raise NotImplementedError

  @property
  def running_task(self):
    raise NotImplementedError

  @property
  def pid(self):
    return self._pid

  @pid.setter
  def pid(self, val):
    self._pid = val

  @property
  def running_config(self):
    return self._running_config

  @property
  def description_config(self):
    return self._description_config

  @property
  def running_platform(self):
    return self.context.params.system['running_platform']

  def download(self, source_path, target_path=None, target_name=None, archive=None):
    if target_path is None:
      target_path = os.curdir

    is_that = re.match('^((https|http|ftp|rtsp|mms)?://)', source_path)
    if is_that is not None:
      download(source_path, target_path, fname=target_name)

      is_gz = re.match('.*\.gz', target_name)
      if is_gz is not None:
        if archive is not None:
          extracted_path = os.path.join(target_path, archive)
        else:
          extracted_path = target_path

        if not os.path.exists(extracted_path):
          os.makedirs(extracted_path)

        tar = tarfile.open(os.path.join(target_path, target_name))
        tar.extractall(extracted_path)
        tar.close()
        target_path = extracted_path

    return target_path

  @property
  def stage(self):
    return self.context.stage

  @stage.setter
  def stage(self, val):
    mlogger.info.experiment_stage = val
    self.context.stage = val

  @property
  def token(self):
    return self.app_token
  @token.setter
  def token(self, val):
    self.app_token = val

  @property
  def experiment_uuid(self):
    return self.context.experiment_uuid
  @experiment_uuid.setter
  def experiment_uuid(self, val):
    self.context.experiment_uuid = val

  @property
  def name(self):
    return self.ant_name

  @property
  def context(self):
    return self.ant_context

  @context.setter
  def context(self, val):
    self.ant_context = val
    self.ant_context.ant = self

  @property
  def proxy(self):
    return self._proxy

  @property
  def signature(self):
    return self._signature

  @property
  def time_stamp(self):
    return self._time_stamp
  
  def reset(self):
    if self.pid != str(os.getpid()):
      # 1.step reset process pid
      self.pid = str(os.getpid())

      # 2.step update context
      ctx = main_context(self.main_file, self.main_folder)
      if self.main_param is not None:
        main_config_path = os.path.join(self.main_folder, self.main_param)
        params = yaml.load(open(main_config_path, 'r'))
        ctx.params = params
      
      if self.context.from_experiment is not None:
        ctx.from_experiment = self.context.from_experiment
      
      self.context = ctx

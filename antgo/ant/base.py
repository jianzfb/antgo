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
import zmq
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
if sys.version > '3':
  PY3 = True
else:
  PY3 = False

FLAGS = flags.AntFLAGS
Config = config.AntConfig

class AntBase(object):
  def __init__(self, ant_name, ant_context=None, ant_token=None, **kwargs):
    self.server_ip = getattr(Config, 'server_ip', 'www.mltalker.com')
    self.http_port = getattr(Config, 'server_port', '8999')
    self.http_prefix = 'crowdsource'
    self.ant_name = ant_name
    self.app_token = os.environ.get('APP_TOKEN', ant_token)
    self.app_connect = os.environ.get('APP_CONNECT', 'tcp://%s:%s' % (self.server_ip, '2345'))
    self.app_file_connect = os.environ.get('APP_FILE_CONNECT', 'tcp://%s:%s' % (self.server_ip, '2346'))
    
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
    
    # current pid
    self._pid = str(os.getpid())
    
    # config zmq connect
    self._zmq_socket = zmq.Context().socket(zmq.REQ)
    self._zmq_socket.connect(self.app_connect)
    
    # config zmq file connect
    self._zmq_file_socket = zmq.Context().socket(zmq.DEALER)
    self._zmq_file_socket.connect(self.app_file_connect)
    
    # server flag
    self.app_server = self.__class__.__name__
    if not PY3:
      self.app_server = unicode(self.app_server)

    # core
    self.ant_context = None
    if ant_context is not None:
      self.ant_context = ant_context
      self.ant_context.ant = self
    
    # non mltalker task
    self._is_non_mltalker_task = False
    
  @property
  def zmq_socket(self):
    return self._zmq_socket
  @zmq_socket.setter
  def zmq_socket(self, val):
    self._zmq_socket = val
    self._zmq_socket.connect(self.app_connect)
  
  @property
  def is_non_mltalker_task(self):
    return self._is_non_mltalker_task
  @is_non_mltalker_task.setter
  def is_non_mltalker_task(self, val):
    self._is_non_mltalker_task = val
  
  @property
  def zmq_file_socket(self):
    return self._zmq_file_socket
  @zmq_file_socket.setter
  def zmq_file_socket(self,val):
    self._zmq_file_socket = val
    self._zmq_file_socket.connect(self.app_file_connect)
  
  @property
  def pid(self):
    return self._pid
  @pid.setter
  def pid(self, val):
    self._pid = val
  
  def send(self, data, stage):
    if self.is_non_mltalker_task:
      return
    
    if self.app_token is not None:
      # now_time = datetime.now().timestamp()
      now_time = timestamp()
      # 0.step add extra data
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.time_stamp
      if self.context is not None:
        if self.context.params is not None:
          data['APP_HYPER_PARAMETER'] = json.dumps(self.context.params.content)
      data['APP_RPC'] = "INFO"
      data['APP_STAGE'] = stage
      data['APP_NOW_TIME'] = now_time
      data["APP_NAME"] = self.ant_name
      data["APP_SERVER"] = self.app_server

      # exclude 'RECORD'
      record_data = None
      if 'RECORD' in data:
        record_data = data['RECORD']
        data.pop('RECORD')

      # 1.step send info
      self.zmq_socket.send(dumps(data))

      # 2.step ignore any receive info
      response = self.zmq_socket.recv(copy=False)
      response = loads(response)
      if 'status' in response:
        if response['status'] != 'OK':
          logger.error('error in uploading, maybe token isnot valid..')
          if self.app_server not in ['AntTrain','AntChallenge']:
            logger.error('perhaps you are using task token')
          return

      # 3.step upload record files
      if record_data is not None and os.path.exists(record_data):
        self.send_record(record_data, stage)
  
  def send_record(self, data, stage):
    if self.is_non_mltalker_task:
      return
    
    if self.app_token is not None:
      # format: token, stage, time_stamp, now_time_stamp, block_id, block_size, max_block_size, block
      # 1.step uuid
      record_id = str(uuid.uuid1()) if PY3 else unicode(uuid.uuid1())
      
      # 2.step tar record
      temp_tar_file_path = os.path.join(tempfile.gettempdir(), '%s.tar.gz'%record_id)
      if os.path.exists(temp_tar_file_path):
        os.remove(temp_tar_file_path)
      tar = tarfile.open(temp_tar_file_path, 'w:gz')
      if os.path.isdir(data):
        # folder
        for f in os.listdir(data):
          if os.path.isfile(os.path.join(data, f)):
            tar.add(os.path.join(data, f), arcname=f)
      else:
        # single file
        tar.add(data)
      tar.close()
      
      # 3.step split data pieces
      with open(temp_tar_file_path, 'rb') as fp:
        BLOCK_SIZE = 8 * 1024
        block_data = fp.read(BLOCK_SIZE)
        
        # send data blocks
        while block_data != b"":
          self.zmq_file_socket.send(dumps((self.app_token,
                                           self.ant_name,
                                           stage,
                                           self.time_stamp,
                                           'EXPERIMENT-RECORD',
                                           record_id,
                                           BLOCK_SIZE,
                                           len(block_data),
                                           block_data)))
          block_data = fp.read(BLOCK_SIZE)
        
        # send data EOF
        self.zmq_file_socket.send(dumps((self.app_token,
                                         self.ant_name,
                                         stage,
                                         self.time_stamp,
                                         'EXPERIMENT-RECORD',
                                         record_id,
                                         BLOCK_SIZE,
                                         0,
                                         b'')))
        # waiting until server tells us it's done
        flag = self.zmq_file_socket.recv()

      # 4.step clear
      if os.path.exists(temp_tar_file_path):
        os.remove(temp_tar_file_path)

  def send_file(self, file_path, name, stage, mode, target_name):
    if self.is_non_mltalker_task:
      return
    
    # 1.step whether file_path exist
    if not os.path.isfile(file_path):
      return False

    # 2.step split data pieces
    with open(file_path, 'rb') as fp:
      BLOCK_SIZE = 8 * 1024
      block_data = fp.read(BLOCK_SIZE)

      # send data blocks
      while block_data != b"":
        self.zmq_file_socket.send(dumps((self.app_token,
                                         name,
                                         stage,
                                         self.time_stamp,
                                         mode,
                                         target_name,
                                         BLOCK_SIZE,
                                         len(block_data),
                                         block_data)))
        block_data = fp.read(BLOCK_SIZE)

      # send data EOF
      self.zmq_file_socket.send(dumps((self.app_token,
                                       name,
                                       stage,
                                       self.time_stamp,
                                       mode,
                                       target_name,
                                       BLOCK_SIZE,
                                       0,
                                       b'')))
      # waiting until server tells us it's done
      flag = self.zmq_file_socket.recv()
      return True

  def rpc(self, cmd=""):
    if self.app_token is not None:
      # 0.step config data
      data = {}
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.time_stamp
      data['APP_RPC'] = cmd
      data['APP_STAGE'] = 'RPC'
      data['APP_NOW_TIME'] = timestamp()
      data["APP_NAME"] = self.ant_name
      data['APP_SERVER'] = self.app_server

      # 1.step send rpc
      self.zmq_socket.send(dumps(data))

      # 2.step receive info
      try:
        response = loads(self.zmq_socket.recv(copy=False))
        if len(response) == 0:
          return None
        return response
      except:
        return None

    return None

  def download(self, source_path, target_path=None, target_name=None, archive=None):
    if target_path is None:
      target_path = os.curdir

    is_that = re.match('^((https|crowdsource|ftp|rtsp|mms)?://)', source_path)
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

  def remote_api_request(self, cmd, data=None, action='get'):
    url = '%s://%s:%s/%s'%(self.http_prefix, self.server_ip, self.http_port, cmd)
    user_authorization = {'Authorization': "token " + self.app_token}
    try:
        response = None
        if action == 'get':
          # get a resource at server
          response = requests.get(url, data=data, headers=user_authorization)
        elif action == 'post':
          # build a resource at server
          response = requests.post(url, data=data, headers=user_authorization)
        elif action == 'patch':
          # update part resource at server
          response = requests.patch(url, data=data, headers=user_authorization)
        elif action == 'delete':
          # delete resource at server
          response = requests.delete(url, data=data, headers=user_authorization)

        if response is None:
          return None

        response_js = json.loads(response.content.decode())
        if 'status' in response_js and response_js['status'] in [404, 500]:
          return None

        return response_js
    except:
        return None

  @property
  def stage(self):
    return self.context.stage
  @stage.setter
  def stage(self, val):
    self.context.stage = val

  @property
  def token(self):
    return self.app_token
  @token.setter
  def token(self, val):
    self.app_token = val

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
  def time_stamp(self):
    return self._time_stamp
  
  def flash(self):
    if self.pid != str(os.getpid()):
      # reset process pid
      self.pid = str(os.getpid())
      
      # update zmq sockets
      # (couldnt share socket in differenet process)
      self.zmq_socket = zmq.Context().socket(zmq.REQ)
      self.zmq_file_socket = zmq.Context().socket(zmq.DEALER)
      
      # update context
      ctx = main_context(self.main_file, self.main_folder)
      if self.main_param is not None:
        main_config_path = os.path.join(self.main_folder, self.main_param)
        params = yaml.load(open(main_config_path, 'r'))
        ctx.params = params
      
      self.context = ctx

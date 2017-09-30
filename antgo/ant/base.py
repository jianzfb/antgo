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
import time
import json
import sys
import tarfile
import tempfile
from antgo.ant import flags
if sys.version > '3':
  PY3 = True
else:
  PY3 = False

FLAGS = flags.AntFLAGS


class AntBase(object):
  def __init__(self, ant_name, ant_context=None, ant_token=None):
    self.ant_name = ant_name
    self.app_token = os.environ.get('APP_TOKEN', ant_token)
    self.app_connect = os.environ.get('APP_CONNECT', 'tcp://10.231.96.130:2345')
    self.app_file_connect = os.environ.get('APP_FILE_CONNECT', 'tcp://10.231.96.130:2346')

    # config zmq connect
    self.zmq_context = zmq.Context()
    self.zmq_socket = self.zmq_context.socket(zmq.REQ)
    self.zmq_socket.connect(self.app_connect)
    
    # config zmq file connect
    self.zmq_file_context = zmq.Context()
    self.zmq_file_socket = self.zmq_file_context.socket(zmq.DEALER)
    self.zmq_file_socket.connect(self.app_file_connect)
    
    # server flag
    self.app_server = self.__class__.__name__
    if not PY3:
      self.app_server = unicode(self.app_server)

    # core
    self.ant_context = None
    if ant_context is not None:
      self.ant_context = ant_context
      self.ant_context.ant = self

    # time
    self.ant_time_stamp = time.time()

  def send(self, data, stage):
    if self.app_token is not None:
      now_time = time.time()
      # 0.step add extra data
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.ant_time_stamp
      # if self.context is not None:
      #   if self.context.params is not None:
      #     data['APP_HYPER_PARAMETER'] = json.dumps(self.context.params)
      data['APP_RPC'] = "INFO"
      data['APP_STAGE'] = stage
      data['APP_NOW_TIME'] = now_time
      data["APP_NAME"] = self.ant_name
      data["APP_SERVER"] = self.app_server

      # 1.step send info
      self.zmq_socket.send(dumps(data))

      # 2.step ignore any receive info
      self.zmq_socket.recv(copy=False)
      
      # 3.step upload record files
      if 'RECORD' in data and os.path.exists(data['RECORD']):
        self.send_record(data['RECORD'], stage)
  
  def send_record(self, data, stage):
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
                                           self.ant_time_stamp,
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
                                         self.ant_time_stamp,
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
  
  def rpc(self, cmd="", dump_dir=None):
    if self.app_token is not None:
      # 0.step config data
      data = {}
      data['APP_TOKEN'] = self.app_token
      data['APP_TIME'] = self.ant_time_stamp
      data['APP_RPC'] = cmd
      data['APP_STAGE'] = 'RPC'
      data['APP_NOW_TIME'] = time.time()
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
    return self.ant_time_stamp
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
from antgo.ant.subgradientrpc import *
from antgo.ant.mltalkerrpc import *
from antgo.ant.warehouse import *
from qiniu import Auth, put_file, etag, urlsafe_base64_encode
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
    self.http_prefix = 'http'
    self.ant_name = ant_name
    self.app_token = os.environ.get('APP_TOKEN', ant_token)
    self.app_connect = os.environ.get('APP_CONNECT', 'tcp://%s:%s' % (self.server_ip, '2345'))
    self.app_file_connect = os.environ.get('APP_FILE_CONNECT', 'tcp://%s:%s' % (self.server_ip, '2346'))

    self.subgradientserver = getattr(Config, 'subgradientserver', {})

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

    # subgradient rpc
    self.subgradient_rpc = SubgradientRPC(self.subgradientserver['subgradientserver_ip'], self.subgradientserver['subgradientserver_port'])
    self.mltalker_rpc = MLTalkerRPC(self.server_ip, self.http_port, self.app_token)

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

    if ant_context is not None and ant_context.params is not None:
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

    self._running_platform = kwargs.get('running_platform', 'local')    # local, cloud

    # core
    self.ant_context = None
    if ant_context is not None:
      self.ant_context = ant_context
      self.ant_context.ant = self

  @property
  def zmq_socket(self):
    return self._zmq_socket
  @zmq_socket.setter
  def zmq_socket(self, val):
    self._zmq_socket = val
    self._zmq_socket.connect(self.app_connect)

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

  @property
  def running_config(self):
    return self._running_config

  @property
  def description_config(self):
    return self._description_config

  @property
  def running_platform(self):
    return self._running_platform

  def _recursive_tar(self, root_path, path, tar, ignore=None):
    if path.split('/')[-1][0] == '.':
      return

    if os.path.isdir(path):
      for sub_path in os.listdir(path):
        self._recursive_tar(root_path, os.path.join(path, sub_path), tar)
    else:
      if ignore is not None:
        if path.split('/')[-1] == ignore:
          return
      arcname = os.path.relpath(path, root_path)
      tar.add(path, arcname=arcname)

  def package_codebase(self, prefix='qiniu'):
    logger.info('package antgo codebase')

    if self.app_token is None:
      if not os.path.exists(os.path.join(self.main_folder, FLAGS.task())):
        shutil.copy(os.path.join(Config.task_factory, FLAGS.task()), os.path.join(self.main_folder))

    random_code_package_name = str(uuid.uuid4())
    code_tar_path = os.path.join(self.main_folder, '%s_code.tar.gz'%random_code_package_name)
    tar = tarfile.open(code_tar_path, 'w:gz')
    for sub_path in os.listdir(self.main_folder):
      self._recursive_tar(self.main_folder,
                          os.path.join(self.main_folder, sub_path),
                          tar,
                          ignore='%s_code.tar.gz'%random_code_package_name)
    tar.close()

    #
    crypto_code = str(uuid.uuid4())
    crypto_shell = 'openssl enc -e -aes256 -in %s -out %s -k %s'%('%s_code.tar.gz'%random_code_package_name,
                                                                  '%s_code_ssl.tar.gz'%random_code_package_name,
                                                                  crypto_code)
    subprocess.call(crypto_shell, shell=True, cwd=self.main_folder)

    logger.info('finish package')
    if prefix == 'qiniu':
      logger.info('upload codebase package')
      qiniu_address = qiniu_upload(os.path.join(self.main_folder, '%s_code_ssl.tar.gz'%random_code_package_name),
                                   bucket='mltalker',
                                   max_size=100)
      return qiniu_address, crypto_code

      # access_key = 'ZSC-X2p4HG5uvEtfmn5fsTZ5nqB3h54oKjHt0tU6'
      # secret_key = 'Ya8qYwIDXZn6jSJDMz_ottWWOZqlbV8bDTNfCGO0'
      # q = Auth(access_key, secret_key)
      # key = 'code.tar.gz'
      # token = q.upload_token('image', key, 3600)
      # ret, info = put_file(token, key, code_tar_path)
      # if ret['key'] == key and ret['hash'] == etag(code_tar_path):
      #   logger.info('finish upload')
      #   return 'qiniu:http://otcf1mj36.bkt.clouddn.com/%s'%key
      # else:
      #   logger.info('fail upload')
      #   return None
    elif prefix == 'ipfs':
      pass
    elif prefix == 'baidu':
      pass

    return None

  def register_ant(self, codebase_address, running_config, server_config={}):
    request_url = '%s://%s:%d/api/aifactory/register'%(self.http_prefix, self.server_ip, self.http_port)

    data_str = json.dumps({'CODE_BASE': codebase_address,
                           'RUNNING_CONFIG': running_config,
                           'SERVER_CONFIG': server_config})
    response = requests.post(request_url, {'DATA': data_str})

    if response is None:
      return None

    if response.status_code in [200, 201]:
      result = json.loads(response.content)
      return result
    else:
      return None

  def submit_ant(self, codebase_address, running_config, server_config={}):
    pass

  def send(self, data, stage):
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

        if response.status_code != 200 and response.status_code != 201:
          return None

        response_js = json.loads(response.content.decode())
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
  
  def clone(self):
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
      
      if self.context.from_experiment is not None:
        ctx.from_experiment = self.context.from_experiment
      
      self.context = ctx

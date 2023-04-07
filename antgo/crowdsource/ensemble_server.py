# -*- coding: UTF-8 -*-
# @Time    : 2021/11/13 6:13 下午
# @File    : ensemble_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from copyreg import pickle
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
import pickle
import copy


def softmax(x, axis=None):
  x = x - x.max(axis=axis, keepdims=True)
  y = np.exp(x)
  return y / y.sum(axis=axis, keepdims=True)


MB = 1024 * 1024
GB = 1024 * MB
MAX_STREAMED_SIZE = 1*GB
@tornado.web.stream_request_body
class AverageApiHandler(BaseHandler):
  executor = ThreadPoolExecutor(16)

  def __init__(self, *args, **kwargs):
    super(AverageApiHandler, self).__init__(*args, **kwargs)
    self.save_seek = 0
    self.worker_prefix = None
    self.file_folder = None
    self.file_name = None

  @property
  def static_folder(self):
    return self.settings.get('static_path')

  def prepare(self):        
    # 设置最大流式数据量
    self.request.connection.set_max_body_size(MAX_STREAMED_SIZE)    
    headers = self.request.headers
    if 'worker_prefix' not in headers:
      return
        
    self.worker_prefix = headers['worker_prefix']   # 用于标识每个模型
    self.file_folder = os.path.join(self.static_folder, f"{self.worker_prefix}")
    if not os.path.exists(self.file_folder):
      os.makedirs(self.file_folder)
    self.file_name = f"{headers['id']}"             # variable ensemble.id

  def data_received(self, chunk):
      if self.file_name:
        if not os.path.exists(os.path.join(self.file_folder, self.file_name)):
          with open(os.path.join(self.file_folder, self.file_name), 'wb') as up:
              pass

        with open(os.path.join(self.file_folder, self.file_name), 'rb+') as up:
          up.seek(self.save_seek)
          up.write(chunk)
          self.save_seek = self.save_seek + len(chunk)

  def cal_shannon_entropy(self, preds, axis):
    uncertainty = -1.0 * np.sum(preds * np.log(preds + 1e-6), axis=axis, keepdims=True)
    return uncertainty

  @run_on_executor
  def avg(self, sample_id, value, weight):
    # 1.step 如果value=None，则直接进入等待聚合结果过程
    if value is None:
      self.consume_cond.acquire()
      while True:
        # 检查sample_id数据是否已经完成融合
        if 'sample' not in self.db or \
          sample_id not in self.db['sample'] or \
          self.db['sample'][sample_id]['ensemble'] == '':
          self.consume_cond.wait()
        else:
          break
      
      self.consume_cond.release()
      with open(self.db['sample'][sample_id]['ensemble'], 'rb') as fp:
        return fp.read()
    
    ###########################################################################
    # 2.step 检查sample_id是否已经在聚合过程中，或者已经完成聚合
    self.lock.acquire()
    is_in_process = 'sample' in self.db and \
        sample_id in self.db['sample'] and \
      'worker' in self.db['sample'] and \
      self.request.headers['worker_prefix'] in self.db['sample']['worker']
    self.lock.release()
    
    if is_in_process:
      # 仅有第二次调用才会进入这里
      self.consume_cond.acquire()
      while True:
        # 检查sample_id数据是否已经完成融合
        if self.db['sample'][sample_id]['ensemble'] == '':
          self.consume_cond.wait()          
        else:
          break
      
      self.consume_cond.release()
      with open(self.db['sample'][sample_id]['ensemble'], 'rb') as fp:
        return fp.read()
    ###########################################################################
    
    # 3.step sample_id 首次出现，进入积累数据并等待聚合过程    
    self.lock.acquire()
    # 数据积累操作（线程安全）
    if 'sample' not in self.db:
      self.db['sample'] = {}

    
    if sample_id not in self.db['sample']:
      self.db['sample'][sample_id] = {
        'data': [], 
        'count': self.producer_num, 
        'weight': [], 
        'worker': self.request.headers['worker_prefix'],
        'ensemble': ''}

    self.db['sample'][sample_id]['data'].append(value)
    self.db['sample'][sample_id]['weight'].append(weight)
    self.lock.release()
    
    self.cond.acquire()
    while True:
      if len(self.db['sample'][sample_id]['data']) == self.producer_num:
        self.cond.notify()
        break

      if len(self.db['sample'][sample_id]['data']) != self.producer_num:
        self.cond.wait()

    self.cond.release()

    # 到此，数据已经完备，开始做融合处理
    data = {}
    if self.uncertain_vote_cfg is not None and \
        len(self.db['sample'][sample_id]['data']) > 1:
      # 计算不确定度，并加权平均
      uncertain_axis = self.uncertain_vote_cfg['axis']
      all_uncertainty_org = []
      all_uncertainty_region = []
      
      keys = self.db['sample'][sample_id]['data'][0].keys()
      for key in keys:
        for sample_data_map, sample_weight_map in \
            zip(self.db['sample'][sample_id]['data'], self.db['sample'][sample_id]['weight']):
          
          sample_data = sample_data_map[key]
          sample_weight = sample_weight_map[key]
          # uncertainty_org 1,1,H,W
          uncertainty_org = self.cal_shannon_entropy(sample_data, uncertain_axis)
          uncertainty_region = np.ones(uncertainty_org.shape, dtype=np.float32)
          if self.uncertain_vote_cfg['thres'] != -1:
            uncertainty_region = (uncertainty_org > self.uncertain_vote_cfg['thres']).astype(np.float32)

          all_uncertainty_org.append(uncertainty_org)
          all_uncertainty_region.append(uncertainty_region)

        all_uncertainty_org = np.concatenate(all_uncertainty_org, axis=uncertain_axis)
        all_uncertainty_org = softmax(all_uncertainty_org, uncertain_axis)
        # certainty prob
        all_certainty_org = \
          np.split(1.0 - all_uncertainty_org,
                  indices_or_sections=all_uncertainty_org.shape[uncertain_axis],
                  axis=uncertain_axis)

        reweight_sample_weight = []
        for sample_weight_map, sample_certainty, sample_limited_region in \
            zip(self.db['sample'][sample_id]['weight'], all_certainty_org, all_uncertainty_region):
          sample_weight = sample_weight_map[key]
          certainty = sample_certainty * sample_limited_region + (1.0-sample_limited_region)
          reweight_sample_weight.append(sample_weight * certainty)

        weighted_data = 0.0
        weighted_norm = 0.0
        for sample_data_map, sample_weight in zip(self.db['sample'][sample_id]['data'],
                                              reweight_sample_weight):
          sample_data = sample_data_map[key]
          weighted_data += sample_data * sample_weight
          weighted_norm += sample_weight

        data[key] = weighted_data / weighted_norm
    else:
      # 加权平均
      weighted_data = 0.0
      weighted_norm = 0.0
      keys = self.db['sample'][sample_id]['data'][0].keys()
      
      for key in keys:
        for sample_data_map, sample_weight in zip(self.db['sample'][sample_id]['data'],
                                              self.db['sample'][sample_id]['weight']):
          sample_data = sample_data_map[key]
          weighted_data += sample_data * sample_weight
          weighted_norm += sample_weight

        data[key] = weighted_data / weighted_norm

    # 清空缓存 (线程安全)
    self.lock.acquire()
    self.db['sample'][sample_id]['count'] -= 1
    assert(self.db['sample'][sample_id]['count'] >= 0)
    if self.db['sample'][sample_id]['count'] == 0:
      # 保存聚合后结果
      self.consume_cond.acquire()
      if not os.path.exists(os.path.join(self.static_folder, 'ensemble')):
        os.makedirs(os.path.join(self.static_folder, 'ensemble'))
            
      ensemble_data_file_path = os.path.join(self.static_folder, 'ensemble', f"{sample_id}")
      with open(ensemble_data_file_path, 'wb') as fp:
        pickle.dump(data, fp)  
      
      self.db['sample'][sample_id].update({'data': [], 'count': 0, 'ensemble': ensemble_data_file_path})
      # 已经完成聚合数据的保存，通知可能处于等待的线程
      self.consume_cond.notifyAll()
      self.consume_cond.release()

    # 删除临时数据
    if os.path.exists(os.path.join(self.file_folder, self.file_name)):
      os.remove(os.path.join(self.file_folder, self.file_name))
    self.lock.release()
    return pickle.dumps(data)

  @gen.coroutine
  def post(self, *args, **kwargs):
    # id,name,value
    id = self.request.headers.get('id')
    feedback = self.request.headers.get('feedback')

    value = None
    weight = 1.0
    if self.worker_prefix is not None:
      if not os.path.exists(os.path.join(self.file_folder, self.file_name)):
        self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
        return

      # 模型数据
      with open(os.path.join(self.file_folder, self.file_name), 'rb') as fp:
        value = pickle.load(fp)
        
      # 模型权重
      weight = (float)(self.request.headers.get('weight', 1.0))

    # 计算加权值
    pickle_data_bytes = yield self.avg(id, value, weight)

    # 下载数据服务
    if feedback == 'True':
      self.set_header('Content-Type', 'application/octet-stream')
      self.set_header('Content-Disposition', f'attachment; filename=data')
      with BytesIO(pickle_data_bytes) as f:
        while True:
          data = f.read(MB)
          if not data:
              break
          self.write(data)

    self.finish()

  @property
  def cond(self):
    return self.settings['cond']

  @property
  def consume_cond(self):
    return self.settings['consume_cond']

  @property
  def producer_num(self):
      return self.settings['producer_num']

  @property
  def lock(self):
    return self.settings['lock']

  @property
  def uncertain_vote_cfg(self):
    return self.settings.get('uncertain_vote_cfg', None)


@tornado.web.stream_request_body
class PutGetApiHandler(BaseHandler):
  executor = ThreadPoolExecutor(16)
  cache = {}
  def __init__(self, *args, **kwargs):
    super(PutGetApiHandler, self).__init__(*args, **kwargs)
      
  def prepare(self):
    # 设置最大流式数据量
    self.request.connection.set_max_body_size(MAX_STREAMED_SIZE)
    headers = self.request.headers
    self.cond.acquire()
    if headers['id'] not in PutGetApiHandler.cache:
        PutGetApiHandler.cache[headers['id']] = {
            'status': False,
            'io':  BytesIO(),
            'consume': 0
        }
    self.cond.release()
          
  def data_received(self, chunk):            
    PutGetApiHandler.cache[self.request.headers['id']]['io'].write(chunk)
    
  @gen.coroutine
  def post(self):
    # here, 数据传输已经完成
    self.cond.acquire()
    PutGetApiHandler.cache[self.request.headers['id']]['status'] = True
    PutGetApiHandler.cache[self.request.headers['id']]['io'].seek(0, os.SEEK_SET)
    self.cond.notifyAll()
    self.cond.release()
    self.finish()
  
  @run_on_executor
  def wait_data_ready(self, id):
    self.cond.acquire()
    while True:
        if PutGetApiHandler.cache[id]['status']:
            break            
        self.cond.wait()
    self.cond.release()
    
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', f'attachment; filename={id}')
    f = copy.deepcopy(PutGetApiHandler.cache[id]['io'])
    while True:
        data = f.read(MB)
        if not data:
            break
          
        self.write(data)
        
    PutGetApiHandler.cache[id]['consume'] += 1
    if PutGetApiHandler.cache[id]['consume'] == self.consumer_num:
        # 已经完成所有消费,删除数据
        PutGetApiHandler.cache.pop(id)
    
  @property
  def cond(self):
    return self.settings['put_get_cond']
  
  @gen.coroutine
  def get(self):
    id = self.request.headers['id']
    
    yield self.wait_data_ready(id)
    self.finish()
      
  @property
  def consumer_num(self):
    return self.settings['consumer_num']

@tornado.web.stream_request_body
class DataApiHandler(BaseHandler):
  @property
  def dump_dir(self):
    return self.settings.get('dump_dir', None)

  @gen.coroutine
  def get(self, file_name):
    # 提供变量数据服务
    #   ave_result = msgpack.packb(ave_result, default=ms.encode)
    file_name = file_name[1:]
    file_path = os.path.join(self.dump_dir, 'ensemble', 'merge', file_name)
    if not os.path.exists(file_path):
      self.set_status(404)
      return

    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', f'attachment; filename={file_name}')
    with open(file_path, 'rb') as f:
      while True:
        data = f.read(MB)
        if not data:
            break
        self.write(data)

    self.finish()


class LiveApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def ensemble_server_start(
    dump_dir,
    server_port,
    worker_num,
    uncertain_vote_cfg=None):
  # uncertain_vote_cfg {'axis': 1, 'thres': 0}
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    static_dir = os.path.join(dump_dir, 'ensemble')
    if not os.path.exists(os.path.join(static_dir, 'static')):
      os.makedirs(os.path.join(static_dir, 'static'))

    # 2.step launch show server
    db = {}
    settings = {
      'static_path': os.path.join(static_dir, 'static'),
      'dump_dir': dump_dir,
      'port': server_port,
      'cookie_secret': str(uuid.uuid4()),
      'worker_num': worker_num,
      'cond': threading.Condition(),
      'consume_cond': threading.Condition(),
      'put_get_cond': threading.Condition(),
      'lock': threading.Lock(),
      'db': db,
      'uncertain_vote_cfg': uncertain_vote_cfg,
      'producer_num': worker_num,
      'consumer_num': worker_num
    }

    app = tornado.web.Application(handlers=[(r"/antgo/api/ensemble/avg/", AverageApiHandler),
                                            (r"/antgo/api/ensemble/put/", PutGetApiHandler),
                                            (r"/antgo/api/ensemble/get/", PutGetApiHandler),
                                            (r"/antgo/api/ping/", LiveApiHandler),
                                            (r"/antgo/api/ensemble/data/(.*)", DataApiHandler)],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('ensemble is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('ensemble stop server')
  except GracefulExitException:
    logger.info('ensemble server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass
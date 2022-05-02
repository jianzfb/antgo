# -*- coding: UTF-8 -*-
# @Time    : 2021/11/13 6:13 下午
# @File    : ensemble_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
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
import msgpack
import msgpack_numpy as ms


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
    self.worker_number = None
    self.file_folder = None
    self.file_name = None
    self.file_id = None

  @property
  def static_folder(self):
    return self.settings.get('static_path')

  def prepare(self):
    # 设置最大流式数据量
    self.request.connection.set_max_body_size(MAX_STREAMED_SIZE)

    # 根据ID，设置数据临时文件
    headers = self.request.headers
    self.worker_prefix = headers['worker_prefix']
    self.worker_number = headers['worker_number']
    self.file_id = headers['file_id']
    self.file_folder = os.path.join(self.static_folder, f"{self.worker_prefix}-{self.worker_number}")
    if not os.path.exists(self.file_folder):
      os.makedirs(self.file_folder)
    self.file_name = f"{headers['id']}.{headers['name']}"    # variable name.id

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
  def avg(self, sample_id, name, value, weight):
    self.cond.acquire()
    # 转换到numpy数据
    if 'sample' not in self.db:
      self.db['sample'] = {}

    if sample_id not in self.db['sample']:
      self.db['sample'][sample_id] = {}

    if name not in self.db['sample'][sample_id]:
      self.db['sample'][sample_id][name] = {'data': [], 'count': self.worker_num, 'weight': []}

    self.db['sample'][sample_id][name]['data'].append(value)
    self.db['sample'][sample_id][name]['weight'].append(weight)

    while True:
      if len(self.db['sample'][sample_id][name]['data']) == self.worker_num:
        self.cond.notify()
        break

      if len(self.db['sample'][sample_id][name]['data']) != self.worker_num:
        self.cond.wait()

    self.cond.release()

    data = None
    if self.uncertain_vote_cfg is not None and \
        len(self.db['sample'][sample_id][name]['data']) > 1 and \
        self.db['sample'][sample_id][name]['data'][0].shape[self.uncertain_vote_cfg['axis']] > 1:
      # 计算不确定度，并加权平均
      uncertain_axis = self.uncertain_vote_cfg['axis']
      all_uncertainty_org = []
      all_uncertainty_region = []
      for sample_data, sample_weight in \
          zip(self.db['sample'][sample_id][name]['data'], self.db['sample'][sample_id][name]['weight']):
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
      for sample_weight, sample_certainty, sample_limited_region in \
          zip(self.db['sample'][sample_id][name]['weight'], all_certainty_org, all_uncertainty_region):
        certainty = sample_certainty * sample_limited_region + (1.0-sample_limited_region)
        reweight_sample_weight.append(sample_weight * certainty)

      weighted_data = 0.0
      weighted_norm = 0.0
      for sample_data, sample_weight in zip(self.db['sample'][sample_id][name]['data'],
                                            reweight_sample_weight):
        weighted_data += sample_data * sample_weight
        weighted_norm += sample_weight

      data = weighted_data / weighted_norm
    else:
      # 加权平均
      weighted_data = 0.0
      weighted_norm = 0.0
      for sample_data, sample_weight in zip(self.db['sample'][sample_id][name]['data'],
                                            self.db['sample'][sample_id][name]['weight']):
        weighted_data += sample_data * sample_weight
        weighted_norm += sample_weight

      data = weighted_data / weighted_norm

    # 清空缓存
    self.lock.acquire()
    self.db['sample'][sample_id][name]['count'] -= 1
    if self.db['sample'][sample_id][name]['count'] == 0:
      self.db['sample'][sample_id][name] = {'data': [], 'count': self.worker_num}

    self.lock.release()
    return data

  @gen.coroutine
  def post(self, *args, **kwargs):
    # id,name,value
    id = self.request.headers.get('id')
    name = self.request.headers.get('name')
    weight = (float)(self.request.headers.get('weight'))
    feedback = self.request.headers.get('feedback')

    if not os.path.exists(os.path.join(self.file_folder, self.file_name)):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 加载数据
    with open(os.path.join(self.file_folder, self.file_name), 'rb') as fp:
      value = msgpack.unpackb(fp.read(), object_hook=ms.decode)

    if not self.data_record:
      # 删除临时文件
      os.remove(os.path.join(self.file_folder, self.file_name))

    # 计算加权值
    ave_result = yield self.avg(id, name, value, weight)

    # 下载数据服务
    if feedback == 'True':
      ave_result = msgpack.packb(ave_result, default=ms.encode)
      self.set_header('Content-Type', 'application/octet-stream')
      self.set_header('Content-Disposition', f'attachment; filename={self.file_id}')
      with BytesIO(ave_result) as f:
        while True:
          data = f.read(MB)
          if not data:
              break
          self.write(data)

    #
    self.finish()

  @property
  def worker_num(self):
    return self.settings['worker_num']

  @property
  def cond(self):
    return self.settings['cond']

  @property
  def lock(self):
    return self.settings['lock']

  @property
  def uncertain_vote_cfg(self):
    return self.settings.get('uncertain_vote_cfg', None)

  @property
  def data_record(self):
    return self.settings.get('data_record', False)


class LiveApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def ensemble_server_start(dump_dir, server_port, worker_num, uncertain_vote_cfg=None, enable_data_record=False):
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

    # 2.step launch web server
    db = {}
    settings = {
      'static_path': os.path.join(static_dir, 'static'),
      'port': server_port,
      'cookie_secret': str(uuid.uuid4()),
      'worker_num': worker_num,
      'cond': threading.Condition(),
      'lock': threading.Lock(),
      'db': db,
      'uncertain_vote_cfg': uncertain_vote_cfg,
      'data_record': enable_data_record
    }

    app = tornado.web.Application(handlers=[(r"/ensemble-api/avg/", AverageApiHandler),
                                            (r"/ensemble-api/live/", LiveApiHandler)],
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
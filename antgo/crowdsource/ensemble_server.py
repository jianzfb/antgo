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
from tornado.concurrent import run_on_executor
from concurrent.futures import ThreadPoolExecutor
import threading
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class AverageApiHandler(BaseHandler):
  executor = ThreadPoolExecutor(16)

  @run_on_executor
  def avg(self, sample_id, name, value):
    self.cond.acquire()
    # 转换到numpy数据
    if 'sample' not in self.db:
      self.db['sample'] = {}

    if sample_id not in self.db['sample']:
      self.db['sample'][sample_id] = {}

    if name not in self.db['sample'][sample_id]:
      self.db['sample'][sample_id][name] = {'data': [], 'count': self.worker_num}

    self.db['sample'][sample_id][name]['data'].append(value)

    while True:
      if len(self.db['sample'][sample_id][name]['data']) == self.worker_num:
        self.cond.notify()
        break

      if len(self.db['sample'][sample_id][name]['data']) != self.worker_num:
        self.cond.wait()

    self.cond.release()

    # 取平均
    data = self.db['sample'][sample_id][name]['data'][0]
    for index in range(1, self.worker_num):
      data += self.db['sample'][sample_id][name]['data'][index]
    data /= (float)(self.worker_num)

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
    id = self.get_json_argument('id', None)
    name = self.get_json_argument('name', None)
    value = self.get_json_argument('value', None)
    value = np.array(value)
    ave_result = yield self.avg(id, name, value)
    ave_result = ave_result.tolist()

    self.response(RESPONSE_STATUS_CODE.SUCCESS,content={
      'data': ave_result
    })

  @property
  def worker_num(self):
    return self.settings['worker_num']

  @property
  def cond(self):
    return self.settings['cond']

  @property
  def lock(self):
    return self.settings['lock']


class LiveApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)

class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def ensemble_server_start(dump_dir, server_port, worker_num):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    static_dir = os.path.join(dump_dir, 'ensemble')

    # 2.step launch web server
    db = {}
    settings = {
      'static_path': os.path.join(static_dir, 'static'),
      'port': server_port,
      'cookie_secret': str(uuid.uuid4()),
      'worker_num': worker_num,
      'cond': threading.Condition(),
      'lock': threading.Lock(),
      'db': db
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
# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:30
# @File    : watch_server.py
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
import traceback
import sys
import uuid
import json
import time
import tarfile


class IndexHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={})


class WatchApiHandler(BaseHandler):
  @property
  def data_queue(self):
    return self.settings['data_queue']

  @gen.coroutine
  def post(self, *args, **kwargs):
    datasource_type = self.get_argument('datasource_type', '')
    datasource_address = self.get_argument('datasource_address', '')
    datasource_keywards = self.get_argument('datasource_keywards', '')

    if datasource_type not in ['spider']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED)
      return

    if datasource_address not in ['baidu', 'bing']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED)
      return

    self.db['watch'].push({
      'watch_time': time.time(),
      'datasource_type': datasource_type,
      'datasource_address': datasource_address,
      'datasource_keywards': datasource_keywards,
      'watch_id': len(self.db['watch']),
      'watch_data': []
    })

    self.data_queue.push([datasource_type, {'datasource_address': datasource_address, 'datasource_keywards': datasource_keywards}])


class UpdateApiHandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    data_type = self.get_argument('processed_type', '')
    data_width = self.get_argument('processed_width', '')
    data_height = self.get_argument('processed_height', '')
    data = self.get_argument('processed_data', '')
    data_tag = self.get_argument('processed_tag', '')
    data_id = self.get_argument('processed_id', '')
    watch_id = self.get_argument('processed_watch_id', '')

    watch_id = (int)(watch_id)
    if watch_id >= len(self.db['watch']):
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED, message='watch id not in list')
      return

    self.db['watch'][watch_id]['watch_data'].push({
      'data_type': data_type,
      'data_width': data_width,
      'data_height': data_height,
      'data': data,
      'data_tag': data_tag,
      'data_id': data_id,
      'data_index': len(self.db['watch'][watch_id]['watch_data']),
      'data_del': False,
      'data_select': False
    })
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class DelApihandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    watch_id = self.get_argument('watch_id', '')
    watch_id = (int)(watch_id)
    if watch_id >= len(self.db['watch']):
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED, message='watch id not in list')
      return

    data_index = self.get_argument('data_index', '')
    data_index = (int)(data_index)
    if data_index >= len(self.db['watch'][watch_id]['watch_data']):
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED, message='data index not in list')
      return

    self.db['watch'][watch_id]['watch_data'][data_index]['data_del'] = True
    self.response(RESPONSE_STATUS_CODE.SUCCESS)

  @gen.coroutine
  def get(self, *args, **kwargs):
    del_data = []
    for watch_id in range(len(self.db['watch'])):
      for data in self.db['watch'][watch_id]['watch_data']:
        if data['data_del']:
          del_data.append(data)

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=del_data)


class SelectApiHandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    watch_id = self.get_argument('watch_id', '')
    watch_id = (int)(watch_id)
    if watch_id >= len(self.db['watch']):
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED, message='watch id not in list')
      return

    data_index = self.get_argument('data_index', '')
    data_index = (int)(data_index)
    if data_index >= len(self.db['watch'][watch_id]['watch_data']):
      self.response(RESPONSE_STATUS_CODE.REQUEST_NOT_ACCEPTED, message='data index not in list')
      return

    self.db['watch'][watch_id]['watch_data'][data_index]['data_select'] = True
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class TarApiHandler(BaseHandler):
  tar_v = 1
  @gen.coroutine
  def post(self, *args, **kwargs):
    tar = None
    for watch_id in range(len(self.db['watch'])):
      for data in self.db['watch'][watch_id]['watch_data']:
        if data['data_select']:
          if tar is None:
            tar_path = os.path.join(self.static_path, 'v-%d.tar.gz'%TarApiHandler.tar_v)
            tar = tarfile.open(tar_path, 'w:gz')

          tar.add(data['data'])
        pass

    TarApiHandler.tar_v += 1
    pass


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def watch_server_start(data_path,
                         browser_dump_dir,
                         response_queue,
                         tags,
                         server_port,
                         offset_configs,
                         profile_config,
                         browser_mode='screening'):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    browser_static_dir = os.path.join(browser_dump_dir, 'watch')
    # 数据数据目录
    browser_dump_dir = os.path.join(browser_dump_dir, 'record')
    if not os.path.exists(browser_dump_dir):
      os.makedirs(browser_dump_dir)

    # 2.step launch web server
    db = {'data': [], 'user': {}, 'dataset': {}}
    for offset_config in offset_configs:
      db['dataset'][offset_config['dataset_flag']] = {'offset':offset_config['dataset_offset']}

    db['dataset'][profile_config['dataset_flag']]['samples_num'] = profile_config['samples_num']
    db['dataset'][profile_config['dataset_flag']]['samples_num_checked'] = profile_config['samples_num_checked']
    db['state'] = profile_config['dataset_flag']

    settings = {
      'static_path': os.path.join(browser_static_dir, 'static'),
      'dump_path': browser_dump_dir,
      'port': server_port,
      'response_queue': response_queue,
      'tags': tags,
      'data_folder': data_path,
      'cookie_secret': str(uuid.uuid4()),
      'db': db,
      'browser_mode': browser_mode
    }

    app = tornado.web.Application(handlers=[(r"/browser-api/prev/", PrevApiHandler),
                                            (r"/browser-api/next/", NextApiHandler),
                                            (r"/browser-api/operator/", OperatorApiHandler),
                                            (r"/browser-api/entry/", EntryApiHandler),
                                            (r"/browser-api/file/", FileApiHandler),
                                            (r"/browser-api/download/", DownloadApiHandler),
                                            (r"/browser-api/config/", ConfigApiHandler),
                                            (r'/(.*)', tornado.web.StaticFileHandler,
                                             {"path": browser_static_dir, "default_filename": "index.html"}),],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('demo is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('demo stop server')
  except GracefulExitException:
    logger.info('demo server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass

# browser_server_start()
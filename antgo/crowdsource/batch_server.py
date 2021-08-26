# -*- coding: UTF-8 -*-
# @Time    : 2020-07-04 10:24
# @File    : batch_server.py
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


class EntryApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    content = self.db['content']
    if content is None:
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if 'dataset' not in content:
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    tags = content['tags']
    dataset_and_results = content['dataset']
    dataset_names = list(dataset_and_results.keys())
    dataset_name = dataset_names[0]

    query_dataset_name = self.get_argument('dataset_name', None)
    if query_dataset_name is not None:
      dataset_name = query_dataset_name
    result_num = len(dataset_and_results[dataset_name])
    current_dataset_name = dataset_name

    if dataset_name not in self.db:
      self.db[dataset_name] = {}

    # 每页只能显示100条记录(固定)
    num_in_page = 100
    # 获得需要展示的页数
    page_num = int(math.ceil(result_num / num_in_page))
    self.db[dataset_name]['num_in_page'] = num_in_page
    self.db[dataset_name]['page_num'] = page_num
    page_index = 0
    start_index = 0 * num_in_page
    start_index = start_index if start_index < result_num else result_num-1
    end_index = 1 * num_in_page
    end_inidex = end_index if end_index < result_num else result_num

    sample_num_in_dataset = {}
    for dataset_name in dataset_names:
      sample_num_in_dataset[dataset_name] = len(self.db['content']['dataset'][dataset_name])

    response_content = {
      'num_in_page': num_in_page,
      'page_num': page_num,
      'page_index': page_index,
      'dataset_name': dataset_names,
      'sample_num_in_dataset': sample_num_in_dataset,
      'current_dataset_name': current_dataset_name,
      'result_num': result_num,
      'result_list': dataset_and_results[dataset_name][start_index: end_index],
      'tags': tags,
      'waiting': self.db['waiting'],
      'finished': self.db['finished'],
      'spider': self.db['spider'] if 'spider' in self.db else 'BAIDU'
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)

class TagChangeApiHandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    page_index = self.get_argument('page_index', None)
    data_index = self.get_argument('data_index', None)
    data_tag = self.get_argument('data_tag', None)
    if page_index is None or data_index is None or data_tag is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
    
    page_index = (int)(page_index)
    data_index = (int)(data_index)
    data_tag = json.loads(data_tag)

    # 修改数据的tag
    dataset_and_results = self.db['content']['dataset']
    dataset_names = list(dataset_and_results.keys())
    dataset_name = dataset_names[0]
    num_in_page = self.db[dataset_name]['num_in_page']

    data_index = page_index * num_in_page + data_index
    self.db['content']['dataset'][dataset_name][data_index]['tag'] = data_tag
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class SearchApiHandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    web_url = self.get_argument('url', None)
    keyword = self.get_argument('keyword', None)
    if web_url.lower().startswith('baidu'):
      web_url = 'baidu'
    elif web_url.lower().startswith('bing'):
      web_url = 'bing'
    elif web_url.lower().startswith('google'):
      web_url = 'google'

    # 记录当前spider
    self.db['spider'] = web_url.upper()

    # 通知爬虫
    self.db['command_queue'].put((
      'baidu', {'datasource_address': web_url, 'datasource_keywards': keyword}
    ))
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class PageApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    page_index = self.get_argument('page_index', None)
    page_index = int(page_index)
    dataet_name = self.get_argument('dataset_name', None)
    if page_index is None or dataet_name is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    content = self.db['content']
    if content is None:
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if dataet_name not in self.db:
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if 'num_in_page' not in self.db[dataet_name] or 'page_num' not in self.db[dataet_name]:
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    if page_index >= self.db[dataet_name]['page_num']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    result_num = len(self.db['content']['dataset'][dataet_name])
    start_index = page_index * self.db[dataet_name]['num_in_page']
    start_index = start_index if start_index < result_num else result_num - 1
    end_index = (page_index + 1) * self.db[dataet_name]['num_in_page']
    end_inidex = end_index if end_index < result_num else result_num

    dataset_names = list(content['dataset'].keys())
    sample_num_in_dataset = {}
    for dataset_name in dataset_names:
      sample_num_in_dataset[dataset_name] = len(self.db['content']['dataset'][dataset_name])

    response_content = {
      'num_in_page': self.db[dataet_name]['num_in_page'],
      'page_num': self.db[dataet_name]['page_num'],
      'page_index': page_index,
      'dataset_name': dataset_names,
      'current_dataset_name': dataet_name,
      'sample_num_in_dataset': sample_num_in_dataset,
      'result_num': result_num,
      'result_list': content['dataset'][dataet_name][start_index: end_index],
      'waiting': self.db['waiting'],
      'finished': self.db['finished']
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


class ConfigApiHandler(BaseHandler):
  @gen.coroutine
  def post(self, *args, **kwargs):
    config_data = self.get_argument('config_data', None)
    if config_data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # config_data: 
    # {'experiment_uuid': experiment_uuid, 'dataset': {'train': [[{},{},...],[]]}, 'tags':[]}
    config_data = json.loads(config_data)
    if len(self.db['content']) == 0:
      self.db['content'] = config_data
    else:
      for key in config_data['dataset'].keys():
        self.db['content']['dataset'][key].extend(config_data['dataset'][key])

    self.db['waiting'] = config_data['waiting']
    self.db['finished'] = config_data['finished']
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class PingApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def batch_server_start(batch_dump_dir, server_port, command_queue):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    batch_static_dir = os.path.join(batch_dump_dir, 'batch')

    # 2.step launch web server
    db = {'content': {}, 'command_queue': command_queue}
    settings = {
      'static_path': os.path.join(batch_static_dir, 'static'),
      'port': server_port,
      'cookie_secret': str(uuid.uuid4()),
      'db': db,
    }

    app = tornado.web.Application(handlers=[(r"/batch-api/entry/", EntryApiHandler),
                                            (r"/batch-api/dataset/", EntryApiHandler),
                                            (r"/batch-api/page/", PageApiHandler),
                                            (r"/batch-api/config/", ConfigApiHandler),
                                            (r"/batch-api/ping/", PingApiHandler),
                                            (r"/batch-api/tag/", TagChangeApiHandler),
                                            (r"/batch-api/search/", SearchApiHandler),
                                            (r'/(.*)', tornado.web.StaticFileHandler,
                                             {"path": batch_static_dir, "default_filename": "index.html"}),],
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
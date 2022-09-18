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
import base64


class EntryApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

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
    end_index = end_index if end_index < result_num else result_num

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
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

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

    # # 通知爬虫
    # self.db['command_queue'].put((
    #   'baidu', {'datasource_address': web_url, 'datasource_keywards': keyword}
    # ))
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class PageApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

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

    statistic_info = {
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'user_name': user['full_name'],
      'short_name': user['short_name'],
      'task_name': 'DEFAULT',
      'task_type': 'DEFAULT',
      'project_type': 'PREDICT',
      'statistic_info': statistic_info
    })


class ProjectInfoHandler(BaseHandler):
  @gen.coroutine
  def get(self):

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'project_type': 'PREDICT',
      'project_state': {

      }
    })
    return


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def batch_server_start(dump_dir, server_port, samples=None, white_users=None):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    batch_dir = os.path.join(dump_dir, 'predict')
    static_dir = os.path.join(batch_dir, 'static')
    if not os.path.exists(static_dir):
      os.makedirs(static_dir)

    # 2.step launch show server
    # cookie
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)

    db = {'content': {}, 'users': {}}
    if samples is not None:
      db['content'] = samples
      db['waiting'] = samples['waiting']
      db['finished'] = samples['finished']

    # 设置白盒用户
    db['white_users'] = white_users

    settings = {
      'static_path': static_dir,
      'port': server_port,
      'cookie_secret': cookie_secret,
      'db': db,
    }

    app = tornado.web.Application(handlers=[(r"/antgo/api/predict/entry/", EntryApiHandler),
                                            (r"/antgo/api/predict/dataset/", EntryApiHandler),
                                            (r"/antgo/api/predict/page/", PageApiHandler),
                                            (r"/antgo/api/predict/config/", ConfigApiHandler),
                                            (r"/antgo/api/predict/tag/", TagChangeApiHandler),
                                            (r"/antgo/api/predict/search/", SearchApiHandler),
                                            (r"/antgo/api/user/login/", LoginHandler),    # 登录，仅支持预先指定用户
                                            (r"/antgo/api/user/logout/", LogoutHandler),  # 退出
                                            (r"/antgo/api/user/info/", UserInfoHandler),  # 获得用户信息
                                            (r"/antgo/api/ping/", PingApiHandler),        # ping 服务
                                            (r"/antgo/api/info/", ProjectInfoHandler),
                                            (r'/(.*)', tornado.web.StaticFileHandler,
                                             {"path": static_dir, "default_filename": "index.html"}),],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('predict is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('predict stop server')
  except GracefulExitException:
    logger.info('predict server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  dump_dir = '/Users/jian/Downloads/BB'
  white_users = {
    'jian@baidu.com':{
      'password': '112233'
    }
  }

  samples = {
    'waiting': 0,
    'finished': 2000,
    'experiment_uuid': '112233445566',
    'dataset': {
      'train': []
    },
    'tags': ['1','2','3','4']
  }

  for _ in range(2000):
    samples['dataset']['train'].append(
      {
        'data': [
          {
            'type': 'IMAGE',
            'data': '/static/data/1.jpeg',
            'width': 1200 // 4,
            'height': 800 // 4,
          },
          {
            'type': 'STRING',
            'data': 'AABBCC',
          }
        ],
        'tag': []
      },
    )

  batch_server_start(dump_dir, 9000, samples=samples, white_users=white_users)

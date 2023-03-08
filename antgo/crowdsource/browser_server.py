# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:30
# @File    : browser_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import copy

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
import base64
# # 新用户，从队列中获取新数据并加入用户队列中
# self.db['data'].append({
#   'value': self.response_queue.get(),
#   'status': False,
#   'time': time.time()
# })

class FreshApiHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 添加样本记录
    samples = self.get_argument('samples', None)
    if samples is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    # 解析json
    samples = json.loads(samples)

    is_replace = self.get_argument('is_replace', None)
    if is_replace:
      self.db['data'] = []

    for sample_i, sample in enumerate(samples):
      for data in sample:
        if data['type'] == 'IMAGE':
          large_size = max(data['width'], data['height'])
          scale = large_size / 400
          if scale > 1.0:
            data['width'] = (int)(data['width'] / scale)
            data['height'] = (int)(data['height'] / scale)

      self.db['data'].append({
        'value': sample,
        'status': False,
        'time': time.time()
      })

    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class EntryApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    if len(self.db['data']) == 0:
      # 当前无数据，直接返回
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'][user_name] = []
    if len(self.db['user_record'][user_name]) == 0:
      self.db['user_record'][user_name].append(0)

    # 获得当前用户浏览位置
    entry_id = self.db['user_record'][user_name][-1]

    # 构建返回数据
    response_content = {
      'value': self.db['data'][entry_id]['value'],
      'step': len(self.db['user_record'][user_name]) - 1,
      'tags': self.settings.get('tags', []),
      'operators': [],
      'dataset_flag': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
    return


class PrevApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    step = self.get_argument('step', None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'] = []
    step = int(step)
    if step < 0 or step >= len(self.db['user_record'][user_name]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user_record'][user_name][step]
    self.db['data'][entry_id] = {'value': data, 'status': True, 'time': time.time()}

    # 保存当前修改到文件
    # 使用 offset
    state = self.db['state']
    offset = self.db['dataset'][state]['offset']
    if not os.path.exists(os.path.join(self.dump_folder, state)):
      os.makedirs(os.path.join(self.dump_folder, state))

    try:
      # 发现数据id
      data_id = None
      for item in data:
        if item['title'] == 'ID':
          data_id = str(item['data'])
      if data_id is None:
        data_id = str(entry_id+offset)

      with open(os.path.join(self.dump_folder, state, '%s.json'%data_id), "w") as file_obj:
        json.dump(data, file_obj)
    except Exception as e:
      print('str(Exception):\t', str(Exception))
      print('str(e):\t\t', str(e))
      print('repr(e):\t', repr(e))
      print('e.message:\t', e.message)

    # 获得当前用户上一步数据
    if step == 0:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'dont have pre step')
      return

    pre_step = step - 1
    pre_entry_id = self.db['user_record'][user_name][pre_step]
    response_content = {
      'value': self.db['data'][pre_entry_id]['value'],
      'step': pre_step,
      'tags': self.settings.get('tags', []),
      'operators': [],
      'state': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


class NextApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    user = self.get_current_user()
    if user is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
      return

    step = self.get_argument("step", None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    user_name = user['full_name']
    if user_name not in self.db['user_record']:
      self.db['user_record'] = []
    step = int(step)
    if step < 0 or step >= len(self.db['user_record'][user_name]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user_record'][user_name][step]
    if not self.db['data'][entry_id]['status']:
      self.db['dataset'][self.db['state']]['samples_num_checked'] += 1

    self.db['data'][entry_id] = {'value': data, 'status': True, 'time': time.time()}

    # 保存当前修改到文件
    # 使用 offset
    state = self.db['state']
    offset = self.db['dataset'][state]['offset']
    if not os.path.exists(os.path.join(self.dump_folder, state)):
      os.makedirs(os.path.join(self.dump_folder, state))

    try:
      # 发现数据id
      data_id = None
      for item in data:
        assert('id' in item or 'ID' in item)
        if 'id' in item:
          data_id = str(item['id'])
        if 'ID' in item:
          data_id = str(item['ID'])

      if data_id is None:
        data_id = str(entry_id + offset)

      with open(os.path.join(self.dump_folder, state, '%s.json' % data_id), "w") as fp:
        json.dump(data, fp)
    except Exception as e:
      print('str(Exception):\t', str(Exception))
      print('str(e):\t\t', str(e))
      print('repr(e):\t', repr(e))
      print('e.message:\t', e.message)

    # 获得用户下一步数据
    if step < len(self.db['user_record'][user_name]) - 1:
      next_step = step + 1
      next_entry_id = self.db['user_record'][user_name][next_step]
      response_content = {
        'value': self.db['data'][next_entry_id]['value'],
        'step': next_step,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
      return

    # 发下下一个还没有进行审核的样本
    next_entry_id = -1
    for id in range(len(self.db['data'])):
      if 'status' not in self.db['data'][id] or not self.db['data'][id]['status']:
        next_entry_id = id
        break

    if next_entry_id == -1:
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
        'value': self.db['data'][self.db['user_record'][user_name][-1]]['value'],
        'step': len(self.db['user_record'][user_name]) - 1,
        'tags': self.settings.get('tags', []),
        'operators': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      })
      return

    # 为当前用户分配下一个审查数据
    self.db['user_record'][user_name].append(next_entry_id)

    #
    response_content = {
      'value': self.db['data'][next_entry_id]['value'],
      'step': len(self.db['user_record'][user_name]) - 1,
      'tags': self.settings.get('tags', []),
      'operators': [],
      'state': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
    return


class FileApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 获得代码目录结构
    root_path = self.data_folder
    files_tree = [{
      'name': ".",
      'type': "folder",
      'size': "",
      'folder': [],
      'path': ''
    }]
    queue = [files_tree[-1]]
    while len(queue) != 0:
      folder = queue.pop(-1)
      folder_path = os.path.join(root_path, folder['path'])
      for f in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, f)):
          folder['folder'].append({
            'name': f,
            'type': "folder",
            'size': "",
            'folder': [],
            'path': '%s/%s' % (folder['path'], f) if folder['path'] != '' else f,
          })

          queue.append(folder['folder'][-1])
        else:
          fsize = os.path.getsize(os.path.join(folder_path, f))
          fsize = fsize / 1024.0  # KB
          folder['folder'].append({
            'name': f,
            'type': "file",
            'size': "%0.2f KB" % round(fsize, 2),
            'path': '%s/%s' % (folder['path'], f) if folder['path'] != '' else f,
          })

    response = {
      'files_tree': files_tree
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response)


class DownloadApiHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    if not os.path.exists(os.path.join(self.dump_folder, self.db['state'])):
      self.finish()
      return
    
    package_data = {}
    for file_name in os.listdir(os.path.join(self.dump_folder, self.db['state'])):
      if not file_name.endswith('json'):
        continue
      
      data_id = file_name.split('.')[0]
      with open(os.path.join(self.dump_folder, self.db['state'], file_name), 'r') as fp:
        package_data[data_id] = json.load(fp)
              
    now_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    download_file = f'{now_time}.json'
    download_path = os.path.join(self.dump_folder, download_file)
    with open(download_path, 'w') as fp:
      json.dump(package_data, fp)

    # Content-Type
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', 'attachment; filename=' + download_file)

    buffer_size = 64 * 1024
    with open(download_path, 'rb') as fp:
      content = fp.read()
      content_size = len(content)
      buffer_segments = content_size // buffer_size
      for buffer_seg_i in range(buffer_segments):
        buffer_data = content[buffer_seg_i * buffer_size: (buffer_seg_i + 1) * buffer_size]
        yield self.write(buffer_data)

      yield self.write(content[buffer_segments * buffer_size:])

    self.finish()


class OperatorApiHandler(BaseHandler):
  def post(self):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class ConfigApiHandler(BaseHandler):
  def post(self):
    offset_config = self.get_argument('offset_config', None)
    if offset_config is not None:
      offset_config = json.loads(offset_config)
      dataset_flag = offset_config['dataset_flag']
      dataset_offset = offset_config['dataset_offset']
      if 'dataset' not in self.db:
        self.db['dataset'] = {}
      if dataset_flag not in self.db['dataset']:
        self.db['dataset'][dataset_flag] = {}

      self.db['dataset'][dataset_flag]['offset'] = dataset_offset

    profile_config = self.get_argument('profile_config', None)
    if profile_config is not None:
      profile_config = json.loads(profile_config)
      dataset_flag = profile_config['dataset_flag']
      samples_num = profile_config['samples_num']
      samples_num_checked = profile_config['samples_num_checked']

      if 'dataset' not in self.db:
        self.db['dataset'] = {}
      if dataset_flag not in self.db['dataset']:
        self.db['dataset'][dataset_flag] = {}

      self.db['dataset'][dataset_flag]['samples_num'] = samples_num
      self.db['dataset'][dataset_flag]['samples_num_checked'] = samples_num_checked

    state = self.get_argument('state', None)
    if state is not None:
      # train,val or test
      self.db['state'] = state

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

    # 遍历所有样本获得本轮，当前用户的标注记录
    statistic_info = {
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'user_name': user['full_name'],
      'short_name': user['short_name'],
      'task_name': 'DEFAULT',
      'task_type': 'DEFAULT',
      'project_type': 'BROWSER',
      'statistic_info': statistic_info
    })


class ProjectInfoHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
      'project_type': 'BROWSER',
      'project_state': {
        'stage': \
          'finish' if len(self.db['data']) > 0 and \
            self.db['dataset'][self.db['state']]['samples_num_checked'] == len(self.db['data'])  else 'checking'
      }
    })
    return

class PingApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    self.response(RESPONSE_STATUS_CODE.SUCCESS)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def browser_server_start(browser_dump_dir,
                         tags,
                         server_port,
                         offset_configs,
                         profile_config,
                         sample_folder=None, sample_list=None,
                         white_users=None):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  try:
    # 静态资源目录
    browser_dir = os.path.join(browser_dump_dir, 'browser')
    static_dir = os.path.join(browser_dir, 'static')
    if not os.path.exists(static_dir):
      os.makedirs(static_dir)

    # 数据数据目录
    browser_dump_dir = os.path.join(browser_dump_dir, 'record')
    if not os.path.exists(browser_dump_dir):
      os.makedirs(browser_dump_dir)

    # 2.step launch show server
    db = {'data': [], 'users': {}, 'dataset': {}, 'user_record': {}}

    if sample_list is not None and sample_folder is not None:
      # 为样本所在目录建立软连接到static下面
      os.system(f'cd {static_dir}; ln -s {sample_folder} dataset;')
      
      # 将数据信息写如本地数据库
      for sample in sample_list:
        file_name = sample['image_file'].split('/')[-1] if sample['image_file'] != '' else sample['image_url'].split('/')[-1]
        convert_sample = {
          'type': 'IMAGE',
          'data': f'/static/dataset/{sample["image_file"]}' if sample['image_file'] != '' else sample['image_url'],
          'width': 256,
          'height': 256,
          'tag': [],
          'title': file_name
        }
        db['data'].append({
          'value': convert_sample,
          'status': False,
          'time': time.time()
        })

    # 设置白盒用户
    db['white_users'] = white_users

    for offset_config in offset_configs:
      db['dataset'][offset_config['dataset_flag']] = {
        'offset': offset_config['dataset_offset']
      }

    db['dataset'][profile_config['dataset_flag']]['samples_num'] = \
      profile_config['samples_num']
    db['dataset'][profile_config['dataset_flag']]['samples_num_checked'] = \
      profile_config['samples_num_checked']
    db['state'] = profile_config['dataset_flag']

    # cookie
    cookie_secret = base64.b64encode(uuid.uuid4().bytes + uuid.uuid4().bytes)

    settings = {
      'static_path': os.path.join(browser_dir, 'static'),
      'dump_path': browser_dump_dir,
      'port': server_port,
      'tags': tags,
      'cookie_secret': cookie_secret,
      'cookie_max_age_days': 30,
      'Content-Security-Policy': "frame-ancestors 'self' {}".format('http://localhost:8080/'),
      'db': db,
    }

    app = tornado.web.Application(handlers=[(r"/antgo/api/browser/sample/prev/", PrevApiHandler),
                                            (r"/antgo/api/browser/sample/next/", NextApiHandler),
                                            (r'/antgo/api/browser/sample/fresh/', FreshApiHandler),
                                            (r"/antgo/api/browser/sample/entry/", EntryApiHandler),
                                            (r"/antgo/api/browser/operators/", OperatorApiHandler),
                                            (r"/antgo/api/browser/file/", FileApiHandler),
                                            (r"/antgo/api/browser/download/", DownloadApiHandler),
                                            (r"/antgo/api/browser/config/", ConfigApiHandler),
                                            (r"/antgo/api/ping/", PingApiHandler),
                                            (r"/antgo/api/user/login/", LoginHandler),    # 登录，仅支持预先指定用户
                                            (r"/antgo/api/user/logout/", LogoutHandler),  # 退出
                                            (r"/antgo/api/user/info/", UserInfoHandler),  # 获得用户信息
                                            (r"/antgo/api/info/", ProjectInfoHandler),
                                            (r'/(.*)', tornado.web.StaticFileHandler,
                                             {"path": static_dir, "default_filename": "index.html"}),],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('browser is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('browser stop server')
  except GracefulExitException:
    logger.info('browser server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    pass


if __name__ == '__main__':
  data_path = ''
  browser_dump_dir = '/Users/jian/Downloads/BB'
  tags = ['A','B','D']
  server_port = 9000
  offset_configs = [{
    'dataset_flag': 'TRAIN',
    'dataset_offset': 0
  }, {
    'dataset_flag': 'VAL',
    'dataset_offset': 0
  }, {
    'dataset_flag': 'TEST',
    'dataset_offset': 0
  }]
  profile_config = {
    'dataset_flag': 'TRAIN',
    'samples_num': 10,
    'samples_num_checked': 0
  }
  white_users = {
    'jian@baidu.com':{
      'password': '112233'
    }
  }
  samples=[]
  for _ in range(10):
    temp = [
      {
        'type': 'IMAGE',
        'data': '/static/data/1.jpeg',
        'width': 1200//4,
        'height': 800//4,
        'tag': ['A'],
        'title': 'MIAO'
      },
      {
        'type': 'STRING',
        'data': 'AABBCC',
        'tag': ['B'],
        'title': 'H'
      }
    ]

    samples.append(copy.deepcopy(temp))

  browser_server_start(
    browser_dump_dir,
    tags,
    server_port,
    offset_configs,
    profile_config,
    white_users=white_users,
    sample_list=samples, sample_folder='')

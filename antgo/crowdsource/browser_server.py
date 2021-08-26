# -*- coding: UTF-8 -*-
# @Time    : 2020-06-25 23:30
# @File    : browser_server.py
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


class EntryApiHandler(BaseHandler):
  @gen.coroutine
  def get(self, *args, **kwargs):
    # notify backend and waiting response
    session_id = self.get_cookie('sessionid')
    if session_id is None:
      session_id = str(uuid.uuid4())
      self.set_secure_cookie('sessionid', session_id)

    if session_id not in self.db['user']:
      self.db['user'][session_id] = []

    # browser/screening
    if self.settings['browser_mode'] == 'browser':
      # 浏览模式
      if len(self.db['user'][session_id]) == 0:
        if len(self.db['data']) == 0:
          if self.response_queue.empty():
            # 无新数据，直接返回
            self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
            return

          # 从队列取出，加入数据库
          self.db['data'].append({
            'value': self.response_queue.get(),
            'status': False,
            'time': time.time()
          })

        self.db['user'][session_id].append(0)

      # 获得当前用户浏览位置
      entry_id = self.db['user'][session_id][-1]

      # 构建返回数据
      response_content = {
        'value': self.db['data'][entry_id]['value'],
        'step': len(self.db['user'][session_id]) - 1,
        'tags': self.settings.get('tags', []),
        'operator': [],
        'dataset_flag': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }

      self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
      return

    # 审查模式
    if len(self.db['user'][session_id]) == 0:
      if self.response_queue.empty():
        # 无新数据，直接返回
        self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND)
        return

      # 新用户，从队列中获取新数据并加入用户队列中
      self.db['data'].append({
        'value': self.response_queue.get(),
        'status': False,
        'time': time.time()
      })

      latest_id = len(self.db['data']) - 1
      self.db['user'][session_id].append(latest_id)

    # 获得当前用户待审查数据ID
    entry_id = self.db['user'][session_id][-1]

    # 构建返回数据
    response_content = {
      'value': self.db['data'][entry_id]['value'],
      'step': len(self.db['user'][session_id]) - 1,
      'tags': self.settings.get('tags', []),
      'operator': [],
      'dataset_flag': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }

    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


class PrevApiHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    step = self.get_argument('step', None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    session_id = self.get_cookie('sessionid')
    if session_id is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'cookie invalid')
      return

    if session_id not in self.db['user']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'cookie invalid')
      return

    step = int(step)
    if step < 0 or step >= len(self.db['user'][session_id]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user'][session_id][step]
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
    pre_entry_id = self.db['user'][session_id][pre_step]
    response_content = {
      'value': self.db['data'][pre_entry_id]['value'],
      'step': pre_step,
      'tags': self.settings.get('tags', []),
      'operator': [],
      'state': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


class NextApiHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    step = self.get_argument("step", None)
    data = self.get_argument("data", None)
    if step is None or data is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    session_id = self.get_cookie('sessionid')
    if session_id is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'cookie invalid')
      return

    if session_id not in self.db['user']:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'cookie invalid')
      return

    step = int(step)
    if step < 0 or step >= len(self.db['user'][session_id]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user'][session_id][step]
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

    # 获得用户下一步数据
    if step < len(self.db['user'][session_id]) - 1:
      next_step = step + 1
      next_entry_id = self.db['user'][session_id][next_step]
      response_content = {
        'value': self.db['data'][next_entry_id]['value'],
        'step': next_step,
        'tags': self.settings.get('tags', []),
        'operator': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      }
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
      return

    if self.settings['browser_mode'] == 'browser':
      # 浏览模式
      if len(self.db['data']) == len(self.db['user'][session_id]):
        if self.response_queue.empty():
          # 无新数据，直接返回
          self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
            'value': self.db['data'][self.db['user'][session_id][-1]]['value'],
            'step': len(self.db['user'][session_id]) - 1,
            'tags': self.settings.get('tags', []),
            'operator': [],
            'state': self.db['state'],
            'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
            'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
          })
          return

        # 加入新数据
        self.db['data'].append({'value': self.response_queue.get(),
                                'status': False,
                                'time': time.time()})

      # 为当前用户分配下一个浏览数据
      next_entry_id = len(self.db['user'][session_id])
      self.db['user'][session_id].append(next_entry_id)

      self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
        'value': self.db['data'][next_entry_id]['value'],
        'step': len(self.db['user'][session_id]) - 1,
        'tags': self.settings.get('tags', []),
        'operator': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      })

      return

    # 审查模式
    # 当前已经到达用户最后一张图，下一张图必须从队列中获取
    # 新数据来源（1，之前分配给某个用户，但一直没有收到反馈；2，新产生的数据）
    # 1. 之前分配给某个用户，但一直没有收到反馈
    next_entry_id = -1
    now_time = time.time()
    for sample_i, sample in enumerate(self.db['data']):
      if not sample['status'] and (now_time - sample['time']) > 60:
        next_entry_id = sample_i
        break

    # 2. 新产生的数据
    if self.response_queue.empty():
      # 无新数据，直接返回
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
        'value': self.db['data'][self.db['user'][session_id][-1]]['value'],
        'step': len(self.db['user'][session_id]) - 1,
        'tags': self.settings.get('tags', []),
        'operator': [],
        'state': self.db['state'],
        'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
        'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
      })
      return

    if next_entry_id == -1:
      self.db['data'].append({'value': self.response_queue.get(), 'status': False, 'time': time.time()})
      next_entry_id = len(self.db['data']) - 1

    # 为当前用户分配下一个审查数据
    self.db['user'][session_id].append(next_entry_id)

    #
    response_content = {
      'value': self.db['data'][next_entry_id]['value'],
      'step': len(self.db['user'][session_id]) - 1,
      'tags': self.settings.get('tags', []),
      'operator': [],
      'state': self.db['state'],
      'samples_num': self.db['dataset'][self.db['state']]['samples_num'],
      'samples_num_checked': self.db['dataset'][self.db['state']]['samples_num_checked'],
    }

    #
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)


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
    download_file = self.get_argument('path', None)
    if download_file is None:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'not set download file')
      return

    download_path = os.path.join(self.data_folder, download_file)
    if not os.path.exists(download_path):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'not download file')
      return

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


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def browser_server_start(data_path,
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
    browser_static_dir = os.path.join(browser_dump_dir, 'browser')
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
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
import sys
import uuid
import json


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

    if len(self.db['user'][session_id]) == 0:
      if len(self.db['data']) == 0:
        self.db['data'].append({
          'value': self.response_queue.get(),
          'status': False
        })

      latest_id = len(self.db['data']) - 1
      self.db['user'][session_id].append(latest_id)

    entry_id = self.db['user'][session_id][-1]
    response_content = {
      'value': self.db['data'][entry_id]['value'],
      'step': len(self.db['user'][session_id]) - 1,
      'tags': self.settings.get('tags', []),
      'operator': []
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

    step = int(step)
    if step < 0 or step >= len(self.db['user'][session_id]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user'][session_id][step]
    self.db['data'][entry_id] = {'value': data, 'status': True}

    # 保存当前修改到文件
    with open(os.path.join(self.dump_folder, '%d.json'%entry_id), "w") as file_obj:
      json.dump(data, file_obj)

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
      'operator': []
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

    step = int(step)
    if step < 0 or step >= len(self.db['user'][session_id]):
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'request parameter wrong')
      return

    # 保存当前修改到内存
    data = json.loads(data)
    entry_id = self.db['user'][session_id][step]
    self.db['data'][entry_id] = {'value': data, 'status': True}

    # 保存当前修改到文件
    with open(os.path.join(self.dump_folder, '%d.json'%entry_id), "w") as file_obj:
      json.dump(data, file_obj)

    # 获得用户下一步数据
    if step < len(self.db['user'][session_id]) - 1:
      next_step = step + 1
      next_entry_id = self.db['user'][session_id][next_step]
      response_content = {
        'value': self.db['data'][next_entry_id]['value'],
        'step': next_step,
        'tags': self.settings.get('tags', []),
        'operator': []
      }
      self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)
      return

    # 重新获取数据
    if entry_id == len(self.db['data']) - 1:
      self.db['data'].append({'value': self.response_queue.get(), 'status': False})

    next_entry_id = -1
    for index in range(entry_id + 1, len(self.db['data'])):
      if self.db['data'][index]['status'] == False:
        next_entry_id = index
        break

    if next_entry_id == -1:
      self.db['data'].append({'value': self.response_queue.get(), 'status': False})
      next_entry_id = len(self.db['data']) - 1

    self.db['user'][session_id].append(next_entry_id)

    response_content = {
      'value': self.db['data'][next_entry_id]['value'],
      'step': len(self.db['user'][session_id]) - 1,
      'tags': self.settings.get('tags', []),
      'operator': []
    }
    print(self.db['data'][next_entry_id]['value'])
    self.response(RESPONSE_STATUS_CODE.SUCCESS, content=response_content)

    # # notify backend and waiting response
    # next_data =  [{
    #   'data': '/static/image/banner-multi-measure.png',
    #   'type': 'image',
    #   'title': 'title',
    #   'tag': ['B', 'D']
    # },
    #   {
    #     'data': '/static/image/banner-multi-measure.png',
    #     'type': 'image',
    #     'title': 'xxx',
    #     'tag': ['A'],
    #   },]
    # operator = ['del']
    # tags=['VV','E']
    #
    # self.response(RESPONSE_STATUS_CODE.SUCCESS, content={
    #   'value': next_data,
    #   'operator': operator,
    #   'tags': tags,
    #   'step': 0,
    # })


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


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def browser_server_start(data_path, browser_dump_dir, response_queue, tags, server_port):
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
    db = {'data': [], 'user': {}}
    settings = {
      'static_path': os.path.join(browser_static_dir, 'static'),
      'dump_path': browser_dump_dir,
      'port': server_port,
      'response_queue': response_queue,
      'tags': tags,
      'data_folder': data_path,
      'cookie_secret': str(uuid.uuid4()),
      'db': db
    }

    app = tornado.web.Application(handlers=[(r"/browser-api/prev/", PrevApiHandler),
                                            (r"/browser-api/next/", NextApiHandler),
                                            (r"/browser-api/operator/", OperatorApiHandler),
                                            (r"/browser-api/entry/", EntryApiHandler),
                                            (r"/browser-api/file/", FileApiHandler),
                                            (r"/browser-api/download/", DownloadApiHandler),
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
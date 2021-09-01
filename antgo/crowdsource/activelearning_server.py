# -*- coding: UTF-8 -*-
# @Time    : 2019/1/23 7:12 PM
# @File    : activelearning_server.py
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
from antgo.crowdsource.base_server import *
import tornado.web
import os
import shutil
import json
import numpy as np
import uuid
import signal
import functools
import base64
import time


class UploadHandler(BaseHandler):
  def post(self):
    # 1.step 获取第几轮索引
    round_index = self.db.get('round', -1)
    if round_index == -1:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'no round in db')
      return

    # 2.step 保存上传数据(已标注)
    slice_index = self.get_argument('sliceIndex')
    slice_index = int(slice_index)
    slice_num = self.get_argument('sliceNum')
    slice_num = int(slice_num)
    file_name = self.get_argument('fileName')

    file_metas = self.request.files['file']
    for meta in file_metas:
      filepath = os.path.join(self.upload_folder, file_name)
      if slice_index == 0:
        with open(filepath, 'wb') as fp:
          fp.write(meta['body'])
      else:
        with open(filepath, 'ab') as fp:
          fp.write(meta['body'])

      self.write('finished!')
      break

    if slice_index == slice_num - 1:
      # 3.step 修改当前状态
      self.db["process_state"] = 'LABEL-CHECK'

      # 4.step notify backend
      self.request_queue.put({'ROUND': round_index, 'FILE': file_name})


class DownloadHandler(BaseHandler):
  @gen.coroutine
  def get(self):
    # 1.step 获取第几轮索引
    round_index = self.db.get('round', -1)
    if round_index == -1:
      self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID, 'no round in db')
      return

    # 2.step 下载数据(未标注)
    unlabeled_dataset = self.db.get('unlabel_dataset', None)
    if unlabeled_dataset is None:
      unlabeled_dataset = 'round_%d.tar.gz'%round_index

    if not os.path.exists(os.path.join(self.download_folder, unlabeled_dataset)):
      self.response(RESPONSE_STATUS_CODE.RESOURCE_NOT_FOUND, 'unlabeled data dont exist')
      return

    # 3.step download
    # Content-Type
    self.set_header('Content-Type', 'application/octet-stream')
    self.set_header('Content-Disposition', 'attachment; filename=' + unlabeled_dataset)

    buffer_size = 64 * 1024
    with open(os.path.join(self.download_folder, unlabeled_dataset), 'rb') as fp:
      content = fp.read()
      content_size = len(content)
      buffer_segments = content_size // buffer_size
      for buffer_seg_i in range(buffer_segments):
        buffer_data = content[buffer_seg_i * buffer_size: (buffer_seg_i + 1) * buffer_size]
        yield self.write(buffer_data)

      yield self.write(content[buffer_segments * buffer_size:])

    # 4.step 返回
    self.finish()


class ActiveLearningState(BaseHandler):
  def patch(self, *args, **kwargs):
    # 1.step update round
    round_index = self.get_argument('round', None)
    if round_index is not None:
      self.db['round'] = int(round_index)

    # 2.step update processing state
    process_state = self.get_argument('process_state', None)
    if process_state is not None:
      self.db["process_state"] = process_state

      if process_state == 'LABEL-FINISH':
        next_round_waiting = self.get_argument('next_round_waiting', None)
        if next_round_waiting is None:
          self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
          return

        self.db["next_round_waiting"] = next_round_waiting
        self.db['time'] = time.time()

      if process_state == 'UNLABEL-RESET':
        unlabel_dataset = self.get_argument('unlabel_dataset', None)
        if unlabel_dataset is None:
          self.response(RESPONSE_STATUS_CODE.REQUEST_INVALID)
          return

        self.db['unlabel_dataset'] = unlabel_dataset

    # 3.step unlabeled data number
    unlabeled_num = self.get_argument('unlabeled_size', None)
    if unlabeled_num is not None:
      self.db['unlabeled_size'] = unlabeled_num

    # 4.step labeled data number
    labeled_num = self.get_argument('labeled_size', None)
    if labeled_num is not None:
      self.db['labeled_size'] = labeled_num

    # 5.step round data number
    round_size = self.get_argument('round_size', None)
    if round_size is not None:
      self.db['round_size'] = round_size

    self.response(RESPONSE_STATUS_CODE.RESOURCE_SUCCESS_CREATED)


class IndexHandler(BaseHandler):
  def get(self):
    process_state = self.db.get('process_state', "UNLABEL-PREPARE")
    info = {'STATE': process_state}
    if process_state == 'UNLABEL-PREPARE' or process_state == "LABEL-FINISH":
      waiting_time = self.db.get('next_round_waiting', 0)
      start_time = self.db.get('time', 0)
      if waiting_time == 0 or start_time == 0:
        info['WAITING_TIME'] = 0
      else:
        info['WAITING_TIME'] = (waiting_time - (time.time() - start_time)) / 60

    if process_state == 'LABEL-ERROR':
      info['MESSAGE'] = "label error, please update label"

    labeled_size = self.get_argument('labeled_size', 0)
    unlabeled_size = self.get_argument('unlabeled_size', 0)
    round_size = self.get_argument('round_size', 0)
    info['LABELED_SIZE'] = labeled_size
    info['UNLABELED_SIZE'] = unlabeled_size
    info['ROUND_SIZE'] = round_size
    info['FINISHED_ROUND'] = self.db.get('round', 0)
    info['PERFORMANCE'] = 0

    info.update(self.keywords_template)

    return self.render(self.html_template, **info)


def activelearning_web_server(activelearning_name,
                              main_folder,
                              html_template,
                              keywords_template,
                              task,
                              server_port,
                              parent_id,
                              download_folder,
                              upload_folder,
                              request_queue):
  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  client_socket = None
  tornado.options.parse_command_line()

  if not os.path.exists(os.path.join(main_folder, 'web', 'template')):
    os.makedirs(os.path.join(main_folder, 'web', 'template'))

  activelearning_server_template_dir = os.path.join(main_folder, 'web', 'template')

  if not os.path.exists(os.path.join(main_folder, 'web', 'static')):
    os.makedirs(os.path.join(main_folder, 'web', 'static'))

  activelearning_server_static_dir = os.path.join(main_folder, 'web', 'static')

  static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
  for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
    if static_file[0] == '.':
      continue
    if not os.path.isfile(os.path.join(static_folder, 'resource', 'static', static_file)):
      continue

    shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file),
                activelearning_server_static_dir)

  if html_template is None:
    shutil.copy(os.path.join(static_folder, 'resource', 'templates', 'activelearning.html'),
                activelearning_server_template_dir)
    html_template = 'activelearning.html'
  else:
    shutil.copy(os.path.join(main_folder, html_template), activelearning_server_template_dir)

  # data folder
  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data'))

  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data', 'annotations')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data', 'annotations'))

  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data', 'images')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data', 'images'))

  db = {'process_state': 'UNLABEL-PREPARE', 'round': 0}
  settings = {
    'port': server_port,
    'name': activelearning_name,
    'static_path': activelearning_server_static_dir,
    'template_path': activelearning_server_template_dir,
    'html_template': html_template,
    'keywords_template': keywords_template,
    'task_name': task.task_name,
    'task_type': task.task_type,
    'db': db,
    'data_folder': os.path.join(main_folder, 'web', 'static', 'data', 'annotations'),
    'cookie_secret': str(uuid.uuid4()),
    'request_queue': request_queue,
    'upload': upload_folder,
    'download': download_folder
  }
  app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                          (r"/activelearning/upload/", UploadHandler),
                                          (r"/activelearning/download/", DownloadHandler),
                                          (r"/activelearning/state/", ActiveLearningState)],
                                **settings)

  http_server = tornado.httpserver.HTTPServer(app)
  http_server.listen(options.port)

  tornado.ioloop.IOLoop.instance().start()


# activelearning_web_server('hello', '/Users/jian/Downloads/Aa', None, None, 10031, '')

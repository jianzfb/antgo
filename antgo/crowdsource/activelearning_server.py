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


class BaseHandler(tornado.web.RequestHandler):
  @property
  def activelearning(self):
    return self.settings.get('activelearning_name', '-')

  @property
  def html_template(self):
    return self.settings['html_template']

  @property
  def data_folder(self):
    return self.settings['data_folder']

  @property
  def db(self):
    return self.settings['db']

  @property
  def labels(self):
    return self.settings['labels']

  @property
  def label_num_per_time(self):
    return self.settings['label_num_per_time']

  @property
  def labels(self):
    return self.settings['labels']


class IndexHandler(BaseHandler):
  def get(self):
    session_id = self.get_cookie('sessionid')
    if session_id is None:
      session_id =str(uuid.uuid4())
      self.set_secure_cookie('sessionid', session_id)

    if session_id not in self.db['user']:
      # record current user label data
      self.db['user'][session_id] = {'map': {}, 'list':[]}

    # does have unlabeled data
    is_pause = True
    if self.db[self.activelearning]['status'] != 'PAUSE':
      for a, b, c in self.db[self.activelearning]['list']:
        if b != 'finish':
          is_pause = False
          break

    if is_pause:
      self.db[self.activelearning]['status'] = 'PAUSE'
      self.write('the %d round label has been finished, the next round would start after 1 hour'%self.db[self.activelearning]['round'])
    else:
      return self.render(self.html_template)


class NotifyRestoreHandler(BaseHandler):
  def post(self, round):
    waiting_list = json.loads(self.get_argument('waiting_data'))
    self.db[self.activelearning]['status'] = self.get_argument('status', 'RUNNING')
    self.db[self.activelearning]['round'] = int(round)
    self.db[self.activelearning]['samples'] = self.get_argument('samples', 0)
    self.db[self.activelearning]['list'].extend([[w, '', time.time()] for w in waiting_list])
    self.finish()


class LabelHandler(BaseHandler):
  def get(self):
    session_id = self.get_cookie('sessionid')
    if session_id is None:
      self.set_status(404)
      self.finish()

    waiting_samples = []
    if len(self.db['user'][session_id]['list']) == 0:
      waiting_samples = [si for si, s in enumerate(self.db[self.activelearning]['list']) if s[1] == ''][0:self.label_num_per_time]
      # reassign unlabeled sample
      self.db['user'][session_id]['list'] = waiting_samples
      self.db['user'][session_id]['map'] = {self.db[self.activelearning]['list'][si][0]: si for si in waiting_samples}

      for si in waiting_samples:
        self.db[self.activelearning]['list'][si][1] = 'waiting'
        self.db[self.activelearning]['list'][si][2] = time.time()

    else:
      total_list = self.db[self.activelearning]['list']
      is_over = True
      for si in self.db['user'][session_id]['list']:
        if total_list[si][1] != 'finish':
          is_over = False
          break

      if is_over:
        waiting_samples = [si for si, s in enumerate(self.db[self.activelearning]['list']) if s[1] == ''][0:self.label_num_per_time]
        # reassign unlabeled sample
        self.db['user'][session_id]['list'] = waiting_samples
        self.db['user'][session_id]['map'] = {self.db[self.activelearning]['list'][si][0]: si for si in waiting_samples}

        for si in waiting_samples:
          self.db[self.activelearning]['list'][si][1] = 'waiting'
          self.db[self.activelearning]['list'][si][2] = time.time()
      else:
        waiting_samples = self.db['user'][session_id]['list']

    imageURLs = []
    annotationURLs = []
    for si in waiting_samples:
      imageURLs.append(os.path.join('static/data/images', self.db[self.activelearning]['list'][si][0]))
      annotationURLs.append(os.path.join('static/data/annotations', self.db[self.activelearning]['list'][si][0]))

    data = {"labels": self.labels,
            "imageURLs": imageURLs,
            "annotationURLs": annotationURLs}

    self.write(json.dumps(data))


class SaveHandler(BaseHandler):
  def post(self, *args, **kwargs):
    session_id = self.get_cookie('sessionid')
    if session_id is None:
      self.set_status(404)
      self.finish()
      return

    content = json.loads(self.request.body)
    data = content['data']
    filename = content['filename']

    if filename not in self.db['user'][session_id]['map']:
      self.set_status(404)
      self.finish()
      return

    with open(os.path.join(self.data_folder, filename), 'wb') as fp:
      data = data.replace('data:image/png;base64,', '')
      fp.write(base64.b64decode(data))

    index = self.db['user'][session_id]['map'][filename]
    self.db[self.activelearning]['list'][index][1] = 'finish'
    self.db[self.activelearning]['list'][index][2] = time.time()
    self.write(json.dumps({'status': 'ok'}))


class IsFinishHandler(BaseHandler):
  def get(self):
    is_finish = True
    for a, b, c in self.db[self.activelearning]['list']:
      if b != 'finish':
        is_finish = False
        break

    self.write(json.dumps({'status': 'finish' if is_finish else 'nofinish'}))


def activelearning_web_server(activelearning_name,
                              main_folder,
                              html_template,
                              task,
                              server_port,
                              parent_id):
  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  if task is not None:
    if task.task_type not in ['SEGMENTATION']:
      print('dont support active learning for %s task'%task.task_type)
      return

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

  # css files
  css_folder = os.path.join(static_folder, 'resource', 'css', 'activelearning')
  if os.path.exists(os.path.join(main_folder, 'web', 'static', 'css')):
    shutil.rmtree(os.path.join(main_folder, 'web', 'static', 'css'))
  shutil.copytree(css_folder, os.path.join(main_folder, 'web', 'static', 'css'))

  # js files
  js_folder = os.path.join(static_folder, 'resource', 'js', 'activelearning')
  if os.path.exists(os.path.join(main_folder, 'web', 'static', 'js')):
    shutil.rmtree(os.path.join(main_folder, 'web', 'static', 'js'))
  shutil.copytree(js_folder, os.path.join(main_folder, 'web', 'static', 'js'))

  # data folder
  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data'))

  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data', 'annotations')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data', 'annotations'))

  if not os.path.exists(os.path.join(main_folder, 'web', 'static', 'data', 'images')):
    os.makedirs(os.path.join(main_folder, 'web', 'static', 'data', 'images'))

  db = {activelearning_name: {'status': 'PAUSE', 'round': -1, 'samples': 0, 'list': []}, 'user': {}}
  settings = {
    'port': server_port,
    'activelearning_name': activelearning_name,
    'static_path': activelearning_server_static_dir,
    'template_path': activelearning_server_template_dir,
    'html_template': html_template,
    'db': db,
    'data_folder': os.path.join(main_folder, 'web', 'static', 'data', 'annotations'),
    'cookie_secret': str(uuid.uuid4()),
    'label_num_per_time': 10,
    'labels': task.class_label if task is not None else ['A', 'B']
  }
  app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                          (r"/label/", LabelHandler),
                                          (r"/save/", SaveHandler),
                                          (r"/notify/restore/([^/]+)/", NotifyRestoreHandler),
                                          (r"/isfinish/", IsFinishHandler),],
                                **settings)
  http_server = tornado.httpserver.HTTPServer(app)
  http_server.listen(options.port)

  tornado.ioloop.IOLoop.instance().start()


# activelearning_web_server('hello', '/Users/jian/Downloads/Aa', None, None, 10031, '')

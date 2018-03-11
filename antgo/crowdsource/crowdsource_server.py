# -*- coding: UTF-8 -*-
# Time: 1/6/18
# File: crowdsource_server.py
# Author: jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import os
import pipes
import json
import uuid
from tornado.options import define, options
from antgo.crowdsource.slaver import *
from antgo.utils import logger
import multiprocessing
import requests
from antgo import config
import signal
import copy
import shutil
import socket
import random
import zmq

Config = config.AntConfig


class BaseHandler(tornado.web.RequestHandler):
  def get_current_user(self):
    return self.get_secure_cookie('user')

  @property
  def task_name(self):
    return self.settings['task_name']

  @property
  def html_template(self):
    return self.settings['html_template']

  @property
  def totem(self):
    return self.settings['totem']

  
class IndexHandler(BaseHandler):
  def get(self):
    # client id
    if self.get_secure_cookie('user') is None:
      self.set_secure_cookie('user', str(uuid.uuid4()))
    
    task = {'title': self.task_name,}
    self.render(self.html_template, task=task)


class HeartBeatHandler(tornado.web.RequestHandler):
  def get(self):
    self.write(json.dumps({'ALIVE': True}))


class ClientQuery(BaseHandler):
  def post(self):
    if self.get_current_user() is None:
      self.send_error(500)
    
    client_query = {}
    client_query['QUERY'] = self.get_argument('QUERY')
    client_query['CLIENT_ID'] = self.get_current_user().decode('utf-8')
    client_query['CLIENT_RESPONSE'] = {}
    client_query['CLIENT_RESPONSE']['WORKSITE'] = self.get_argument('CLIENT_RESPONSE_WORKSITE', None)
    client_query['CLIENT_RESPONSE']['CONCLUSION'] = self.get_argument('CLIENT_RESPONSE_CONCLUSION', None)
    client_query['QUERY_INDEX'] = int(self.get_argument('QUERY_INDEX', -1))
    client_query['QUERY_STATUS'] = self.get_argument('QUERY_STATUS', '')

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect('ipc://%s'%self.totem)
    socket.send_json(client_query)
    server_response = socket.recv_json()
    socket.close()

    # return
    self.write(json.dumps(server_response))


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()

signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)


def crowdsrouce_server_start(totem,
                             experiment_id,
                             app_token,
                             dump_dir,
                             task_name,
                             html_template,
                             server_port):
  # 0.step define tornado http server port
  define('port', default=server_port, help="run on the given port", type=int)
  print('http server %d'%int(server_port))

  # reverse tcp tunnel (inner net pass through)
  reverse_tcp_tunnel_process = None
  try:
    # 0.step prepare static resource to dump_dir
    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
      if static_file[0] == '.':
        continue

      shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file), dump_dir)

    # 1.step request open reverse tcp tunnel port
    request_url = 'http://%s:%s/hub/api/crowdsource/evaluation/experiment/%s'%(Config.server_ip, Config.server_port, experiment_id)
    user_authorization = {'Authorization': "token " + app_token}
    if app_token is None:
      logger.error('couldnt build connection with mltalker (token is None)')
      return

    time.sleep(5)
    res = requests.post(request_url, data=None, headers=user_authorization)
    content = json.loads(res.content)
    inner_port = None
    if content['STATUS'] == 'SUCCESS':
      inner_port = content['INNER_PORT']

    if inner_port is None:
      logger.error('fail to apply public port in mltalker')
      return

    # 2.step launch reverse tcp tunnel
    master = '%s:%s'%(Config.server_ip, str(inner_port))
    reverse_tcp_tunnel_process = multiprocessing.Process(target=launch_slaver_proxy,
                                                         args=(master, '127.0.0.1:%d'%server_port))
    reverse_tcp_tunnel_process.start()

    # 3.step launch crowdsource server (http server)
    tornado.options.parse_command_line()
    settings={'template_path': os.path.join(static_folder, 'resource', 'templates'),
              'static_path': dump_dir,
              'totem': totem,
              'task_name': task_name,
              'html_template':html_template,
              'cookie_secret': str(uuid.uuid4())}
    app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                            (r"/crowdsource/query", ClientQuery),
                                            (r"/heartbeat", HeartBeatHandler),],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)

    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
  except GracefulExitException:
    # 1.step notify mltalker stop crowdserver
    request_url = 'http://%s:%s/hub/api/crowdsource/evaluation/experiment/%s' % (Config.server_ip, Config.server_port, experiment_id)
    user_authorization = {'Authorization': "token " + app_token}
    if app_token is None:
      logger.error('couldnt build connection with mltalker (token is None)')
      return

    res = requests.delete(request_url, data=None, headers=user_authorization)
    content = json.loads(res.content)

    if content['STATUS'] == 'SUCCESS':
      logger.info('success to delete crowdsource router in mltalker')
    else:
      logger.error('fail to delete crowdsource router in mltalker')

    # 2.step stop inner net proxy
    if reverse_tcp_tunnel_process is not None:
      os.kill(reverse_tcp_tunnel_process.pid, signal.SIGKILL)
    logger.info('exit crowdsource server')
  except Exception as err:
    # exception
    logger.error(err.message)
  finally:
    sys.exit(0)

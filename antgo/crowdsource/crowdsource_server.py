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
from tornado import gen
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
from zmq.eventloop import future
from antgo.crowdsource.utils import *
Config = config.AntConfig


class BaseHandler(tornado.web.RequestHandler):
  @property
  def task_name(self):
    return self.settings['task_name']

  @property
  def html_template(self):
    return self.settings['html_template']

  @property
  def db(self):
    return self.settings['db']

  @property
  def port(self):
    return self.settings['port']

  @property
  def token(self):
    return self.settings['token']

  @property
  def zmq_client_socket(self):
    return self.settings['zmq_client_socket']


class IndexHandler(BaseHandler):
  def get(self, experiment_id, user_id):
    session_id = self.get_cookie('sessionid')
    if session_id is None:
      session_id =str(uuid.uuid4())
      self.set_secure_cookie('sessionid', session_id)


    if session_id not in self.db['user']:
      self.db['user'][session_id] = user_id

    # render page
    self.render(self.html_template, task={'title': self.task_name,})


class HeartBeatHandler(tornado.web.RequestHandler):
  def get(self):
    self.write(json.dumps({'ALIVE': True}))


class ClientQuery(BaseHandler):
  @gen.coroutine
  def post(self, experiment_id, user_id):
    if self.get_cookie('sessionid') not in self.db['user']:
      self.set_status(500)
      self.finish()
      return

    client_query = {}
    client_query['QUERY'] = self.get_argument('QUERY')
    client_query['CLIENT_ID'] = self.get_cookie('sessionid')
    client_query['CLIENT_RESPONSE'] = {}
    client_query['CLIENT_RESPONSE']['WORKSITE'] = self.get_argument('CLIENT_RESPONSE_WORKSITE', None)
    client_query['CLIENT_RESPONSE']['CONCLUSION'] = self.get_argument('CLIENT_RESPONSE_CONCLUSION', None)
    client_query['QUERY_INDEX'] = int(self.get_argument('QUERY_INDEX', -1))
    client_query['QUERY_STATUS'] = self.get_argument('QUERY_STATUS', '')

    # 1.step get process result
    self.zmq_client_socket.send_json(client_query)
    server_response = yield self.zmq_client_socket.recv_json()

    # 2.step is over flag
    if 'PAGE_STATUS' in server_response and server_response['PAGE_STATUS'] == 'STOP':
      if self.token is not None:
        user_authorization = {'Authorization': "token " + self.token}
        request_url = 'http://%s:%s/hub/api/crowdsource/%s/proxy' % (Config.server_ip, Config.server_port, experiment_id)
        requests.delete(request_url, data={'user_name': user_id}, headers=user_authorization)

    # 3.step return server response
    self.write(json.dumps(server_response))


class PrefixRedirectHandler(BaseHandler):
  def get(self):
    static_pi = self.request.uri.find('static')
    path = self.request.uri[static_pi:]
    self.redirect('http://127.0.0.1:%d/%s'%(self.port, path), permanent=False)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def crowdsrouce_server_start(parent_id,
                             experiment_id,
                             app_token,
                             dump_dir,
                             task_name,
                             html_template,
                             server_port,
                             crowdsource_info={}):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)
  
  # log
  logger.info('crowdsource server prepare serving on %d'%server_port)

  # 0.step define tornado http server port
  define('port', default=server_port, help="run on the given port", type=int)

  # reverse tcp tunnel (inner net pass through)
  reverse_tcp_tunnel_process = None
  try:
    # 0.step prepare static resource to dump_dir
    if not os.path.exists(os.path.join(dump_dir, 'static')):
      os.makedirs(os.path.join(dump_dir, 'static'))

    crowdsource_static_folder = os.path.join(dump_dir, 'static')

    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
      if static_file[0] == '.':
        continue

      shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file), crowdsource_static_folder)

    # 1.step prepare template resource
    crowdsource_tempate_dir = os.path.join(dump_dir, 'templates')
    if not os.path.exists(crowdsource_tempate_dir):
      os.makedirs(crowdsource_tempate_dir)

    shutil.copy(os.path.join(static_folder, 'resource', 'templates', 'crowdsource.html'), crowdsource_tempate_dir)
    if os.path.exists(os.path.join(static_folder, 'resource', 'templates', html_template)):
      shutil.copy(os.path.join(static_folder, 'resource', 'templates', html_template), crowdsource_tempate_dir)

    if os.path.exists(os.path.join(os.curdir, html_template)):
      shutil.copy(os.path.join(os.curdir, html_template), crowdsource_tempate_dir)

    # 1.step request open reverse tcp tunnel port
    if app_token is not None:
      try:
        request_url = 'http://%s:%s/hub/api/crowdsource/evaluation/experiment/%s'%(Config.server_ip, Config.server_port, experiment_id)
        user_authorization = {'Authorization': "token " + app_token}
        if app_token is None:
          logger.error('couldnt build connection with mltalker (token is None)')
          return

        time.sleep(5)
        # collect crowdsource basic information
        res = requests.post(request_url, data=crowdsource_info, headers=user_authorization)
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
      except:
        logger.error('couldnt build reverse proxy')

    # 3.step launch crowdsource server (http server)
    tornado.options.parse_command_line()
    context = future.Context.instance()
    client_socket = context.socket(zmq.REQ)
    client_socket.bind('ipc://%s'%str(parent_id))

    settings={'template_path': crowdsource_tempate_dir,
              'static_path': crowdsource_static_folder,
              'task_name': task_name,
              'html_template':html_template,
              'cookie_secret': str(uuid.uuid4()),
              'port':server_port,
              'db': {'user':{}},
              'token': app_token,
              'zmq_client_socket': client_socket,}
    app = tornado.web.Application(handlers=[(r"/crowdsource/([^/]+)/user/([^/]+)/", IndexHandler),
                                            (r"/crowdsource/([^/]+)/user/([^/]+)/query", ClientQuery),
                                            (r"/heartbeat", HeartBeatHandler),
                                            (r"/.*/static/.*", PrefixRedirectHandler)],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)

    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
  except GracefulExitException:
    # 1.step notify mltalker stop crowdserver
    if app_token is not None:
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

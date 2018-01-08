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
from tornado.options import define, options
define('port', default=8000, help="run on the given port", type=int)


class BaseHandler(tornado.web.RequestHandler):
  def get_current_user(self):
    return self.get_secure_cookie('user')
  
  @property
  def server_pipe(self):
    return self.settings['client_pipe']
  
  
class IndexHandler(BaseHandler):
  def get(self):
    # client id
    if self.get_secure_cookie('user') is None:
      self.set_secure_cookie('user', str(uuid.uuid4()))
    
    task = {'content': 'welcome the MLTalker', 'html': '<p id="AA"></p>', 'title': 'Jian'}
    self.render('crowdsource.html', task=task)


class CustomRender(BaseHandler):
  def post(self):
    self.write(json.dumps(self.settings['custom_response_html']))


class ClientQuery(BaseHandler):
  def post(self):
    print(self.get_current_user())
    
    if self.get_current_user() is None:
      self.send_error(500)
    
    client_query = {}
    client_query['QUERY'] = self.get_argument('QUERY')
    client_query['CLIENT_ID'] = self.get_current_user().decode('utf-8')
    client_query['CLIENT_RESPONSE'] = self.get_argument('CLIENT_RESPONSE')
    self.server_pipe.send(client_query)
    server_response = self.server_pipe.recv()
    # data = {'PAGE_DATA': {'AA': {'DATA': 'hello the world', 'TYPE': 'START'}}}
    #
    self.write(json.dumps(server_response))


def crowdsrouce_server_start(client_pipe, dump_dir, task_html, custom_response_html):
  dump_dir = os.curdir

  tornado.options.parse_command_line()
  static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])

  settings={'template_path': os.path.join(static_folder, 'html', 'templates'),
            'static_path': os.path.join(static_folder, 'html', "static"),
            'client_pipe': client_pipe,
            'custom_response_html': custom_response_html,
            'cookie_secret': str(uuid.uuid4())}
  app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                          (r"/crowdsource/query", ClientQuery),
                                          (r"/crowdsource/render", CustomRender), ],
                                **settings)
  http_server = tornado.httpserver.HTTPServer(app)

  http_server.listen(options.port)
  tornado.ioloop.IOLoop.instance().start()
# -*- coding: UTF-8 -*-
# @Time    : 18-4-27
# @File    : demo_server.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
import tornado.web
import os
import zmq
import shutil


class BaseHandler(tornado.web.RequestHandler):
  @property
  def demo_name(self):
    return self.settings.get('demo_name', '-')
  @property
  def demo_type(self):
    return self.settings.get('demo_type', '-')
  
  @property
  def demo_dump(self):
    return self.settings.get('demo_dump', '')
  
  @property
  def html_template(self):
    return self.settings['html_template']
  
  @property
  def db(self):
    return self.settings['db']
  
  @property
  def port(self):
    return self.settings['port']


class IndexHandler(BaseHandler):
  def get(self):
    self.render(self.html_template, demo={'name': self.demo_name, 'type': self.demo_type})
    # self.write('hello the world')


class ClientQuery(BaseHandler):
  def post(self):
    # send query to client
    pass


class ClientComment(BaseHandler):
  def post(self):
    pass


class PrefixRedirectHandler(BaseHandler):
  def get(self):
    static_pi = self.request.uri.find('static')
    path = self.request.uri[static_pi:]
    self.redirect('http://127.0.0.1:%d/%s'%(self.port, path), permanent=False)


def demo_server_start(demo_name, demo_type, demo_dump_dir, html_template, server_port):
  # 0.step define http server port
  define('port', default=server_port, help='run on port')

  # 1.step prepare static resource
  static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
  demo_static_dir = os.path.join(demo_dump_dir, 'static')
  if not os.path.exists(demo_static_dir):
    os.makedirs(demo_static_dir)

  for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
    if static_file[0] == '.':
      continue
  
    shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file), demo_static_dir)

  # 2.step prepare html template
  demo_tempate_dir = os.path.join(demo_dump_dir, 'templates')

  if not os.path.exists(demo_tempate_dir):
    os.makedirs(demo_tempate_dir)

  if html_template is None:
    html_template = 'demo.html'

  if not os.path.exists(os.path.join(demo_tempate_dir, html_template)):
    assert(os.path.exists(os.path.join(static_folder, 'resource', 'templates',html_template)))
    shutil.copy(os.path.join(static_folder, 'resource', 'templates',html_template), demo_tempate_dir)

  tornado.options.parse_command_line()
  settings = {
    'template_path': demo_tempate_dir,
    'static_path': demo_static_dir,
    'html_template': html_template,
    'port': server_port,
    'demo_dump': demo_dump_dir,
    'demo_name': demo_name,
    'demo_type': demo_type
  }
  app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                          (r"/demo", IndexHandler),
                                          (r"/api/query", ClientQuery),
                                          (r"/api/comment", ClientComment),
                                          (r"/.*/static/.*", PrefixRedirectHandler)],
    **settings)
  http_server = tornado.httpserver.HTTPServer(app)
  http_server.listen(options.port)
  tornado.ioloop.IOLoop.instance().start()
  
demo_server_start('world','IMAGE_SEGMENTATION','/Users/jian/Downloads/ww',None,6990)
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
from tornado.options import define, options
define('port', default=8000, help="run on the given port", type=int)

class IndexHandler(tornado.web.RequestHandler):
  def get(self):
#    greeting = self.get_argument('greeting', 'Hello')
#    self.write(greeting+', friendly user!')
    task = 'zhangjain'
    self.settings['client_pipe'].send({'QUERY': 'START', 'CLIENT_ID': 'aabbcc'})
    self.render('crowdsource.html', task=task)


def crowdsrouce_server_start(client_pipe, dump_dir, task_html):
  dump_dir = os.curdir

  tornado.options.parse_command_line()
  static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])

  settings={'template_path': os.path.join(static_folder,'html','templates'),
            'static_path':os.path.join(static_folder, 'html', "static"),
            'client_pipe': client_pipe}
  app = tornado.web.Application(handlers=[(r"/", IndexHandler)],
                                **settings)
  http_server = tornado.httpserver.HTTPServer(app)

  http_server.listen(options.port)
  tornado.ioloop.IOLoop.instance().start()
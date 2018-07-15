# -*- coding: UTF-8 -*-
# @Time : 2018/6/22
# @File : api_server.py
# @Author: Jian <jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import tornado.httpserver
import tornado.ioloop
import tornado.options
from tornado.options import define, options
from tornado import web
import tornado.web
from antgo.utils import logger
import os
import shutil
import json
import numpy as np
from antgo.utils.fs import *
from antgo.utils.encode import *
from PIL import Image
import imageio
import uuid
import signal


class BaseHandler(tornado.web.RequestHandler):
  @property
  def api_name(self):
    return self.settings.get('api_name', '')

  @property
  def api_task(self):
    return self.settings.get('api_task', '')
  @property
  def api_static_path(self):
    return self.settings.get('api_static_path', '')

  @property
  def api_input_queue(self):
    return self.settings.get('api_input_queue', None)

  @property
  def api_output_queue(self):
    return self.settings.get('api_output_queue', None)

  @property
  def input_data_format(self):
    return self.settings.get('input_data_format', {})

  @property
  def output_data_format(self):
    return self.settings.get('output_data_format',{})


class APIHandler(BaseHandler):
  def dispatch_prepare_data(self, data, data_type):
    data_path = None
    data_name = None
    if data_type == 'URL':
      # download data
      download_path = os.path.join(self.api_static_path, 'input')
      if not os.path.exists(download_path):
        os.makedirs(download_path)

      data_name = os.path.normpath(data).split('/')[-1]
      data_name = '%s_%s' % (str(uuid.uuid4()), data_name)
      data_path = download(data, os.path.join(self.api_static_path, 'input'), data_name)
      data_path = os.path.normpath(data_path)
    elif data_type == 'PATH':
      data_name = data.split('/')[-1]
      if os.path.exists(os.path.join(self.api_static_path, 'input', data_name)):
        data_path = os.path.join(self.api_static_path, 'input', data_name)

    if data_type == 'URL' or data_type == 'PATH':
      ext_name = data_path.split('/')[-1].split('.')[-1].lower()
      if ext_name in ['jpg', 'jpeg', 'png', 'bmp']:
        image_data = Image.open(data_path)
        img_data = np.fromstring(image_data.tobytes(), dtype=np.uint8)
        img_data = img_data.reshape((image_data.size[1], image_data.size[0], len(image_data.getbands())))
        fsize = os.path.getsize(data_path)
        fsize = fsize / float(1024 * 1024)
        return img_data, data_name, 'IMAGE', round(fsize, 2), {}
      elif ext_name in ['mp4']:
        # TODO: bug
        reader = imageio.get_reader(data_path)
        image_list = []
        for im in reader:
          img_data = np.fromstring(im.tobytes(), dtype=np.uint8)
          img_data = img_data.reshape((im.shape[0], im.shape[1], im.shape[2]))
          image_list.append(np.expand_dims(img_data, 0))

        image_volume = np.vstack(image_list)

        fsize = os.path.getsize(data_path)
        fsize = fsize / float(1024 * 1024)
        return image_volume, data_name, 'VIDEO', round(fsize,2), {'FPS': reader.get_meta_data()['fps']}
      else:
        # TODO: support video and sound
        logger.warn('dont support file type %s' % ext_name)

    return None, None, None, 0


  def post_process_model_response(self, model_result, data_params):
    output_info = {}
    for k,v in self.output_data_format.items():
      data = model_result[k]
      data_type = v

      if data_type == 'IMAGE':
        data = data.astype(np.uint8)
        file_name = str(uuid.uuid4())
        imageio.imsave(os.path.join(self.api_static_path,
                                    'output',
                                    '%s.png'%file_name),
                       data)
        output_info[k] = '/static/output/%s.png'%file_name
      elif data_type == 'VIDEO':
        file_name = str(uuid.uuid4())
        video_path = os.path.join(self.api_static_path,
                                  'output',
                                  '%s.mp4' % file_name)
        writer = imageio.get_writer(video_path, fps=data_params['FPS'])
        if type(data) != list:
          data = np.split(data, data.shape[-1],axis=-1)

        for im in data:
          writer.append_data(np.squeeze(im, -1))
        writer.close()

        output_info[k] = '/static/output/%s.mp4'%file_name
      else:
        pass

    return output_info


  def post(self):
    # 0.step input data
    data = self.get_argument('DATA', None)
    data_type = self.get_argument('DATA_TYPE', None)
    if data is None or data_type is None:
      raise web.HTTPError(500)

    # 1.step check input parameters
    for k,v in self.input_data_format.items():
      if k in ['DATA_TYPE', 'DATA_SIZE']:
        continue

      val = self.get_argument(k, None)
      if val is None:
        self.write(json.dumps({'result': 'fail', 'reason': 'parameters are not incomplete (missing %s)'%k}))
        self.set_status(500)
        return

    # 2.step parse data and check
    # DATA, DATA_NAME, DATA_TYPE, DATA_SIZE, DATA_PARAMS
    model_data, model_data_name, model_data_type, model_data_size, model_data_params = \
      self.dispatch_prepare_data(data, data_type)
    if model_data is None:
      self.write(json.dumps({'result': 'fail', 'reason': 'data parse fail'}))
      self.set_status(500)
      return

    if model_data_type != self.input_data_format['DATA_TYPE']:
      self.write(json.dumps({'result': 'fail', 'reason': 'data type not supported'}))
      self.set_status(500)
      return

    if 'DATA_SIZE' in self.input_data_format:
      if model_data_size > self.input_data_format['DATA_SIZE']:
        self.write(json.dumps({'result': 'fail',
                               'reason': 'data size is too large (<%dMB)'%self.input_data_format['DATA_SIZE']}))
        self.set_status(500)
        return

    # 3.step push to input data pipeline
    self.api_input_queue(model_data)

    # 4.step waiting response
    _, api_result = self.api_output_queue.get()

    # 5.step post process and render
    # api_result = {'key': 'value', 'key': 'value'}
    api_result = self.post_process_model_response(api_result, model_data_params)
    self.write(json.dumps(api_result))


class UploadHandler(BaseHandler):
  def post(self):
    pass


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def api_server_start(api_name,
                     api_task,
                     api_dump_dir,
                     server_port,
                     api_input_queue,
                     api_output_queue,
                     input_data_format,
                     output_data_format,
                     parent_id):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)

  try:
    # 0.step define http server port
    define('port', default=server_port, help='run on port')

    # 1.step prepare static resource
    api_static_dir = os.path.join(api_dump_dir, 'static')
    if not os.path.exists(api_static_dir):
      os.makedirs(api_static_dir)

    if not os.path.exists(os.path.join(api_static_dir, 'input')):
      os.makedirs(os.path.join(api_static_dir, 'input'))
    if not os.path.exists(os.path.join(api_static_dir, 'output')):
      os.makedirs(os.path.join(api_static_dir, 'output'))

    tornado.options.parse_command_line()
    settings = {
      'api_static_path': api_static_dir,
      'api_port': server_port,
      'api_name': api_name,
      'api_task': api_task,
      'api_input_queue': api_input_queue,
      'api_output_queue': api_output_queue,
      'input_data_format': input_data_format,
      'output_data_format': output_data_format,
    }
    app = tornado.web.Application(handlers=[(r"/api/", APIHandler)],
                                  **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)

    logger.info('api server is providing server on port %d' % server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('api stop server')
  except GracefulExitException:
    logger.info('demo server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    os.kill(parent_id, signal.SIGKILL)
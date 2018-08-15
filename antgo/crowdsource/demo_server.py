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
from tornado import web, gen
import tornado.web
from antgo.utils import logger
import os
import zmq
import shutil
import json
import numpy as np
from antgo.utils.fs import *
from antgo.utils.encode import *
from PIL import Image
import imageio
import uuid
import signal
from zmq.eventloop import future
from ..utils.serialize import loads,dumps

class BaseHandler(tornado.web.RequestHandler):
  @property
  def demo_name(self):
    return self.settings.get('demo_name', '-')

  @property
  def demo_type(self):
    return self.settings.get('demo_type', '-')

  @property
  def demo_description(self):
    return self.settings.get('demo_description',{})

  @property
  def demo_dump(self):
    return self.settings.get('demo_dump', '')

  @property
  def demo_support_user_upload(self):
    return self.settings.get('support_user_upload', True)

  @property
  def demo_support_user_input(self):
    return self.settings.get('support_user_input', True)

  @property
  def demo_support_user_interaction(self):
    return self.settings.get('support_user_interaction', False)
  
  @property
  def demo_support_upload_formats(self):
    return self.settings.get('support_upload_formats', '')

  @property
  def demo_support_user_comment(self):
    return self.settings.get('support_user_comment', False)

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
  def zmq_client_socket(self):
    return self.settings['zmq_client_socket']

  def _transfer_data(self, data, uuid_flag):
    # 5.step postprocess demo predict result
    demo_response = {'DATA': {}}
    for k, v in data.items():
      if type(v) == np.ndarray and len(v.shape) > 1:
        # transform to image and save
        if not os.path.exists(os.path.join(self.demo_dump, 'static', 'output')):
          os.makedirs(os.path.join(self.demo_dump, 'static', 'output'))
        
        is_image = True
        image = None
        if len(v.shape) == 2:
          image = ((v - np.min(v)) / (np.max(v) - np.min(v)) * 255).astype(np.uint8)
        elif len(v.shape) == 3:
          assert(v.shape[2] == 1 or v.shape[2] == 3)
          image = v.astype(np.uint8)
        else:
          is_image = False
          assert(len(v.shape) == 4)
          
          # split images along axis=0
          image = np.split(v, v.shape[0], 0)
          for index, image_data in enumerate(image):
            image[index] = np.squeeze(image_data, 0)

          if v.dtype != np.uint8:
            for index in range(len(image)):
              min_v = np.min(image[index])
              max_v = np.max(image[index])
              image[index] = ((image[index] - min_v)/(max_v - min_v) * 255).astype(np.uint8)
          
        if is_image:
          if k == 'RESULT':
            with open(os.path.join(self.demo_dump, 'static', 'output', '%s.png' % uuid_flag), 'wb') as fp:
              fp.write(png_encode(image))
          
            demo_response['DATA'].update({'RESULT': {'DATA': '/static/output/%s.png' % uuid_flag, 'TYPE': 'IMAGE'}})
          else:
            with open(os.path.join(self.demo_dump, 'static', 'output', '%s_%s.png' % (uuid_flag, str(k))), 'wb') as fp:
              fp.write(png_encode(image))
    
            demo_response['DATA'].update({str(k): {'DATA': '/static/output/%s_%s.png' % (uuid_flag, str(k)), 'TYPE': 'IMAGE'}})
        else:
          if k == 'RESULT':
            video_path = os.path.join(self.demo_dump, 'static', 'output', '%s.mp4' % uuid_flag)
            writer = imageio.get_writer(video_path, fps=30)
            for im in image:
              writer.append_data(im)
            writer.close()
            
            demo_response['DATA'].update({'RESULT': {'DATA': '/static/output/%s.mp4'%uuid_flag, 'TYPE': 'VIDEO'}})
          else:
            video_path = os.path.join(self.demo_dump, 'static', 'output', '%s_%s.mp4' % (uuid_flag, str(k)))
            writer = imageio.get_writer(video_path, fps=30)
            for im in image:
              writer.append_data(im)
            writer.close()
            
            demo_response['DATA'].update({str(k): {'DATA': '/static/output/%s_%s.mp4'%(uuid_flag, str(k)), 'TYPE': 'VIDEO'}})
      else:
        # default string
        if type(v) == np.ndarray and len(v.shape) == 1:
          v = v.tolist()
        demo_response['DATA'].update({k: {'DATA':str(v), 'TYPE': 'STRING'}})
      
    demo_response['DEMO_TYPE'] = self.demo_type
    demo_response['DEMO_NAME'] = self.demo_name
  
    return demo_response
  
  def dispatch_prepare_data(self, data, data_type):
    data_path = None
    data_name = None
    if data_type == 'URL':
      # download data
      download_path = os.path.join(self.demo_dump, 'static', 'input')
      if not os.path.exists(download_path):
        os.makedirs(download_path)

      data_name = os.path.normpath(data).split('/')[-1]
      data_name = '%s_%s'%(str(uuid.uuid4()), data_name)
      data_path = download(data, os.path.join(self.demo_dump, 'static', 'input'), data_name)
      data_path = os.path.normpath(data_path)
    elif data_type == 'PATH':
      data_name = data.split('/')[-1]
      if os.path.exists(os.path.join(self.demo_dump, 'static', 'input',data_name)):
        data_path = os.path.join(self.demo_dump, 'static', 'input',data_name)

    if data_type == 'URL' or data_type == 'PATH':
      ext_name = data_path.split('/')[-1].split('.')[-1].lower()
      if ext_name in ['jpg', 'jpeg', 'png', 'bmp']:
        image_data = Image.open(data_path)
        img_data = np.fromstring(image_data.tobytes(), dtype=np.uint8)
        img_data = img_data.reshape((image_data.size[1], image_data.size[0], len(image_data.getbands())))
        return img_data, data_name, 'IMAGE'
      elif ext_name in ['mp4']:
        reader = imageio.get_reader(data_path)
        image_list = []
        for im in reader:
          img_data = np.fromstring(im.tobytes(), dtype=np.uint8)
          img_data = img_data.reshape((im.shape[0], im.shape[1], im.shape[2]))
          image_list.append(np.expand_dims(img_data,0))
          
        image_volume = np.vstack(image_list)
        return image_volume, data_name, 'VIDEO'
      else:
        #TODO: support video and sound
        logger.warn('dont support file type %s'%ext_name)

    return None

  def post_process_model_response(self, uuid_flag, demo_result):
    demo_predict = None
    demo_predict_label = {}
    if type(demo_result) == list or type(demo_result) == tuple:
      demo_predict, demo_predict_label = demo_result[0:2]
      if type(demo_predict_label) != dict:
        logger.warn('demo predict label only dict type')
        demo_predict_label = {}
    else:
      demo_predict = demo_result
    
    demo_predict_label['RESULT'] = demo_predict
    demo_response = self._transfer_data(demo_predict_label, uuid_flag)
    return demo_response
    
    
class IndexHandler(BaseHandler):
  def get(self):
    image_history_data = []
    history_folder = os.path.join(self.demo_dump, 'static', 'input')
    if os.path.exists(history_folder):
      for f in os.listdir(history_folder):
        ext_name = f.split('.')[-1]
        if ext_name.lower() in ['jpg', 'jpeg', 'png', 'gif']:
          image_history_data.append('/static/input/%s'%f)
    
    input_filter = ''
    if self.demo_support_upload_formats is not None:
      for support_format in self.demo_support_upload_formats.split(','):
        if support_format.lower() in ['jpg', 'jpeg', 'png', 'gif']:
          input_filter += 'image/%s,'%support_format.lower()
        elif support_format.lower() in ['mp4']:
          input_filter += 'video/%s,'%support_format.lower()
        elif support_format.lower() in ['mp3', 'wav']:
          input_filter += 'audio/%s,'%support_format.lower()
        else:
          input_filter += '%s,'%support_format.lower()

    if len(input_filter) > 0:
      input_filter = input_filter[0:-1]
    else:
      input_filter = '*'

    self.render(self.html_template, demo={'name': self.demo_name,
                                          'type': self.demo_type,
                                          'description': self.demo_description,
                                          'upload': self.demo_support_user_upload,
                                          'upload_accept': input_filter,
                                          'input': self.demo_support_user_input,
                                          'image_history': image_history_data})


class ClientQueryHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 0.step check support status
    if not self.demo_support_user_input:
      raise web.HTTPError(500)

    # 1.step parse query data
    data = self.get_argument('DATA', None)
    data_type = self.get_argument('DATA_TYPE', None)
    if data is None or data_type is None:
      raise web.HTTPError(500)

    # DATA_TYPE, DATA, TASK_NAME, TASK_TYPE
    model_data, model_data_name, model_data_type = self.dispatch_prepare_data(data, data_type)
    if model_data is None:
      raise web.HTTPError(500)

    # no block
    self.zmq_client_socket.send(dumps(model_data))

    # asyn
    result = yield self.zmq_client_socket.recv()
    result = loads(result)
    _, demo_result = result

    # 4.step post process and render
    demo_response = self.post_process_model_response(model_data_name, demo_result)
    demo_response['INPUT_TYPE'] = model_data_type
    if model_data_type in ['FILE', 'IMAGE', 'AUDIO', 'VIDEO']:
      demo_response['INPUT'] = '/static/input/%s' % model_data_name
    else:
      demo_response['INPUT'] = model_data

    self.write(json.dumps(demo_response))
    self.finish()


class ClientFileUploadAndProcessHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 0.step check support status
    if not self.demo_support_user_upload:
      raise web.HTTPError(500)

    # 1.step receive client file
    upload_path = os.path.join(self.demo_dump, 'static', 'input')
    if not os.path.exists(upload_path):
      os.makedirs(upload_path)

    file_metas = self.request.files.get('file', None)

    if not file_metas:
      raise web.HTTPError(500)

    file_paths = []
    file_names = []
    for meta in file_metas:
      _file_name = '%s_%s'%(str(uuid.uuid4()), meta['filename'])
      _file_path = os.path.join(upload_path, _file_name)

      with open(_file_path, 'wb') as fp:
        fp.write(meta['body'])

      file_paths.append(_file_path)
      file_names.append(_file_name)

    if len(file_paths) == 0 or len(file_names) == 0:
      raise web.HTTPError(500)

    # 2.step parse query data
    model_datas = []
    model_data_names = []
    model_data_types = []

    for file_path in file_paths:
      model_data, model_data_name, model_data_type = self.dispatch_prepare_data(file_path, 'PATH')
      if model_data is None:
        raise web.HTTPError(500)

      model_datas.append(model_data)
      model_data_names.append(model_data_name)
      model_data_types.append(model_data_type)

    # 3.step preprocess data, then submit to model
    model_input = model_datas if len(model_datas) > 1 else model_datas[0]
    # self.demo_data_queue.put(model_input)
    #
    # # 4.step waiting model response
    # _, demo_result = self.demo_result_queue.get()

    # no block
    self.zmq_client_socket.send(dumps(model_input))

    # asyn
    result = yield self.zmq_client_socket.recv()
    result = loads(result)
    _, demo_result = result

    # 5.step post process and render
    demo_response = self.post_process_model_response(model_data_names[0], demo_result)

    if len(model_data_names) == 1:
      demo_response['INPUT_TYPE'] = model_data_types[0]
      if model_data_types[0] in ['FILE', 'IMAGE', 'AUDIO', 'VIDEO']:
        demo_response['INPUT'] = '/static/input/%s' % model_data_names[0]
      else:
        demo_response['INPUT'] = model_datas[0]
    else:
      demo_response['INPUT_TYPE'] = []
      demo_response['INPUT'] = []
      for index in range(len(model_data_names)):
        demo_response['INPUT_TYPE'].append(model_data_types[index])
        if model_data_types[index] in ['FILE', 'IMAGE', 'AUDIO', 'VIDEO']:
          demo_response['INPUT'].append('/static/input/%s' % model_data_names[index])
        else:
          demo_response['INPUT'].append(model_datas[index])

    self.write(json.dumps(demo_response))
    self.finish()


class ClientCommentHandler(BaseHandler):
  def post(self):
    self.set_status(201)
    self.finish()


class PrefixRedirectHandler(BaseHandler):
  def get(self):
    static_pi = self.request.uri.find('static')
    path = self.request.uri[static_pi:]
    self.redirect('http://127.0.0.1:%d/%s'%(self.port, path), permanent=False)


class GracefulExitException(Exception):
  @staticmethod
  def sigterm_handler(signum, frame):
    raise GracefulExitException()


def demo_server_start(demo_name,
                      demo_type,
                      demo_description,
                      support_user_upload,
                      support_user_input,
                      support_user_interaction,
                      support_upload_formats,
                      demo_dump_dir,
                      html_template,
                      server_port,
                      parent_id):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)
  
  try:
    # 0.step define http server port
    define('port', default=server_port, help='run on port')

    zmq_ctx = future.Context.instance()
    client_socket = zmq_ctx.socket(zmq.REQ)
    client_socket.bind('ipc://%s'%str(parent_id))

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
    else:
      shutil.copy(html_template, demo_tempate_dir)

    tornado.options.parse_command_line()
    settings = {
      'template_path': demo_tempate_dir,
      'static_path': demo_static_dir,
      'html_template': html_template,
      'port': server_port,
      'demo_dump': demo_dump_dir,
      'demo_name': demo_name,
      'demo_type': demo_type,
      'demo_description': demo_description,
      'support_user_upload': support_user_upload,
      'support_user_input': support_user_input,
      'support_user_interaction': support_user_interaction,
      'support_upload_formats': support_upload_formats,
      'zmq_client_socket': client_socket,
    }
    app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                            (r"/demo", IndexHandler),
                                            (r"/api/query/", ClientQueryHandler),
                                            (r"/api/comment/", ClientCommentHandler),
                                            (r"/submit/", ClientFileUploadAndProcessHandler),
                                            (r"/.*/static/.*", PrefixRedirectHandler)],
      **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
  
    logger.info('demo is providing server on port %d'%server_port)
    tornado.ioloop.IOLoop.instance().start()
    logger.info('demo stop server')
  except GracefulExitException:
    logger.info('demo server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    os.kill(parent_id, signal.SIGKILL)
    
# demo_server_start('world','IMAGE_SEGMENTATION','/Users/jian/Downloads/ww',None,6990)
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
import shutil
import json
import numpy as np
from antgo.utils.fs import *
from antgo.utils.encode import *
from PIL import Image
import uuid
import signal
from ..utils.serialize import loads,dumps
from antgo.crowdsource.utils import *
from tornado.concurrent import run_on_executor
import requests
from concurrent.futures import ThreadPoolExecutor
import functools
import base64
try:
    import queue
except:
    import Queue as queue


class BaseHandler(tornado.web.RequestHandler):
  waiting_queue = {}
  executor = ThreadPoolExecutor(10)

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
  def demo_constraint(self):
    constraint = self.settings.get('support_user_constraint', None)
    if constraint is None:
      constraint = {}

    return constraint

  @property
  def dataset_queue(self):
    return self.settings['dataset_queue']

  @property
  def request_waiting_time(self):
    return self.settings['request_waiting_time']

  def _transfer(self, data_key, data_value, data_type, demo_response):
    if data_type == 'IMAGE':
      if os.path.exists(data_value):
        data = os.path.normpath(data_value)
        shutil.copy(data, os.path.join(self.demo_dump, 'static', 'output'))
        demo_response['DATA'].update({data_key: {'DATA': '/static/output/%s' % data.split('/')[-1], 'TYPE': 'IMAGE'}})
    elif data_type == 'VIDEO':
      if os.path.exists(data_value):
        data = os.path.normpath(data_value)
        shutil.copy(data, os.path.join(self.demo_dump, 'static', 'output'))
        demo_response['DATA'].update({data_key: {'DATA': '/static/output/%s' % data.split('/')[-1], 'TYPE': 'VIDEO'}})
    elif data_type == 'AUDIO':
      if os.path.exists(data_value):
        data = os.path.normpath(data_value)
        shutil.copy(data, os.path.join(self.demo_dump, 'static', 'output'))
        demo_response['DATA'].update({data_key: {'DATA': '/static/output/%s' % data.split('/')[-1], 'TYPE': 'AUDIO'}})
    elif data_type == 'FILE':
      if os.path.exists(data_value):
        data = os.path.normpath(data_value)
        shutil.copy(data, os.path.join(self.demo_dump, 'static', 'output'))
        demo_response['DATA'].update({data_key: {'DATA': '/static/output/%s' % data.split('/')[-1], 'TYPE': 'FILE'}})
    else:
      # string
      demo_response['DATA'].update({data_key: {'DATA': str(data_value), 'TYPE': 'STRING'}})

    return demo_response

  @run_on_executor
  def waitingResponse(self, data_id):
    if data_id in BaseHandler.waiting_queue:
      # 阻塞等待响应
      try:
        response_data = BaseHandler.waiting_queue[data_id].get(timeout=self.request_waiting_time)
      except:
        # 空队列，直接返回空
        BaseHandler.waiting_queue.pop(data_id)
        return {}

      # 清空等待队列s
      BaseHandler.waiting_queue.pop(data_id)

      if response_data is None:
        return {}

      processed_data = response_data['data']
      if len(processed_data) == 0:
        return {}

      # build output folder (static/output)
      if not os.path.exists(os.path.join(self.demo_dump, 'static', 'output')):
        os.makedirs(os.path.join(self.demo_dump, 'static', 'output'))

      demo_response = {'DATA': {}}
      demo_response = {'DATA': {'RESULT': {'DATA':'', 'TYPE': 'STRING'}}}
      for data in processed_data[0]:
        item_type = data['type']
        item_data = data['data']
        item_title = data['title']
        
        demo_response = self._transfer(item_title, item_data, item_type, demo_response)
      return demo_response

    return {}

  @run_on_executor
  def asynProcess(self, preprocess_type, data):
    if preprocess_type == 'DOWNLOAD':
      try:
        # 1.step 下载(如果有必要)
        pic = requests.get(data['url'], timeout=7)
        download_path = os.path.join(self.demo_dump, 'static', 'input', data['file_name'])
        with open(download_path, 'wb') as fp:
          fp.write(pic.content)

        # 2.step 检查
        # 2.1.step 检查文件大小
        if 'file_size' in self.demo_constraint:
          max_file_size = self.demo_constraint['file_size']
          fsize = os.path.getsize(download_path) / float(1024 * 1024)
          if round(fsize,2) > max_file_size:
            return {'status': 400,
                      'code': 'InvalidImageSize', 
                      'message': 'The input file size is too large (>%f MB)'%float(max_file_size)}

        # 2.2.step 检查文件格式
        # 图片格式检测（图片文件后缀可能不对）
        download_path = os.path.normpath(download_path)
        file_type = imghdr.what(download_path)
        if file_type in ['jpeg', 'png', 'gif', 'bmp']:
          # 图像文件
          file_name = '%s.%s'%(data['file_name'], file_type)
          os.rename(download_path, os.path.join('/'.join(download_path.split('/')[0:-1]), file_name))
          download_path = os.path.join('/'.join(download_path.split('/')[0:-1]), file_name)
        else:
          # 非图像文件
          file_type = download_path.split('.')[-1]

        if 'file_type' in self.demo_constraint:
          if file_type not in self.demo_constraint['file_type']:
            return {'status': 400,
                      'code': 'InvalidImageFormat', 
                      'message': 'The input file is not in a valid image format that the service can support'}

        return {
          'status': 200,
          'path': download_path
        }
      except:
        print('Fail to download %s'%data['url'])
        return {
          'status': 500,
          'code': 'UnkownError'
        }   
    elif preprocess_type == 'RECEIVE':
      try:
        # 1.step 保存文件
        file_path = data['file_path']
        file_data = data['file_data']
        with open(file_path, 'wb') as fp:
          fp.write(file_data)
        
        # 2.step 检查
        # 2.1.step 检查文件大小
        if 'file_size' in self.demo_constraint:
          max_file_size = self.demo_constraint['file_size']
          fsize = os.path.getsize(file_path) / float(1024 * 1024)
          if round(fsize,2) > max_file_size:
            return {'status': 400,
                      'code': 'InvalidImageSize', 
                      'message': 'The input file size is too large (>%f MB)'%float(max_file_size)}

        # 2.2.step 检查文件格式
        # 图片格式检测（图片文件后缀可能不对）
        file_path = os.path.normpath(file_path)
        file_type = imghdr.what(file_path)
        if file_type in ['jpeg', 'png', 'gif', 'bmp']:
          # 图像文件
          file_name = '%s.%s'%(file_path.split('/')[-1].split('.')[0], file_type)
          os.rename(file_path, os.path.join('/'.join(file_path.split('/')[0:-1]), file_name))
          file_path = os.path.join('/'.join(file_path.split('/')[0:-1]), file_name)
        else:
          # 非图像文件，通过获取扩展名作为文件类型
          file_type = file_path.split('.')[-1]

        if 'file_type' in self.demo_constraint:
          if file_type not in self.demo_constraint['file_type']:
            return {'status': 400,
                      'code': 'InvalidImageFormat', 
                      'message': 'The input file is not in a valid image format that the service can support'}
        return {
          'status': 200,
          'path': file_path
        }
      except:
        return {
          'status': 500,
          'code': 'UnkownError'
        }   
    elif preprocess_type == 'PACKAGE':
        data_path = os.path.normpath(data['path'])
        data_name = data_path.split('/')[-1]
        ext_name = data_path.split('/')[-1].split('.')[-1].lower()

        # 根据后缀判断上传的数据类型
        if ext_name in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
          return {'data': (data_path, data_name, 'IMAGE'), 'status': 200}
        elif ext_name in ['mp4', 'avi', 'mov']:
          return {'data': (data_path, data_name, 'VIDEO'), 'status': 200}
        elif ext_name in ['txt']:
          return {'data': (data_path, data_name, 'FILE'), 'status': 200}
        else:
          # TODO: support video and sound
          logger.warn('dont support file type %s' % ext_name)
          return {'status': 400, 'code': 'InvalidPackage', 'message': 'Fail package'}
    elif preprocess_type == 'API_QUERY':
      # 1.step base64解码
      # format {'image': '', 'video': None, 'params': [{'data': ,'type': , 'name': ,},{}]}
      image_str = None
      if 'image' in data:
        image_str = data['image']

      if image_str is None:
        return {'status': 400,
          'code': 'InvalidData', 
          'message': 'Missing query data'}

      image_b = base64.b64decode(image_str)
      return {
        'status': 200,
        'data': {'image': image_b, 'params': data['params']}
      }

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

    if 'file_type' in self.demo_constraint:
      for support_format in self.demo_constraint['file_type']:
        if support_format.lower() in ['jpg', 'jpeg', 'png', 'gif']:
          input_filter += 'image/%s,'%support_format.lower()
        elif support_format.lower() in ['mp4','mov','avi']:
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
    if not self.demo_support_user_upload:
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSupport', 'message': 'demo dont support upload'}))
      self.finish()
      return

    upload_path = os.path.join(self.demo_dump, 'static', 'input')
    if not os.path.exists(upload_path):
      os.makedirs(upload_path)

    # 1.step parse query data
    model_datas = []
    model_data_names = []
    model_data_types = []
    request_param = None

    data = self.get_argument('DATA', None)
    data_type = self.get_argument('DATA_TYPE', None)
    if data is not None and data_type is not None:
      if data_type == 'URL':
        # 下载
        file_name = str(uuid.uuid4())
        result = yield self.asynProcess('DOWNLOAD', {'url': data, 'file_name': file_name})
        if result['status'] != 200:
          self.set_status(result['status'])
          self.write(json.dumps(result))
          return

        # 获得下载后的路径
        file_path = result['path']

        # 打包
        result = yield self.asynProcess('PACKAGE', {'path': file_path})
        if result['status'] != 200:
          self.set_status(result['status'])
          self.write(json.dumps(result))
          return
        
        model_data, model_data_name, model_data_type = result['data']

        # 保存
        model_datas.append(model_data)
        model_data_names.append(model_data_name)
        model_data_types.append(model_data_type)
    else:
      if len(self.request.files) == 0:
        self.set_status(400)
        self.write(json.dumps({'code': 'InvalidUploadFile', 
                                'message': 'The input file is not uploaded correctly'}))
        self.finish()
        return

      file_paths = []
      file_names = []
      for _, meta in self.request.files.items():
        _file_name = '%s_%s' % (str(uuid.uuid4()), meta[0]['filename'])
        _file_path = os.path.join(upload_path, _file_name)
        
        # 接受文件
        result = yield self.asynProcess('RECEIVE', {'file_path': _file_path, 'file_data': meta[0]['body']})
        if result['status'] != 200:
          self.set_status(result['status'])
          result['code'] = 'InvalidUpload'
          result['message'] = 'Couldnt upload request file'
          self.write(json.dumps(result))
          return

        _file_path = result['path']
        _file_name = _file_path.split('/')

        file_paths.append(_file_path)
        file_names.append(_file_name)

        # 仅支持一个文件处理
        break

      # 打包
      result = yield self.asynProcess('PACKAGE', {'path': file_paths[0]})
      if result['status'] != 200:
        self.set_status(result['status'])
        self.write(json.dumps(result))
        return
      
      model_data, model_data_name, model_data_type = result['data']
      model_datas.append(model_data)
      model_data_names.append(model_data_name)
      model_data_types.append(model_data_type)

      request_param = self.get_argument('params', None)
      if request_param is not None:
        request_param = json.loads(request_param)

    if request_param is None:
      request_param = {}
      
    request_param.update({
      'id': model_data_names[0]
    })
    
    # push to backend
    model_input = (model_datas[0], model_data_types[0], request_param)

    # 设置等待队列
    BaseHandler.waiting_queue[model_data_names[0]] = queue.Queue()

    # 发送到处理队列
    self.dataset_queue.put(model_input)
    
    # 异步等待响应(设置等待队列，并在异步线程中等待)
    data_response = yield self.waitingResponse(model_data_names[0])

    # 绑定输入数据,返回
    data_response['INPUT_TYPE'] = model_data_types[0]
    if model_data_types[0] in ['FILE', 'IMAGE', 'AUDIO', 'VIDEO']:
      data_response['INPUT'] = '/static/input/%s' % model_data_names[0]
    else:
      data_response['INPUT'] = model_datas[0]

    # 返回成功
    self.write(json.dumps(data_response))
    self.finish()


class ClientCliQueryHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    request_data = self.get_argument('data', None)
    if request_data is None:
      self.set_status(400)
      self.write(json.dumps({'code': 'InvalidQuery', 'message': 'Invalid query data'}))
      return

    # 解析请求字符串
    request_data = json.loads(request_data)

    '''
    format: {'data': {'image': '', 'params': [{'data': ,'type': , 'name': ,},{}]}, 'time': ,}
    '''
    # query_time = request_data['time']
    query_data = request_data['data']

    result = yield self.asynProcess('API_QUERY', query_data)
    if result['status'] != 200:
        self.set_status(result['status'])
        self.write(json.dumps(result))
        return

    data_id = str(uuid.uuid4())
    request_param = {'id': data_id}
    image = result['data']['image']
    request_param.update(result['data']['params'])

    # push to backend
    model_input = (image, 'IMAGE_MEMORY', request_param)

    # 设置等待队列
    BaseHandler.waiting_queue[data_id] = queue.Queue()

    # 发送到处理队列
    self.dataset_queue.put(model_input)
    
    # 异步等待响应(设置等待队列，并在异步线程中等待)
    data_response = yield self.waitingResponse(data_id)

    # 返回成功
    self.write(json.dumps(data_response))
    self.finish()


class ClientResponseHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 返回数据
    # data type: 'FILE', 'STRING', 'IMAGE', 'VIDEO', 'AUDIO'
    # data:       PATH,  '',        PATH,    PATH,    PATH
    response_data = self.get_argument('response', None)
    if response_data is None:
      return
    
    response_data = json.loads(response_data)
    id = response_data['id']
    if id not in BaseHandler.waiting_queue:
      return
    
    # 
    BaseHandler.waiting_queue[id].put(response_data)


class ClientFileUploadAndProcessHandler(BaseHandler):
  @gen.coroutine
  def post(self):
    # 0.step check support status
    if not self.demo_support_user_upload:
      self.set_status(500)
      self.write(json.dumps({'code': 'InvalidSupport', 
                            'message': 'demo dont support upload'}))
      self.finish()
      return

    # 1.step receive client upload file
    upload_path = os.path.join(self.demo_dump, 'static', 'input')
    if not os.path.exists(upload_path):
      os.makedirs(upload_path)

    file_metas = self.request.files.get('file', None)
    if not file_metas:
      self.set_status(400)
      self.write(json.dumps({'code': 'InvalidUploadFile', 
                              'message': 'The input file is not uploaded correctly'}))
      self.finish()
      return

    file_paths = []
    file_names = []
    for meta in file_metas:
      _file_name = '%s_%s'%(str(uuid.uuid4()), meta['filename'])
      _file_path = os.path.join(upload_path, _file_name)

      # 接受文件
      result = yield self.asynProcess('RECEIVE', {
                  'file_path': _file_path,
                  'file_data': meta['body']
                })
      if result['status'] != 200:
        self.set_status(result['status'])
        result['code'] = 'InvalidUpload'
        result['message'] = 'Couldnt upload request file'
        self.write(json.dumps(result))
        return

      _file_path = result['path']
      _file_name = _file_path.split('/')
      file_paths.append(_file_path)
      file_names.append(_file_name)

      # 仅支持单文件上传，处理
      break

    model_datas = []
    model_data_names = []
    model_data_types = []
    # 打包
    result = yield self.asynProcess('PACKAGE', {'path': file_paths[0]})
    if result['status'] != 200:
      self.set_status(result['status'])
      self.write(json.dumps(result))
      return
    
    model_data, model_data_name, model_data_type = result['data']
    model_datas.append(model_data)
    model_data_names.append(model_data_name)
    model_data_types.append(model_data_type)

    request_param = {'id': model_data_names[0]}
    model_input = (model_datas[0], model_data_types[0], request_param)

    # 设置等待队列
    BaseHandler.waiting_queue[model_data_names[0]] = queue.Queue()

    # 发送到处理队列
    self.dataset_queue.put(model_input)
    
    # 异步等待响应(设置等待队列，并在异步线程中等待)
    data_response = yield self.waitingResponse(model_data_names[0])

    # 绑定输入数据,返回
    data_response['INPUT_TYPE'] = model_data_types[0]
    if model_data_types[0] in ['FILE', 'IMAGE', 'AUDIO', 'VIDEO']:
      data_response['INPUT'] = '/static/input/%s' % model_data_names[0]
    else:
      data_response['INPUT'] = model_datas[0]

    # 返回成功
    self.write(json.dumps(data_response))
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
                      demo_config,
                      parent_id,
                      dataset_queue,
                      request_waiting_time=30):
  # register sig
  signal.signal(signal.SIGTERM, GracefulExitException.sigterm_handler)
  
  try:
    # 0.step define http server port
    define('port', default=demo_config['port'], help='run on port')

    # 1.step prepare static resource
    static_folder = '/'.join(os.path.dirname(__file__).split('/')[0:-1])
    demo_static_dir = os.path.join(demo_config['dump_dir'], 'static')
    if not os.path.exists(demo_static_dir):
      os.makedirs(demo_static_dir)
  
    for static_file in os.listdir(os.path.join(static_folder, 'resource', 'static')):
      if static_file[0] == '.':
        continue
    
      shutil.copy(os.path.join(static_folder, 'resource', 'static', static_file), demo_static_dir)
  
    # 2.step prepare html template
    demo_tempate_dir = os.path.join(demo_config['dump_dir'], 'templates')
  
    if not os.path.exists(demo_tempate_dir):
      os.makedirs(demo_tempate_dir)

    html_template = demo_config['html_template']
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
      'port': demo_config['port'],
      'demo_dump': demo_config['dump_dir'],
      'demo_name': demo_name,
      'demo_type': demo_type,
      'demo_description': demo_config['description_config'],
      'support_user_upload': demo_config['interaction']['support_user_upload'],
      'support_user_input': demo_config['interaction']['support_user_input'],
      'support_user_interaction': demo_config['interaction']['support_user_interaction'],
      'support_user_constraint': demo_config['interaction']['support_user_constraint'],
      'dataset_queue': dataset_queue,
      'request_waiting_time': request_waiting_time
    }
    app = tornado.web.Application(handlers=[(r"/", IndexHandler),
                                            (r"/demo", IndexHandler),
                                            (r"/api/query/", ClientQueryHandler),
                                            (r"/api/comment/", ClientCommentHandler),
                                            (r"/api/response/", ClientResponseHandler),
                                            (r"/api/cli-query/", ClientCliQueryHandler),
                                            (r"/submit/", ClientFileUploadAndProcessHandler),
                                            (r"/.*/static/.*", PrefixRedirectHandler)],
      **settings)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
  
    logger.info('demo is providing server on port %d'%demo_config['port'])
    tornado.ioloop.IOLoop.instance().start()
    logger.info('demo stop server')
  except GracefulExitException:
    logger.info('demo server exit')
    sys.exit(0)
  except KeyboardInterrupt:
    os.kill(parent_id, signal.SIGKILL)
    

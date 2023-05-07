# -*- coding: UTF-8 -*-
# @Time    : 2022/9/18 13:31
# @File    : demo.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import queue
import shutil
import threading
import concurrent.futures
import uuid

from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.functional.option import Some
from .serve import _APIWrapper,_PipeWrapper, _decode_content
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Response
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import cv2


class DemoMixin:
  def demo(self, input=[], output=[], app=None, default_config=None):
    if app is None:
      from fastapi import FastAPI, Request
      app = FastAPI()
    else:
      from fastapi import Request
    api = _APIWrapper.tls.place_holder
    pipeline = _PipeWrapper(self._iterable, api)

    input_selection = [cc['data'] for cc in input]
    input_selection_types = [cc['type'] for cc in input]
    for ui_type in input_selection_types:
      assert(ui_type in ['image', 'video', 'text', 'slider', 'checkbox', 'select'])
    
    output_selection = [cc['data'] for cc in output]
    output_selection_types = [cc['type'] for cc in output]
    for ui_type in output_selection_types:
      assert (ui_type in ['image', 'video', 'text', 'number'])

    input_config = default_config
    if default_config is None:
      input_config = [{} for _ in range(len(input_selection))]

    dump_folder = './dump'
    if not os.path.exists(dump_folder):
      os.makedirs(dump_folder)

    resource_dir = '/'.join(os.path.dirname(__file__).split('/')[0:-3])
    static_folder = os.path.join(dump_folder, 'demo', 'static')
    if os.path.exists(static_folder):
      shutil.rmtree(static_folder)
    shutil.copytree(os.path.join(resource_dir, 'resource', 'app'), static_folder)

    if not os.path.exists(os.path.join(static_folder, 'image')):
      os.makedirs(os.path.join(static_folder, 'image'))

    app.add_middleware(
      CORSMiddleware,
      # 允许跨域的源列表，例如 ["http://www.example.org"] 等等，["*"] 表示允许任何源
      allow_origins=["http://localhost:8080"],
      # 跨域请求是否支持 cookie，默认是 False，如果为 True，allow_origins 必须为具体的源，不可以是 ["*"]
      allow_credentials=True,
      # 允许跨域请求的 HTTP 方法列表，默认是 ["GET"]
      allow_methods=["*"],
      # 允许跨域请求的 HTTP 请求头列表，默认是 []，可以使用 ["*"] 表示允许所有的请求头
      # 当然 Accept、Accept-Language、Content-Language 以及 Content-Type 总之被允许的
      allow_headers=["*"],
      # 可以被浏览器访问的响应头, 默认是 []，一般很少指定
      # expose_headers=["*"]
      # 设定浏览器缓存 CORS 响应的最长时间，单位是秒。默认为 600，一般也很少指定
      # max_age=1000
    )

    @app.post('/antgo/api/demo/submit/')
    async def wrapper(req: Request):
      nonlocal pipeline
      req = await _decode_content(req)
      req = json.loads(req['query'])
      for i, b in enumerate(input_selection_types):
        if b in ['image', 'video', 'file']:
          req[i] = os.path.join(dump_folder, req[i])
        if b == 'checkbox':
          req[i] = bool(req[i])

      if len(req) == 1:
        req = req[0]
      rsp = pipeline.execute(req)
      # 输出与类型对齐
      rsp_value = rsp.get()
      output_info = {}
      for i, b in enumerate(output_selection):
        if output_selection_types[i] in ['image', 'video', 'file']:
          value = rsp_value.__dict__[b]
          if type(value) == str:
            shutil.copyfile(value, os.path.join(static_folder, 'image'))
            file_name = value.split('/')[-1]
            value = f'image/{file_name}'
          else:
            if value.dtype == np.uint8:
              transfer_result = value
            else:
              data_min = np.min(value)
              data_max = np.max(value)
              transfer_result = ((value - data_min) / (data_max - data_min) * 255).astype(np.uint8)

            if len(value.shape) == 3:
              assert (value.shape[2] == 3 or value.shape[2] == 4)

            assert (len(value.shape) == 2 or len(value.shape) == 3)
            file_name = f'{uuid.uuid4()}.png'
            cv2.imwrite(os.path.join(static_folder, 'image', file_name), transfer_result)
            value = f'image/{file_name}'

          output_info[b] = {
            'type': output_selection_types[i],
            'name': b,
            'value': value
          }
        else:
          output_info[b] = {
            'type': output_selection_types[i],
            'name': b,
            'value': rsp_value.__dict__[b]
          }
      return output_info

    @app.get('/')
    async def home(request: Request):
      return FileResponse(os.path.join(static_folder, 'index.html'))

    @app.get('/antgo/api/info/')
    async def info():
      return {
        'status': 'OK',
        'message': '',
        'content': {
          'project_type': 'DEMO',
          'project_state': {}
        }
      }

    @app.get('/antgo/api/user/info/')
    async def user_info():
      return {
        'status': 'OK',
        'message': '',
        'content': {
          'user_name': 'ANTGO',
          'short_name': 'A',
          'task_name': 'DEFAULT',
          'task_type': 'DEFAULT',
          'project_type': 'DEMO',
        }
      }

    @app.post('/antgo/api/demo/upload/')
    async def upload(file: UploadFile = File(...)):
      try:
        contents = file.file.read()
        with open(os.path.join(dump_folder, file.filename), 'wb') as f:
          f.write(contents)
      except Exception:
        return {"message": "There was an error uploading the file"}
      finally:
        file.file.close()

    @app.get('/antgo/api/demo/query_config/')
    async def query_config():
      input_info = []
      for k, v, config in zip(input_selection, input_selection_types, input_config):
        info = {
          'type': v,
          'name': k,
          'value': ''
        }
        if v == 'text':
          info['value'] = config.get('value', '')
        if v == 'slider':
          info['value'] = config.get('value', 0)
          info['min'] = config.get('min', 0)
          info['max'] = config.get('max', 100)
        if v == 'checkbox':
          info['value'] = (int)(config.get('value', False))
        if v == 'select':
          info['value'] = config.get('value', '')
          info['options'] = config.get('options', [])
          assert(len(info['options']) >= 1)
          value_list = [option['value'] for option in info['options']]
          assert(info['value'] in value_list)

        input_info.append(info)

      return input_info

    # static resource
    app.mount("/", StaticFiles(directory=static_folder), name="static")
    return app

  @classmethod
  def web(cls, index=None):
    return _APIWrapper(index=index, cls=cls)
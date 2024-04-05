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
import imagesize

from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.functional.option import Some
from .interactive import *
from .serve import _APIWrapper,_PipeWrapper, _decode_content
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Response
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
import json
import numpy as np
import cv2


class DemoMixin:
  app = None
  pipeline_info = {}
  def demo(self, input=[], output=[], title="", description="", default_config=None):
    api = _APIWrapper.tls.placeholder
    DemoMixin.pipeline_info[api._name] = {
       'exe': _PipeWrapper(self._iterable, api)
	  }

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

    DemoMixin.pipeline_info[api._name].update({
       'input_selection': input_selection,
       'input_selection_types': input_selection_types,
       'output_selection': output_selection,
       'output_selection_types': output_selection_types,
       'input_config': input_config,
       'title': title,
       'description': description,
       'interactive': {},
       'step_i': api.step_i,
       'step_num': api.step_num
	})

    for b in input_selection:
        if b in InteractiveMixin.interactive_elements:
            DemoMixin.pipeline_info[api._name]['interactive'][b] = {
				'mode': InteractiveMixin.interactive_elements[b]['mode'],
				'num': InteractiveMixin.interactive_elements[b]['num']
			}

    dump_folder = './dump'
    if not os.path.exists(dump_folder):
      os.makedirs(dump_folder)
    resource_dir = '/'.join(os.path.dirname(__file__).split('/')[0:-3])
    static_folder = os.path.join(dump_folder, 'demo', 'static')

    if DemoMixin.app is not None:
        return DemoMixin.app

    from fastapi import FastAPI, Request
    DemoMixin.app = FastAPI()

    if os.path.exists(static_folder):
      shutil.rmtree(static_folder)
    shutil.copytree(os.path.join(resource_dir, 'resource', 'app'), static_folder)

    if not os.path.exists(os.path.join(static_folder, 'image', 'query')):
      os.makedirs(os.path.join(static_folder, 'image', 'query'))

    if not os.path.exists(os.path.join(static_folder, 'image', 'response')):
      os.makedirs(os.path.join(static_folder, 'image', 'response'))

    DemoMixin.app.add_middleware(
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

    @DemoMixin.app.post('/antgo/api/demo/submit/')
    async def wrapper(req: Request):        
        req = await _decode_content(req)
        req = json.loads(req['query'])

        demo_name = ''
        if 'demo' in req:
           demo_name = req['demo']
        if demo_name == '':
           demo_name = list(DemoMixin.pipeline_info.keys())[0]

        if demo_name not in DemoMixin.pipeline_info:
            raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")

        input_req = req['input']
        element_req = req['element']
        input_selection_types = DemoMixin.pipeline_info[demo_name]['input_selection_types']
        # image: 文件
        # video: 文件
        query_folder = os.path.join(static_folder, 'image', 'query')
        for i, b in enumerate(input_selection_types):
            if b in ['image', 'video', 'file']:
                if b == 'image':
                   input_req[i] = '/'.join(input_req[i].split('/')[2:])
                   input_req[i] = cv2.imread(f'{query_folder}/{input_req[i]}')
                else:
                   input_req[i] = '/'.join(input_req[i].split('/')[2:])
                   input_req[i] = f'{query_folder}/{input_req[i]}'
            if b == 'checkbox':
                input_req[i] = bool(input_req[i])

        input_selection = DemoMixin.pipeline_info[demo_name]['input_selection']
        feed_info = {}
        for a,b in zip(input_selection, input_req):
           feed_info[a] = b

        interactive_info = {}
        for i,b in enumerate(element_req):
            data = []
            for info in b['value']:
               data.append(info['data'])
            bind_name = b['name']
            assert(bind_name in InteractiveMixin.interactive_elements)
            interactive_info[InteractiveMixin.interactive_elements[bind_name]['target']] = data

        feed_info.update(interactive_info)
        rsp_value = DemoMixin.pipeline_info[demo_name]['exe'].execute(feed_info)

        if rsp_value is None:
           raise HTTPException(status_code=500, detail="管线执行错误")

        # 输出与类型对齐
        output_selection = DemoMixin.pipeline_info[demo_name]['output_selection']
        output_selection_types = DemoMixin.pipeline_info[demo_name]['output_selection_types']
        output_info = {}
        for i, b in enumerate(output_selection):
            if output_selection_types[i] in ['image', 'video', 'file']:
                if b not in rsp_value.__dict__:
                   continue

                value = rsp_value.__dict__[b]
                image_width, image_height = 0, 0
                
                if output_selection_types[i] == 'image':
                    if type(value) == str:
                        shutil.copyfile(value, os.path.join(static_folder, 'image', 'response'))
                        image_width, image_height = imagesize.get(value)
                        file_name = value.split('/')[-1]
                        value = f'image/response/{file_name}'
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
                        cv2.imwrite(os.path.join(static_folder, 'image', 'response', file_name), transfer_result)
                        image_width, image_height = imagesize.get(os.path.join(static_folder, 'image', 'response', file_name))
                        value = f'image/response/{file_name}'

                    output_info[b] = {
                        'type': output_selection_types[i],
                        'name': b,
                        'value': value,
                        'height': image_height,
                        'width': image_width
                    }
                    if b in InteractiveMixin.interactive_elements:
                        output_info[b]['interactive'] = True
                        output_info[b]['element'] = {
						    'mode': InteractiveMixin.interactive_elements[b]['mode'],
							'num': InteractiveMixin.interactive_elements[b]['num']
                        }
                else:
                    shutil.copyfile(value, os.path.join(static_folder, 'image', 'response', value.split('/')[-1]))
                    output_info[b] = {
                        'type': output_selection_types[i],
                        'name': b,
                        'value': 'image/response/'+value.split('/')[-1]
                    }
            else:
                output_info[b] = {
					'type': output_selection_types[i],
					'name': b,
					'value': rsp_value.__dict__[b]
                }

        return output_info

    @DemoMixin.app.get('/')
    async def home(request: Request):
      return FileResponse(os.path.join(static_folder, 'index.html'))

    @DemoMixin.app.get('/antgo/api/info/')
    async def info():
      return {
        'status': 'OK',
        'message': '',
        'content': {
          'project_type': 'DEMO',
          'project_state': {}
        }
      }

    @DemoMixin.app.get('/antgo/api/user/info/')
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

    @DemoMixin.app.post('/antgo/api/demo/upload/')
    async def upload(file: UploadFile = File(...)):
        response = {}
        try:
            contents = file.file.read()
            filename = file.filename
            query_folder = os.path.join(static_folder, 'image', 'query')
            
            random_id = str(uuid.uuid4())
            with open(os.path.join(query_folder, f'{random_id}_{filename}'), 'wb') as f:
                f.write(contents)

            image_path = os.path.join(query_folder, f'{random_id}_{filename}')
            response['path'] = f'image/query/{random_id}_{filename}'
            response['width'] = 0
            response['height'] = 0

            if image_path.split('.')[-1].lower() in ['jpg', 'jpeg', 'png']:
               image_width, image_height = imagesize.get(image_path)
               response['width'] = image_width
               response['height'] = image_height
        except Exception:
            return {"message": "There was an error uploading the file"}
        finally:
            file.file.close()

        return response

    @DemoMixin.app.get('/antgo/api/demo/query_config/')
    async def query_config(req: Request):
      demo_name = ''
      if req.query_params['demo'] != '':
         demo_name = req.query_params['demo']

      if demo_name == '':
         demo_name = list(DemoMixin.pipeline_info.keys())[0]

      if demo_name not in DemoMixin.pipeline_info:
         raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")

      input_selection = DemoMixin.pipeline_info[demo_name]['input_selection']
      input_selection_types = DemoMixin.pipeline_info[demo_name]['input_selection_types']
      input_config = DemoMixin.pipeline_info[demo_name]['input_config']
      title = DemoMixin.pipeline_info[demo_name]['title']
      description = DemoMixin.pipeline_info[demo_name]['description']
      interactive = DemoMixin.pipeline_info[demo_name]['interactive']

      input_info = []
      for k, v, config in zip(input_selection, input_selection_types, input_config):
        info = {
          'type': v,
          'name': k,
          'value': '',
          'interactive': {},
          'ready': False
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

        if k in interactive:
             info['interactive'] = interactive[k]
        input_info.append(info)

      step_i = DemoMixin.pipeline_info[demo_name]['step_i']
      step_num = DemoMixin.pipeline_info[demo_name]['step_num']
      pre_step = ''
      next_step = ''
      if step_i - 1 >= 0:
        for pname, pinfo in DemoMixin.pipeline_info.items():
          if pinfo['step_i'] == step_i - 1:
            pre_step = pname
      if step_i + 1 < len(DemoMixin.pipeline_info):
        for pname, pinfo in DemoMixin.pipeline_info.items():
          if pinfo['step_i'] == step_i + 1:
            next_step = pname

      info = {
        'input': input_info,
        'title': title,
        'description': description,
        'pre_step': pre_step,
        'next_step': next_step
	    }
      return info

    # static resource
    DemoMixin.app.mount("/", StaticFiles(directory=static_folder), name="static")

    return DemoMixin.app

  @classmethod
  def web(cls, index=None, name='demo', **kwargs):
    return _APIWrapper(index=index, cls=cls, name=name, **kwargs)
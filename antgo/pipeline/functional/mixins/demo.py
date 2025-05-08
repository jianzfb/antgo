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
import pathlib
import logging

from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.functional.option import Some
from antgo.tools.download_funcs import *
from .interactive import *
from .serve import PipelineExecuter, _decode_content
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Response
from fastapi import File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
import secrets

import json
import numpy as np
import cv2


class DemoMixin:
  def demo(self, input=[], output=[], title="", description="", default_config=None, **kwargs):
    pipeline_name = ServerInfo.pipeline_name
    server_index = ServerInfo.pipeline_info[pipeline_name]['step_i']
    server_num = ServerInfo.pipeline_info[pipeline_name]['step_num']    
    ServerInfo.pipeline_info[pipeline_name].update({
       'exe': PipelineExecuter(self._iterable, ServerInfo.pipeline_info[pipeline_name]['entry'])
	  })

    input_selection = [cc['data'] for cc in input]
    input_selection_types = [cc['type'] for cc in input]
    for ui_type in input_selection_types:
      assert(ui_type in ['image', 'video', 'text', 'slider', 'checkbox', 'select', 'image-search'])

    output_selection = [cc['data'] for cc in output]
    output_selection_types = [cc['type'] for cc in output]
    for ui_type in output_selection_types:
      assert (ui_type in ['image', 'video', 'text', 'number', 'file', 'json'])

    input_config = default_config
    if default_config is None:
      input_config = [{} for _ in range(len(input_selection))]

    ServerInfo.pipeline_info[pipeline_name].update({
       'input_selection': input_selection,
       'input_selection_types': input_selection_types,
       'output_selection': output_selection,
       'output_selection_types': output_selection_types,
       'input_config': input_config,
       'title': title,
       'description': description,
       'interactive': {},
       'step_i': server_index,
       'step_num': server_num
	})

    for b in input_selection:
        if f'{pipeline_name}/{b}' in InteractiveMixin.interactive_elements:
            ServerInfo.pipeline_info[pipeline_name]['interactive'][b] = {
				'mode': InteractiveMixin.interactive_elements[f'{pipeline_name}/{b}']['mode'],
				'num': InteractiveMixin.interactive_elements[f'{pipeline_name}/{b}']['num']
			}

    dump_folder = './dump'
    if not os.path.exists(dump_folder):
      os.makedirs(dump_folder)
    resource_dir = '/'.join(os.path.dirname(__file__).split('/')[0:-3])
    static_folder = os.path.join(dump_folder, 'demo', 'static')

    if ServerInfo.app is not None:
        return ServerInfo.app

    from fastapi import FastAPI, Request
    ServerInfo.app = FastAPI()
    ServerInfo.app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

    if not os.path.exists(static_folder):
      # 复制标准web工程
      shutil.copytree(os.path.join(resource_dir, 'resource', 'app'), static_folder)

    if not os.path.exists(os.path.join(static_folder, 'image', 'query')):
      os.makedirs(os.path.join(static_folder, 'image', 'query'))

    if not os.path.exists(os.path.join(static_folder, 'image', 'response')):
      os.makedirs(os.path.join(static_folder, 'image', 'response'))

    ServerInfo.app.add_middleware(
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

    @ServerInfo.app.post('/antgo/api/demo/submit/')
    async def wrapper(req: Request, response: RedirectResponse):        
        if(req.session.get("session_id") is None):
          session_id = req.session["session_id"] = secrets.token_urlsafe(32)
          response.set_cookie(key="Authorization", value=session_id)

        session_id = req.session.get("session_id")
        req = await _decode_content(req)
        req = json.loads(req['query'])

        demo_name = ''
        if 'demo' in req:
           demo_name = req['demo']
        if demo_name == '':
           demo_name = list(ServerInfo.pipeline_info.keys())[0]

        if demo_name not in ServerInfo.pipeline_info:
            raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")

        input_req = req['input']
        element_req = req['element']
        input_selection_types = ServerInfo.pipeline_info[demo_name]['input_selection_types']
        # image: 文件
        # video: 文件
        query_folder = os.path.join(static_folder, 'image', 'query')
        for i, b in enumerate(input_selection_types):
            if b in ['image', 'video', 'file']:
                if b == 'image':
                   if input_req[i] is not None:
                    input_req[i] = '/'.join(input_req[i].split('/')[2:])
                    input_req[i] = cv2.imread(f'{query_folder}/{input_req[i]}', cv2.IMREAD_UNCHANGED)
                else:
                   if input_req[i] is not None:
                    input_req[i] = '/'.join(input_req[i].split('/')[2:])
                    input_req[i] = f'{query_folder}/{input_req[i]}'
            if b == 'checkbox':
                input_req[i] = bool(int(input_req[i]))

            if b == 'image-search':
                selected_image_list = []
                for selected_filename in input_req[i]:
                   selected_filepath = f'{static_folder}/{selected_filename}'
                   selected_image_list.append(selected_filepath)
                input_req[i] = selected_image_list

        input_selection = ServerInfo.pipeline_info[demo_name]['input_selection']
        feed_info = {}
        for a,b in zip(input_selection, input_req):
           feed_info[a] = b

        interactive_info = {}
        for i,b in enumerate(element_req):
            data = []
            for info in b['value']:
               data.append(info['data'])
            bind_name = b['name']
            assert(f'{demo_name}/{bind_name}' in InteractiveMixin.interactive_elements)
            interactive_info[InteractiveMixin.interactive_elements[f'{demo_name}/{bind_name}']['target']] = data

        feed_info.update(interactive_info)
        feed_info.update(
          {'session_id': session_id}
        )

        rsp_value = await ServerInfo.pipeline_info[demo_name]['exe'].execute(feed_info)
        if rsp_value is None:
           raise HTTPException(status_code=500, detail="server abnormal")

        # 输出与类型对齐
        output_selection = ServerInfo.pipeline_info[demo_name]['output_selection']
        output_selection_types = ServerInfo.pipeline_info[demo_name]['output_selection_types']
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
                    if f'{demo_name}/{b}' in InteractiveMixin.interactive_elements:
                        output_info[b]['interactive'] = True
                        output_info[b]['element'] = {
                            'mode': InteractiveMixin.interactive_elements[f'{demo_name}/{b}']['mode'],
                            'num': InteractiveMixin.interactive_elements[f'{demo_name}/{b}']['num']
                        }
                else:
                    shutil.copyfile(value, os.path.join(static_folder, 'image', 'response', value.split('/')[-1]))
                    output_info[b] = {
                        'type': output_selection_types[i],
                        'name': b,
                        'value': 'image/response/'+value.split('/')[-1]
                    }
            elif output_selection_types[i] == 'json':
                value = rsp_value.__dict__[b]
                if not isinstance(value, str):
                  value = json.dumps(value)
                output_info[b] = {
                  'type': 'text',
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

    @ServerInfo.app.get('/')
    async def home(request: Request):
      return FileResponse(os.path.join(static_folder, 'index.html'))

    @ServerInfo.app.get('/antgo/api/info/')
    async def info():
      return {
        'status': 'OK',
        'message': '',
        'content': {
          'project_type': 'DEMO',
          'project_state': {}
        }
      }

    @ServerInfo.app.get('/antgo/api/user/info/')
    async def user_info():
        return {
            'status': 'OK',
            'message': '',
            'content': {
                'user_name': 'ANTGO',
                'short_name': 'ANTGO',
                'task_name': 'DEFAULT',
                'task_type': 'DEFAULT',
                'project_type': 'DEMO',
            }
        }

    @ServerInfo.app.post('/antgo/api/demo/upload/')
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

    @ServerInfo.app.get('/antgo/api/demo/query_config/')
    async def query_config(req: Request):
      demo_name = ''
      if req.query_params['demo'] != '':
         demo_name = req.query_params['demo']

      if demo_name == '':
         demo_name = list(ServerInfo.pipeline_info.keys())[0]

      if demo_name not in ServerInfo.pipeline_info:
         raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")

      input_selection = ServerInfo.pipeline_info[demo_name]['input_selection']
      input_selection_types = ServerInfo.pipeline_info[demo_name]['input_selection_types']
      input_config = ServerInfo.pipeline_info[demo_name]['input_config']
      title = ServerInfo.pipeline_info[demo_name]['title']
      description = ServerInfo.pipeline_info[demo_name]['description']
      interactive = ServerInfo.pipeline_info[demo_name]['interactive']

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

      step_i = ServerInfo.pipeline_info[demo_name]['step_i']
      step_num = ServerInfo.pipeline_info[demo_name]['step_num']
      pre_step = ''
      next_step = ''
      if step_i - 1 >= 0:
        for pname, pinfo in ServerInfo.pipeline_info.items():
          if pinfo['step_i'] == step_i - 1:
            pre_step = pname
      if step_i + 1 < len(ServerInfo.pipeline_info):
        for pname, pinfo in ServerInfo.pipeline_info.items():
          if pinfo['step_i'] == step_i + 1:
            next_step = pname
      info = {
        'input': input_info,
        'title': title,
        'description': description,
        'pre_step': pre_step,
        'next_step': next_step,
	    }
      return info

    @ServerInfo.app.get('/antgo/api/demo/search/')
    async def search(req: Request):
      # 调用下载功能
      demo_name = ''
      if 'demo' in req.query_params and req.query_params['demo'] != '':
         demo_name = req.query_params['demo']

      if demo_name == '':
         demo_name = list(ServerInfo.pipeline_info.keys())[0]

      if demo_name not in ServerInfo.pipeline_info:
         raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")
      
      print(req.query_params)
      search_engine = req.query_params['search_engine']
      search_word = req.query_params['search_word']
      
      search_func = {
         'baidu': download_from_baidu,
         'bing': download_from_bing,
         'vcg': download_from_vcg
      }

      search_word = search_word.replace(',','/')
      search_word = search_word.replace(' ', '/')
      keys = f'type:image,keyword:{search_word}'
      print(keys)
      timestamp = str(time.time())
      target_folder = os.path.join(static_folder, 'image', 'download', demo_name, search_engine, timestamp)
      os.makedirs(target_folder, exist_ok=True)

      t = threading.Thread(target=search_func[search_engine], args=(target_folder, keys, None, 50))
      t.start()

    @ServerInfo.app.get('/antgo/api/demo/searchprocess/')
    async def process(req: Request):
        demo_name = ''
        if 'demo' in req.query_params and req.query_params['demo'] != '':
            demo_name = req.query_params['demo']

        if demo_name == '':
            demo_name = list(ServerInfo.pipeline_info.keys())[0]

        if demo_name not in ServerInfo.pipeline_info:
            raise HTTPException(status_code=404, detail=f"{demo_name} not exist.")

        target_folder = os.path.join(static_folder, 'image', 'download', demo_name)
        if not os.path.exists(target_folder):
           return {'imagelist': []}

        ready_file_list = []
        for engine_name in os.listdir(target_folder):
            if engine_name[0] == '.':
               continue
            for subfolder in os.listdir(os.path.join(target_folder, engine_name)):
                if subfolder[0] == '.':
                   continue
                for filename in os.listdir(os.path.join(target_folder, engine_name, subfolder)):
                    if filename[0] == '.':
                       continue
                    filepath = os.path.join(target_folder, engine_name, subfolder, filename)
                    fileext = filepath.split('.')[-1]
                    if fileext.lower() not in ['jpeg', 'jpg', 'png', 'webp']:
                        continue
                    ready_file_list.append(f'image/download/{demo_name}/{engine_name}/{subfolder}/{filename}')

        return {
           'imagelist': ready_file_list
        }

    @ServerInfo.app.post("/{pipeline_name}/execute/")
    async def execute(pipeline_name: str, req: Request):
        input_req = await _decode_content(req)
        try:
            input_req = json.loads(input_req)
        except:
            logging.error('Fail to parsing request.')
            raise HTTPException(status_code=404, detail="请求不合规")

        if pipeline_name not in ServerInfo.pipeline_info:
            raise HTTPException(status_code=404, detail="服务不存在")

        input_selection_types = ServerInfo.pipeline_info[pipeline_name]['input_selection_types']
        input_selection = ServerInfo.pipeline_info[pipeline_name]['input_selection']
        for input_name, input_type in zip(input_selection, input_selection_types):
            if input_name not in input_req:
                input_req[input_name] = None
                continue

            if input_type == 'image':
              decoded_data = base64.b64decode(input_req[input_name])
              decoded_data = np.frombuffer(decoded_data, dtype='uint8')
              input_req[input_name] = cv2.imdecode(decoded_data, cv2.IMREAD_UNCHANGED)
            elif input_type in ['video', 'file']:
              input_req[input_name] = os.path.join(static_folder, 'image', 'query', input_req[input_name])

        feed_info = {}
        for input_name in input_selection:
          if input_name not in input_req:
              feed_info[input_name] = None
              continue

          feed_info[input_name] = input_req[input_name]
        feed_info.update(
          {'session_id': uuid.uuid4()}
        )

        rsp_value = await ServerInfo.pipeline_info[pipeline_name]['exe'].execute(feed_info)
        if rsp_value is None:
            raise HTTPException(status_code=500, detail="管线执行错误")

        output_selection_types = ServerInfo.pipeline_info[pipeline_name]['output_selection_types']
        output_selection = ServerInfo.pipeline_info[pipeline_name]['output_selection']

        output_info = {}
        for i, b in enumerate(output_selection):
            if output_selection_types[i] in ['image', 'video', 'file']:
                if b not in rsp_value.__dict__:
                   continue

                value = rsp_value.__dict__[b]
                if output_selection_types[i] == 'image':
                    # 图像存储方式
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
                    output_info[b] = f'image/response/{file_name}'
                else:
                    # 视频或其他文件处理方式
                    assert(isinstance(value, str))
                    shutil.copyfile(value, os.path.join(static_folder, 'image', 'response', value.split('/')[-1]))
                    output_info[b] = os.path.join(static_folder, 'image', 'response', value.split('/')[-1])
                    output_info[b] = f'image/response/{value.split("/")[-1]}'
            else:
                output_info[b] = rsp_value.__dict__[b]

        return output_info

    @ServerInfo.app.get("/file/download/")
    async def download(req: Request):
      file_path = req.query_params['file_name']
      return FileResponse(os.path.join(static_folder, file_path))

    @ServerInfo.app.post("/file/upload/")
    async def upload(file: UploadFile):
      filename = file.filename
      file_size = file.size
      fileid = str(uuid.uuid4())

      unique_filename = f'{fileid}-{filename}'
      os.makedirs(os.path.join(static_folder, 'image', 'query'), exist_ok=True)
      with open(os.path.join(static_folder, 'image', 'query', unique_filename), "wb") as f:
          for chunk in iter(lambda: file.file.read(1024), b''):
              f.write(chunk)
      return {"fileid": unique_filename, 'filepath': f'/image/query/{unique_filename}'}

    # static resource
    ServerInfo.app.mount("/", StaticFiles(directory=static_folder), name="static")

    return ServerInfo.app

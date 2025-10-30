# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 22:43
# @File    : serve.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import torchaudio
import torch
import queue
import concurrent.futures
from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.functional.option import Some
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.application.common.db import *
from antgo.pipeline.application.table import *
from antgo.pipeline.functional.common.env import *
from antgo.pipeline.utils.reserved import *
from antgo.pipeline.functional.g import *
from fastapi.responses import RedirectResponse,HTMLResponse, FileResponse
from fastapi import HTTPException
from fastapi import File, UploadFile
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles
from concurrent.futures import ThreadPoolExecutor
import starlette
import asyncio
import urllib
from urllib.parse import parse_qs
import secrets
import logging
import json
import uuid
import base64
import io
import cv2
import logging


class ServerInfo(object):
    app = None
    db = None
    pipeline_info = {}
    pipeline_name = None
    index = -1


class SourceEntry:
    # 数据源
    def __init__(self, index=None, cls=None, name='demo', **kwargs) -> None:
        self._cls = cls
        self._name = name

        if index is not None:
            self._index = index if isinstance(index, list) else [index]
        else:
            self._index = index

    def __iter__(self):
        while True:
            # do nothing
            print('shouldnt run default pipeline drive')
            yield None

    def __enter__(self):
        return self._cls(self).stream()

    def __exit__(self, exc_type, exc_value, traceback):
        pass


async def _decode_content(req):
    if getattr(req, 'headers', None):
        from multipart.multipart import parse_options_header
        content_type_header = req.headers.get('Content-Type')
        content_type, _ = parse_options_header(content_type_header)
        if content_type in {b'multipart/form-data'}:
            return await req.form()
        if content_type.startswith(b'image/'):
            return await req.body()
        if content_type.startswith(b'application/x-www-form-urlencoded'):
            body_str = (await req.body()).decode()
            body_info = parse_qs(body_str)
            body_info = {k: v[0] for k, v in body_info.items()}
            return body_info

        if len(req._query_params) > 0:
            return dict(req._query_params)

    info = (await req.body()).decode()
    return info


class ServeMixin:
    """
    Mixin for API serve
    """
    isDebug = False
    def serve(self, input=[], output=[], default_config=None, db_config=None, **kwargs):
        """
        Serve the DataFrame as a RESTful API

        Args:
            path (str, optional): API path. Defaults to '/'.
            app (_type_, optional): The FastAPI app the API bind to, will create one if None.

        Returns:
            _type_: the app that bind to
        """      
        pipeline_name = ServerInfo.pipeline_name
        input_selection = [cc['data'] for cc in input]
        input_selection_types = [cc['type'] for cc in input]
        for ui_type in input_selection_types:
            assert(ui_type in ['image', 'sound', 'sound/pcm', 'video', 'text', 'slider', 'checkbox', 'select', 'image-search', 'header', 'image-ext', 'sound-ext'])

        output_selection = [cc['data'] for cc in output]
        output_selection_types = [cc['type'] for cc in output]
        for ui_type in output_selection_types:
            assert (ui_type in ['image', 'sound', 'sound/pcm', 'video', 'text', 'number', 'file', 'json'])

        input_config = default_config
        if default_config is None:
            input_config = [{} for _ in range(len(input_selection))]

        ServerInfo.pipeline_info[pipeline_name].update(
            {
                'input_selection': input_selection,
                'input_selection_types': input_selection_types,
                'output_selection': output_selection,
                'output_selection_types': output_selection_types,
                'input_config': input_config,
                'response_unwarp': kwargs.get('response_unwarp',False)
            }
        )

        # 动态生成orm（基于管线算子需求）
        # 创建/加载数据库
        if ServerInfo.db is None and db_config is not None:
            is_new_create = False
            if not os.path.exists('./orm.py'):
                # 生成db orm
                # table_config_info = []
                # for table_info in get_table_info().values():
                #     table_config_info.append(table_info)

                create_db_orm(get_table_info())
                is_new_create = True

            update_db_orm(__import__('orm'))
            ServerInfo.db = create_db_session(db_config['db_url'])
            if is_new_create:
                # 创建默认记录
                table_default_records = get_table_default()
                db = ServerInfo.db()
                for table_name, tabel_default_records in table_default_records.items():
                    orm_table = getattr(get_db_orm(), table_name.capitalize())
                    for record_info in tabel_default_records:
                        update_info = []
                        for field_name, field_value in record_info.items():
                            # 如果field_name是foreign key, 则需要获取db 记录
                            if '/' in field_name:
                                foreign_table_name, foreign_field_name = field_name.split('/')
                                foreign_orm_table = getattr(get_db_orm(), foreign_table_name.capitalize())
                                foreign_record = db.query(foreign_orm_table).filter(getattr(foreign_orm_table, foreign_field_name) == field_value).one_or_none()
                                update_info.append((field_name, {foreign_table_name: foreign_record}))

                        if len(update_info) > 0:
                            for old_field_name, new_info in update_info:
                                record_info.pop(old_field_name)
                                record_info.update(new_info)
                        record = orm_table(**record_info)
                        db.add(record)
                db.commit()

        if ServerInfo.app is not None:
            return ServerInfo.app

        static_folder = './dump'
        os.makedirs(static_folder, exist_ok=True)
        os.makedirs( os.path.join(static_folder, 'image', 'query'), exist_ok=True)
        os.makedirs( os.path.join(static_folder, 'image', 'response'), exist_ok=True)

        from fastapi import FastAPI, Request
        ServerInfo.app = FastAPI()
        ServerInfo.app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

        @ServerInfo.app.post("/{server_name}/execute/")
        @ServerInfo.app.post("/{server_name}/execute")
        async def wrapper(server_name: str, req: Request, response: RedirectResponse):
            # 获得session_id
            if(req.session.get("session_id") is None):
                session_id = req.session["session_id"] = secrets.token_urlsafe(32)
            session_id = req.session.get("session_id")

            # 解析请求
            try:
                input_req = await _decode_content(req)
                if ServeMixin.isDebug:
                    print("--------- Request Info ------------")
                    print(input_req)
                    print("--------------------------------------")

                if input_req == '':
                    input_req = '{}'
                if isinstance(input_req, str):
                    input_req = json.loads(input_req)
                else:
                    kvmaps = {}
                    for k,v in zip(input_req.keys(), input_req.values()):
                        kvmaps[k] = v
                    input_req = kvmaps
            except:
                logging.error('Fail to parsing request.')
                raise HTTPException(status_code=400, detail="request format abnormal")

            # 判断服务是否存在
            if server_name not in ServerInfo.pipeline_info:
                raise HTTPException(status_code=404, detail=f"server {server_name} not exist")

            # 转换请求数据
            input_selection_types = ServerInfo.pipeline_info[server_name]['input_selection_types']
            input_selection = ServerInfo.pipeline_info[server_name]['input_selection']
            for input_name, input_type in zip(input_selection, input_selection_types):
                if input_type == 'header':
                    # 从http header获取数据
                    input_req[input_name] = req.headers.get(input_name, None)
                    continue

                if input_name not in input_req:
                    input_req[input_name] = None
                    raise HTTPException(status_code=400, detail=f"request param {input_name} missing")

                if input_type.startswith('image'):
                    # 支持base64格式+URL格式+UploadFile
                    if isinstance(input_req[input_name], starlette.datastructures.UploadFile):
                        try:
                            file = input_req[input_name]
                            contents = await file.read()
                            filename = file.filename
                            if contents == b'':
                                raise HTTPException(status_code=400, detail=f"request {input_name}(image) read multi-form abnormal")

                            if input_type == 'image':
                                input_req[input_name] = cv2.imdecode(np.asarray(bytearray(contents), dtype="uint8"), 1)
                            else:
                                input_req[input_name] = {
                                    'image': cv2.imdecode(np.asarray(bytearray(contents), dtype="uint8"), 1),
                                    'filename': filename
                                }
                        except Exception:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(image) read multi-form abnormal")
                        else:
                            file.file.close()

                    elif input_req[input_name].startswith('http') or input_req[input_name].startswith('https'):
                        # url 格式
                        url = input_req[input_name]
                        try:
                            res = urllib.request.urlopen(input_req[input_name])
                            img = np.asarray(bytearray(res.read()), dtype="uint8")
                            if input_type == 'image':
                                input_req[input_name] = cv2.imdecode(img, 1)
                            else:
                                input_req[input_name] = {
                                    'image': cv2.imdecode(img, 1),
                                    'filename': url
                                }
                        except:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(image) read url abnormal")
                    else:
                        # base64格式
                        try:
                            decoded_data = base64.b64decode(input_req[input_name])
                            decoded_data = np.frombuffer(decoded_data, dtype='uint8')
                            if input_type == 'image':
                                input_req[input_name] = cv2.imdecode(decoded_data, 1)
                            else:
                                input_req[input_name] = {
                                    'image': cv2.imdecode(decoded_data, 1),
                                    'filename': None
                                }
                        except:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(image) read base64 abnormal")
                elif input_type.startswith('sound'):
                    # 支持base64格式+URL格式+UploadFile
                    if isinstance(input_req[input_name], starlette.datastructures.UploadFile):
                        signal, fs = None, None
                        try:
                            file = input_req[input_name]
                            contents = await file.read()
                            filename = file.filename
                            if contents == b'':
                                raise HTTPException(status_code=400, detail=f"request {input_name}(sound) read multi-form abnormal")

                            if not filename.endswith('pcm'):
                                signal, fs = torchaudio.load(io.BytesIO(contents), channels_first = False)
                            else:
                                # PCM格式，需要设置采样率，通道数
                                # 位深度：s16le，采样率：16000
                                sound_format = input_req.get(f'{input_name}/format', 's16le')
                                sound_sample_rate = input_req.get(f'{input_name}/sample_rate', '16000')
                                sound_channel_num = input_req.get(f'{input_name}/channel_num', '1')
                                streamer = torchaudio.io.StreamReader(src=io.BytesIO(contents), format=sound_format, option={"sample_rate": sound_sample_rate})
                                streamer.add_basic_audio_stream(
                                    frames_per_chunk=-1,  # 读取整个文件
                                    num_channels=sound_channel_num
                                )
                                
                                signal = []
                                for (chunk,) in streamer.stream():
                                    signal.append(chunk)
                                signal = torch.cat(signal, 0)
                                fs = int(sound_sample_rate)
                        except:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(sound) read multi-form abnormal")
                        else:
                            file.file.close()

                        sound_data = {
                            'signal': signal,
                            'fs': fs,
                            'filename': filename
                        }
                        input_req[input_name] = sound_data
                    elif input_req[input_name].startswith('http') or input_req[input_name].startswith('https'):
                        url = input_req[input_name]
                        signal, fs = None, None
                        # url 格式
                        try:
                            if not url.endswith('pcm'):
                                signal, fs = torchaudio.load(input_req[input_name], channels_first = False)
                            else:
                                # PCM格式，需要设置采样率，通道数
                                # 位深度：s16le，采样率：16000                                
                                sound_format = input_req.get(f'{input_name}/format', 's16le')
                                sound_sample_rate = input_req.get(f'{input_name}/sample_rate', '16000')
                                sound_channel_num = input_req.get(f'{input_name}/channel_num', '1')
                                streamer = torchaudio.io.StreamReader(src=input_req[input_name], format=sound_format, option={"sample_rate": sound_sample_rate})
                                streamer.add_basic_audio_stream(
                                    frames_per_chunk=-1,  # 读取整个文件
                                    num_channels=sound_channel_num
                                )

                                signal = []
                                for (chunk,) in streamer.stream():
                                    signal.append(chunk)
                                signal = torch.cat(signal, 0)
                                fs = int(sound_sample_rate)
                        except:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(sound) read url abnormal")

                        sound_data = {
                            'signal': signal,
                            'fs': fs,
                            'filename': url
                        }
                        input_req[input_name] = sound_data
                    else:
                        signal, fs = None, None
                        # base64格式
                        try:
                            if not input_type.endswith('pcm'):
                                decoded_data = base64.b64decode(input_req[input_name])
                                signal, fs = torchaudio.load(io.BytesIO(decoded_data), channels_first = False)
                            else:
                                # PCM格式，需要设置采样率，通道数
                                # 位深度：s16le，采样率：16000                                
                                sound_format = input_req.get(f'{input_name}/format', 's16le')
                                sound_sample_rate = input_req.get(f'{input_name}/sample_rate', '16000')
                                sound_channel_num = input_req.get(f'{input_name}/channel_num', '1')
                                decoded_data = base64.b64decode(input_req[input_name])
                                streamer = torchaudio.io.StreamReader(src=io.BytesIO(decoded_data), format=sound_format, option={"sample_rate": sound_sample_rate})
                                streamer.add_basic_audio_stream(
                                    frames_per_chunk=-1,  # 读取整个文件
                                    num_channels=sound_channel_num
                                )

                                signal = []
                                for (chunk,) in streamer.stream():
                                    signal.append(chunk)
                                signal = torch.cat(signal, 0)
                                fs = int(sound_sample_rate)
                        except:
                            raise HTTPException(status_code=400, detail=f"request {input_name}(sound) read base64 abnormal")
                        
                        sound_data = {
                            'signal': signal,
                            'fs': fs,
                            'filename': None
                        }
                        input_req[input_name] = sound_data
                elif input_type in ['video', 'file']:
                    input_req[input_name] = os.path.join(static_folder, 'image', 'query', input_req[input_name])

            # 填充管线数据（请求参数）
            feed_info = {}
            for input_name in input_selection:
                if input_name not in input_req:
                    feed_info[input_name] = None
                feed_info[input_name] = input_req[input_name]

            # 填充保留字段
            class CookieHandler(object):
                def __init__(self, s, r):
                    self.s = s
                    self.r = r
                def get(self, key, default=None):
                    return self.s.get(key, default)
                def set(self, key, value):
                    self.r.set_cookie(key, value, path="/", domain=None, secure=None, httponly=True, samesite="lax")
                def clear(self, key):
                    self.r.delete_cookie(key)

            feed_info.update(
                {
                    'session_id': session_id, 
                    'ST': input_req.get('ST', None), 
                    'token': input_req.get('token', None), 
                    'username': input_req.get('username', None), 
                    'password': input_req.get('password', None),
                    'headers': req.headers,
                    'cookie': CookieHandler(req.cookies, response)
                }
            )

            # 驱动管线处理流程
            rsp_value = Entity(**feed_info)
            try:
                loop = asyncio.get_running_loop()
                # 服务调用
                for op in gEnv._g_pipeline_op_map[server_name]:
                    # python 3.9+ 支持
                    # rsp_value = await asyncio.to_thread(op, rsp_value)
                    # 兼容python 3.8
                    rsp_value = await loop.run_in_executor(None, op, rsp_value)
                    if isinstance(rsp_value, ReservedRtnType):
                        break
            except Exception as e:  # swallow any exception
                traceback.print_exc()
                raise HTTPException(status_code=500, detail='server pipeline inner error 1') 

            # 检测是否是特殊返回标记
            # 不满足管线完整执行条件，退出/跳转/执行崩溃
            if isinstance(rsp_value, ReservedRtnType):
                if rsp_value.redirect_url is not None:
                    # 是否需要跳转
                    return RedirectResponse(rsp_value.redirect_url)
                if rsp_value.status_code == 500:
                    # 管线执行错误
                    raise HTTPException(status_code=500, detail="server pipeline inner error 2")

                rsp_value = Entity(__response__=rsp_value.data)

            output_selection_types = ServerInfo.pipeline_info[server_name]['output_selection_types']
            output_selection = ServerInfo.pipeline_info[server_name]['output_selection']

            # 重组返回数据
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
                elif output_selection_types[i] == 'json':
                    if b not in rsp_value.__dict__:
                        continue
                    value = rsp_value.__dict__[b]
                    # if not isinstance(value, str):
                    #     value = json.dumps(value, ensure_ascii=False)
                    output_info[b] = value
                else:
                    if b not in rsp_value.__dict__:
                        continue
                    output_info[b] = rsp_value.__dict__[b]

            # 执行正常返回时，返回status_code=200
            response_info = {
                'code': 0,
                'message': 'success',
            }
            if '__response__' in rsp_value.__dict__:
                # 将保留字段信息写入响应中
                response_info.update(
                    rsp_value.__dict__['__response__']
                )
            if response_info['code'] != 0:
                return response_info

            if len(output_info) == 1 and ServerInfo.pipeline_info[server_name]['response_unwarp']:
                output_info = list(output_info.values())[0]
                if not isinstance(output_info, dict):
                    raise HTTPException(status_code=500, detail='server output info parse abnormal') 

            response_info.update(output_info)

            if ServeMixin.isDebug:
                print("--------- Response Info ------------")
                print(response_info)
                print("--------------------------------------")
            return response_info

        @ServerInfo.app.get("/file/download/")
        @ServerInfo.app.get("/file/download")
        async def download(req: Request):
            file_path = req.query_params['file_name']
            return FileResponse(os.path.join(static_folder, file_path))

        @ServerInfo.app.post("/file/upload/")
        @ServerInfo.app.post("/file/upload")
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

    @classmethod
    def web(cls, index=None, name='demo', **kwargs):
        # 设置日志级别
        logging_level = kwargs.get('logging_level', 'INFO')
        logging_level_map = {
            'INFO': logging.INFO,
            'DEBUG': logging.DEBUG,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR
        }
        if logging_level not in logging_level_map:
            logging_level = logging.INFO

        # multipart logger 
        multipart_logger = logging.getLogger('multipart')
        multipart_logger.setLevel(logging_level)
        # requests logger (底层依赖urllib3)
        requests_logger = logging.getLogger("urllib3")
        requests_logger.setLevel(logging_level)

        # 创建处理管线
        source_entry = SourceEntry(
            index=index, 
            cls=cls, 
            name=name, **kwargs
        )

        # 当前激活的管线（设置后，会记录管线所有的算子信息）
        gEnv._g_active_pipeline_name = name

        # 记录管线基本信息到全局
        ServerInfo.index += 1
        ServerInfo.pipeline_name = name

        ServerInfo.pipeline_info.update({
            name: {
                'step_i': kwargs.get('step_i', ServerInfo.index),
            }
        })

        # 更新服务总数
        for server_name, server_config in ServerInfo.pipeline_info.items():
            server_config['step_num'] = len(ServerInfo.pipeline_info)

        return source_entry

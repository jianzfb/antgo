# -*- coding: UTF-8 -*-
# @Time    : 2022/9/12 22:43
# @File    : serve.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import queue
import threading
import concurrent.futures
from antgo.pipeline.functional.entity import Entity
from antgo.pipeline.functional.option import Some
from antgo.pipeline.functional.common.config import *
from antgo.pipeline.application.table.table import *
from antgo.pipeline.functional.mixins.db import *
from antgo.pipeline.functional.common.env import *
from fastapi.responses import RedirectResponse,HTMLResponse, FileResponse
from fastapi import HTTPException
from fastapi import File, UploadFile
from fastapi import Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.staticfiles import StaticFiles

import secrets
import logging
import json
import uuid


class _APIWrapper:
    """
    API Wrapper
    """
    tls = threading.local()

    def __init__(self, index=None, cls=None, name='demo', **kwargs) -> None:
        self._queue = queue.Queue()
        self._cls = cls
        self._name = name

        if index is not None:
            self._index = index if isinstance(index, list) else [index]
        else:
            self._index = index

        self.step_i = kwargs.get('step_i', 0)
        self.step_num = kwargs.get('step_num', 0)

    def feed(self, x) -> None:
        entity = Entity(**x)
        # entity = Some(entity)
        self._queue.put(entity)

    def __iter__(self):
        while True:
            yield self._queue.get()

    def __enter__(self):
        _APIWrapper.tls.placeholder = self

        return self._cls(self).stream()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(_APIWrapper.tls, 'placeholder'):
            _APIWrapper.tls.placeholder = None


class _PipeWrapper:
    """
    Wrapper for execute pipeline as function
    """
    def __init__(self, pipe, placeholder) -> None:
        self._pipe = pipe
        self._placeholder = placeholder
        self._futures = queue.Queue()
        self._executor = threading.Thread(target=self.worker, daemon=True)
        self._executor.start()

    def worker(self):
        while True:
            future = self._futures.get()
            # 驱动管线执行(线程内绑定db)
            result = None
            with thread_session_context():
                result = next(self._pipe)

            future.set_result(result)

    def execute(self, x):
        future = concurrent.futures.Future()
        self._futures.put(future)
        self._placeholder.feed(x)
        return future.result()


async def _decode_content(req):
    from multipart.multipart import parse_options_header
    content_type_header = req.headers.get('Content-Type')
    content_type, _ = parse_options_header(content_type_header)
    if content_type in {b'multipart/form-data'}:
        return await req.form()
    if content_type.startswith(b'image/'):
        return await req.body()
    return (await req.body()).decode()


class ServeMixin:
    """
    Mixin for API serve
    """
    server_app = None
    server_db = None
    pipeline_info = {}
    def serve(self, input=[], output=[], default_config=None, db_config=None, **kwargs):
        """
        Serve the DataFrame as a RESTful API

        Args:
            path (str, optional): API path. Defaults to '/'.
            app (_type_, optional): The FastAPI app the API bind to, will create one if None.

        Returns:
            _type_: the app that bind to
        """      
        api = _APIWrapper.tls.placeholder
        ServeMixin.pipeline_info[api._name] = {
           'api': _PipeWrapper(self._iterable, api)
	    }

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

        ServeMixin.pipeline_info[api._name].update(
            {
                'input_selection': input_selection,
                'input_selection_types': input_selection_types,
                'output_selection': output_selection,
                'output_selection_types': output_selection_types,
                'input_config': input_config,
            }
        )

        # 动态生成orm（基于管线算子需求）
        # 创建/加载数据库
        if ServeMixin.server_db is None and db_config is not None:
            if not os.path.exists('./orm.py'):
                # 生成db orm
                # table_config_info = []
                # for table_info in get_table_info().values():
                #     table_config_info.append(table_info)

                create_db_orm(get_table_info())

            update_db_orm(__import__('orm'))
            ServeMixin.server_db = create_db_session(db_config['db_url'])

        if ServeMixin.server_app is not None:
            return ServeMixin.server_app

        static_folder = './dump'
        from fastapi import FastAPI, Request
        ServeMixin.server_app = FastAPI()
        ServeMixin.server_app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

        @ServeMixin.server_app.post("/{server_name}/execute")
        async def wrapper(server_name: str, req: Request, response: RedirectResponse):
            # 获得session_id
            if(req.session.get("session_id") is None):
                session_id = req.session["session_id"] = secrets.token_urlsafe(32)
                response.set_cookie(key="Authorization", value=session_id)
            session_id = req.session.get("session_id")
            # 上下文绑定：session_id-cookie
            update_context_cookie_info(session_id, response)

            # 解析请求
            input_req = await _decode_content(req)
            try:
                input_req = json.loads(input_req)
            except:
                logging.error('Fail to parsing request.')
                raise HTTPException(status_code=404, detail="请求不合规")

            # 判断服务是否存在
            if server_name not in ServeMixin.pipeline_info:
                raise HTTPException(status_code=404, detail="服务不存在")

            # 转换请求数据
            input_selection_types = ServeMixin.pipeline_info[server_name]['input_selection_types']
            input_selection = ServeMixin.pipeline_info[server_name]['input_selection']
            for input_name, input_type in zip(input_selection, input_selection_types):
                if input_type == 'image':
                    decoded_data = base64.b64decode(input_req[input_name])
                    decoded_data = np.frombuffer(decoded_data, dtype='uint8')
                    input_req[input_name] = cv2.imdecode(decoded_data, 1)
                elif input_type in ['video', 'file']:
                    input_req[input_name] = os.path.join(static_folder, 'image', 'query', input_req[input_name])

            # 填充管线数据（请求参数）
            feed_info = {}
            for input_name in input_selection:
                if input_name not in input_req:
                    raise HTTPException(status_code=403, detail="request params not match")
                feed_info[input_name] = input_req[input_name]
            feed_info.update(
                {'session_id': session_id, 'ST': input_req.get('ST', None), 'token': input_req.get('token', None)}
            )

            # 驱动管线处理流程
            # TODO，加入多线程管线，增强处理能力
            rsp_value = ServeMixin.pipeline_info[server_name]['api'].execute(feed_info)

            # 检查是否需要跳转
            redirect_url = get_context_redirect_info(session_id, None)
            if redirect_url is not None:
                clear_context_env_info(session_id)
                return RedirectResponse(redirect_url)

            # 检查由是否于不满足条件退出，返回退出原因
            exit_condition = get_context_exit_info(session_id, None)
            if exit_condition is not None:
                clear_context_env_info(session_id)
                raise HTTPException(status_code=403, detail=exit_condition) 

            # 检查执行异常
            # 由于计算过程产生BUG
            if rsp_value is None:
                clear_context_env_info(session_id)
                raise HTTPException(status_code=500, detail="pipeline execute abnormal")

            # 清空session_id绑定的上下文
            clear_context_env_info(session_id)
            output_selection_types = ServeMixin.pipeline_info[server_name]['output_selection_types']
            output_selection = ServeMixin.pipeline_info[server_name]['output_selection']

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
                else:
                    output_info[b] = rsp_value.__dict__[b]

            return output_info

        @ServeMixin.server_app.get("/file/download/")
        async def download(req: Request):
            file_path = req.query_params['file_name']
            return FileResponse(os.path.join(static_folder, file_path))

        @ServeMixin.server_app.post("/file/upload/")
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
        ServeMixin.server_app.mount("/", StaticFiles(directory=static_folder), name="static")

        return ServeMixin.server_app

    @classmethod
    def api(cls, index=None, name='serve'):
        return _APIWrapper(index=index, cls=cls, name=name)
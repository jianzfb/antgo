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
from fastapi import HTTPException
import logging
import json


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
        self._lock = threading.Lock()
        self._executor = threading.Thread(target=self.worker, daemon=True)
        self._executor.start()

    def worker(self):
        while True:
            future = self._futures.get()
            try:
                result = next(self._pipe)
            except:
                logging.error('pipeline execute error.')
                result = None

            future.set_result(result)

    def execute(self, x):
        with self._lock:
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
    pipeline_info = {}
    def serve(self, input=[], output=[], default_config=None, **kwargs):
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
            assert (ui_type in ['image', 'video', 'text', 'number', 'file'])

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
        if ServeMixin.server_app is not None:
            return ServeMixin.server_app

        from fastapi import FastAPI, Request
        ServeMixin.server_app = FastAPI()

        @ServeMixin.server_app.post("/{server_name}/execute")
        async def wrapper(server_name: str, req: Request):
            input_req = await _decode_content(req)
            try:
                input_req = json.loads(input_req)
            except:
                logging.error('Fail to parsing request.')
                raise HTTPException(status_code=404, detail="请求不合规")

            if server_name not in ServeMixin.pipeline_info:
                raise HTTPException(status_code=404, detail="服务不存在")

            input_selection_types = DemoMixin.pipeline_info[server_name]['input_selection_types']
            input_selection = DemoMixin.pipeline_info[server_name]['input_selection']
            for input_name, input_type in zip(input_selection, input_selection_types):
                if input_type == 'image':
                    decoded_data = base64.b64decode(input_req[input_name])
                    decoded_data = np.frombuffer(decoded_data, dtype='uint8')
                    input_req[input_name] = cv2.imdecode(decoded_data, 1)
                elif input_type in ['video', 'file']:
                    input_req[input_name] = os.path.join(static_folder, 'image', 'query', input_req[input_name])

            feed_info = {}
            for input_name in input_selection:
                feed_info[input_name] = input_req[input_name]
            feed_info.update(
                {'session_id': uuid.uuid4()}
            )

            rsp_value = ServeMixin.pipeline_info[server_name]['api'].execute(feed_info)
            if rsp_value is None:
                raise HTTPException(status_code=500, detail="管线执行错误")

            output_selection_types = DemoMixin.pipeline_info[server_name]['output_selection_types']
            output_selection = DemoMixin.pipeline_info[server_name]['output_selection']

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

        @DemoMixin.app.get("/file/download/")
        async def download(req: Request):
            file_path = req.query_params['file_name']
            return FileResponse(os.path.join(static_folder, file_path))

        @DemoMixin.app.post("/file/upload/")
        async def upload(file: UploadFile):
            filename = file.filename
            file_size = file.size
            fileid = str(uuid.uuid4())

            unique_filename = f'{fileid}-{filename}'
            os.makedirs(os.path.join(static_folder, 'image', 'query'), exist_ok=True)
            with open(os.path.join(static_folder, 'image', 'query', unique_filename), "wb") as f:
                for chunk in iter(lambda: file.file.read(1024), b''):
                    f.write(chunk)
            return {"fileid": unique_filename}

        return ServeMixin.server_app

    @classmethod
    def api(cls, index=None, name='serve'):
        return _APIWrapper(index=index, cls=cls, name=name)
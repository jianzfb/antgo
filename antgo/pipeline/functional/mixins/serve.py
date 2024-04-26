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
    def serve(self):
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

        if ServeMixin.server_app is not None:
            return ServeMixin.server_app

        from fastapi import FastAPI, Request
        ServeMixin.server_app = FastAPI()

        @ServeMixin.server_app.post("/{serve_name}")
        async def wrapper(serve_name: str, req: Request):
            req = await _decode_content(req)
            try:
                req = json.loads(req)
            except:
                logging.error('Fail to parsing request.')
                raise HTTPException(status_code=404, detail="请求不合规")

            if serve_name not in ServeMixin.pipeline_info:
                raise HTTPException(status_code=404, detail="服务不存在")

            rsp = ServeMixin.pipeline_info[serve_name]['api'].execute(req)
            if rsp is None:
                raise HTTPException(status_code=500, detail="管线执行错误")

            return rsp.__dict__

        return ServeMixin.server_app

    @classmethod
    def api(cls, index=None, name='serve'):
        return _APIWrapper(index=index, cls=cls, name=name)
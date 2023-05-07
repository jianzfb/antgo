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


class _APIWrapper:
    """
    API Wrapper
    """
    tls = threading.local()

    def __init__(self, index=None, cls=None) -> None:
        self._queue = queue.Queue()
        self._cls = cls

        if index is not None:
            self._index = index if isinstance(index, list) else [index]
        else:
            self._index = index

    def feed(self, x) -> None:
        if self._index is None:
            entity = x
        else:
            index = self._index
            if len(index) == 2:
                input_selection, _ = index
            else:
                input_selection = index

            if type(input_selection) == str:
                input_selection = [input_selection]

            if len(input_selection) == 1:
                x = (x, )

            data = dict(zip(input_selection, x))
            entity = Entity(**data)
        entity = Some(entity)
        self._queue.put(entity)

    def __iter__(self):
        while True:
            yield self._queue.get()

    def __enter__(self):
        _APIWrapper.tls.place_holder = self
        return self._cls(self).stream()

    def __exit__(self, exc_type, exc_value, traceback):
        if hasattr(_APIWrapper.tls, 'place_holder'):
            _APIWrapper.tls.place_holder = None


class _PipeWrapper:
    """
    Wrapper for execute pipeline as function
    """

    def __init__(self, pipe, place_holder) -> None:
        self._pipe = pipe
        self._place_holder = place_holder
        self._futures = queue.Queue()
        self._lock = threading.Lock()
        self._executor = threading.Thread(target=self.worker, daemon=True)
        self._executor.start()

    def worker(self):
        while True:
            future = self._futures.get()
            result = next(self._pipe)
            future.set_result(result)

    def execute(self, x):
        with self._lock:
            future = concurrent.futures.Future()
            self._futures.put(future)
            self._place_holder.feed(x)
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

    def serve(self, path='/', app=None):
        """
        Serve the DataFrame as a RESTful API

        Args:
            path (str, optional): API path. Defaults to '/'.
            app (_type_, optional): The FastAPI app the API bind to, will create one if None.

        Returns:
            _type_: the app that bind to
        """
        if app is None:
            from fastapi import FastAPI, Request
            app = FastAPI()
        else:
            from fastapi import Request

        api = _APIWrapper.tls.place_holder

        pipeline = _PipeWrapper(self._iterable, api)

        @app.post(path)
        async def wrapper(req: Request):
            nonlocal pipeline
            req = await _decode_content(req)
            rsp = pipeline.execute(req)
            if rsp.is_empty():
                return rsp.get()
            return rsp.get()

        return app

    @classmethod
    def api(cls, index=None):
        return _APIWrapper(index=index, cls=cls)
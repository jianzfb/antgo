# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : data.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import threading
import uuid


class AttrMap(object):
    def __init__(self, **kwarg):
        self._info = kwarg

    def __getattr__(self, name):
        return self._info.get(name, None)

    def all(self):
        return {name: obj.get() for name, obj in self._info.items()}


class DataObj(object):
    data_factory = {}
    def __init__(self, name="", default=None, proxy=None):
        self.name = name
        self.value = default
        self.callback = []
        self.proxy = proxy

    def set(self, value):
        if self.proxy is None:
            self.value = value
            for callback in self.callback:
                callback(value)
            return

        self.proxy.set(self.name, value)

    def get(self):
        if self.proxy is None:
            return self.value

        return self.proxy.get(self.name)

    def watch(self, callback):
        if self.proxy is None:
            self.callback.append(callback)
            return

        self.proxy.watch(self.name, callback)

    @classmethod
    def getorcreate(cls, name, default, proxy):
        if name not in cls.data_factory:
            cls.data_factory[name] = cls(name, default, proxy)

        return cls.data_factory[name]


class DataS(object):
    def __init__(self, name=None, default=None, proxy=None):
        if name is None:
            name = str(uuid.uuid4())
        self.data_obj = DataObj.getorcreate(name, default, proxy)

    def set(self, value):
        self.data_obj.set(value)

    def get(self):
        return self.data_obj.get()

    def watch(self, callback):
        self.data_obj.watch(callback)


class DataOp(object):
    def __init__(self, source=None):
        self.is_init = False
        self.lock = threading.Lock()
        self.source = source
        self.data_map = {}
        self._update_funcs = {}

    def get(self, name):
        return self.data_map.get(name, None)

    def set(self, name, value):
        self.lock.acquire()
        if name in self._update_funcs:
            for callback in self._update_funcs[name]:
                callback(value)
        self.data_map[name] = value
        self.lock.release()

    def init(self):
        # 初始化
        # TODO，添加初始化内容
        self.is_init = True

    def watch(self, name, callback):
        # 更新触发回调函数
        self.lock.acquire()
        if name not in self._update_funcs:
            self._update_funcs[name] = []
        self._update_funcs[name].append(callback)
        self.lock.release()

    def __call__(self, *args, **kwds):
        indexs = [self._index] if isinstance(self._index, str) else self._index
        data_proxys = [DataS(name=name, proxy=self) for name in indexs]
        if not self.is_init:
            data_config = {}
            for data_proxy, data_name in zip(data_proxys, indexs):
                data_config[data_name] = data_proxy
            self.source(**data_config)
        self.is_init = True
        if len(indexs) == 1:
            return data_proxys[0]
        else:
            return data_proxys
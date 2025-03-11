# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : data.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import threading


class DataProxy(object):
    def __init__(self, proxy, name):
        self.proxy = proxy
        self.name = name

    def set(self, value):
        self.proxy.set(self.name, value)

    def get(self):
        return self.proxy.get(self.name)

    def update(self, callback):
        self.proxy.update(self.name, callback)


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

    def update(self, name, callback):
        # 更新触发回调函数
        self.lock.acquire()
        if name not in self._update_funcs:
            self._update_funcs[name] = []
        self._update_funcs[name].append(callback)
        self.lock.release()

    def __call__(self, *args, **kwds):
        data_proxys = [DataProxy(self, name) for name in self._index]
        if not self.is_init:
            data_config = {}
            for data_proxy, data_name in zip(data_proxys, self._index):
                data_config[data_name] = data_proxy
            self.source(**data_config)
        self.is_init = True
        if len(self._index) == 1:
            return data_proxys[0]
        else:
            return data_proxys
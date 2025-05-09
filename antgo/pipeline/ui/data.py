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
        self.callback = {}
        self.proxy = proxy
        self.lock = threading.Lock()
        self.waiting_cancel_list = []
        self.waiting_watch_list = []

    def set(self, value):
        self.lock.acquire()
        # 检查是否需要添加新回调函数
        self._watch()
        # 检查是否需要删除回调函数
        self._cancel()
        self.lock.release()

        # 触发回调函数
        if self.proxy is None:
            self.value = value
            for callback_name, callback_func in self.callback.items():
                callback_func(value)
            return

        self.proxy.set(self.name, value)

    def get(self):
        if self.proxy is None:
            return self.value

        return self.proxy.get(self.name)

    def _watch(self):
        for callback_func, callback_name in self.waiting_watch_list:
            if self.proxy is None:
                self.callback[callback_name] = callback_func
                return

            self.proxy.watch(self.name, callback_func, callback_name)

        self.waiting_watch_list = []

    def watch(self, callback_func, callback_name):
        self.lock.acquire()
        self.waiting_watch_list.append((callback_func, callback_name))
        self.lock.release()

    def _cancel(self):
        for callback_name in self.waiting_cancel_list:
            if self.proxy is None:
                self.callback.pop(callback_name)
                return

            self.proxy.cancel(self.name, callback_name)

        self.waiting_cancel_list = []

    def cancel(self, callback_name):
        self.lock.acquire()
        self.waiting_cancel_list.append(callback_name)
        self.lock.release()

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

    def config_proxy(self, proxy):
        self.data_obj.proxy = proxy

    def set(self, value):
        self.data_obj.set(value)

    def get(self):
        return self.data_obj.get()

    def watch(self, callback_func, callback_name=None):
        if callback_name is None:
            callback_name = str(uuid.uuid4())
        self.data_obj.watch(callback_func, callback_name)

    def cancel(self, callback_name):
        self.data_obj.cancel(callback_name)


class DataOp(object):
    def __init__(self, data_gen=None):
        self.is_init = False
        self.data_gen = data_gen
        self.data_map = {}
        self._update_funcs = {}

    def get(self, name):
        return self.data_map.get(name, None)

    def set(self, name, value):
        # 触发绑定的回调
        if name in self._update_funcs:
            for callback_name, callback_func in self._update_funcs[name].items():
                callback_func(value)
        # 更新数据记录
        self.data_map[name] = value

    def init(self):
        # 初始化
        # TODO，添加初始化内容
        self.is_init = True

    def watch(self, name, callback_func, callback_name):
        # 更新触发回调函数
        if name not in self._update_funcs:
            self._update_funcs[name] = {}
        self._update_funcs[name][callback_name] = callback_func

    def cancel(self, name, callback_name):
        # 取消回调函数
        if name not in self._update_funcs:
            return

        if callback_name in self._update_funcs[name]:
            self._update_funcs[name].pop(callback_name)

    def __call__(self, *args, **kwds):
        indexs = [self._index] if isinstance(self._index, str) else self._index
        data_s = [DataS(name=name, proxy=self) for name in indexs]
        if not self.is_init:
            data_config = {}
            for data_proxy, data_name in zip(data_s, indexs):
                data_config[data_name] = data_proxy
            self.data_gen(**data_config)
        self.is_init = True

        if len(indexs) == 1:
            return data_s[0]
        else:
            return tuple(data_s)
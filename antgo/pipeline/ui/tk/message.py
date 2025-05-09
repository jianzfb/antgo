# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : messagebox.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk
from tkinter import messagebox
from antgo.pipeline.ui.data import DataS, AttrMap

class MessageOp(object):
    def __init__(self, win_type="info", text="", title="message"):
        self._attr = AttrMap(
            win_type=DataS(default=win_type) if not isinstance(win_type, DataS) else win_type,
            title=DataS(default=title) if not isinstance(title, DataS) else title
        )
        self._message_info = DataS(default=text) if not isinstance(text, DataS) else text
        self._root = None

    @property
    def element(self):
        return None

    @property
    def attr(self):
        return self._attr

    def open(self, value):
        # 需要在UI线程中执行
        if not isinstance(value, str):
            print("Not support non str in messagebox")
            return

        if self._root is None:
            print("Mesage must be child in parent")
            return

        title = self._message_info.get()
        if self._attr.win_type.get() == 'error':
            self._root.after(0, lambda: messagebox.showerror(title=title, message=value))
        elif self._attr.win_type.get() == 'warning':
            self._root.after(0, lambda: messagebox.showwarning(title=title, message=value))
        else:
            self._root.after(0, lambda: messagebox.showinfo(title=title, message=value))

    def __call__(self, *args, **kwds):
        # 绑定root，需要依靠root的线程来触发message box打开
        self._root = args[0].element

        if len(args) > 1:
            # 使用用户指定数据源绑定
            self._message_info = args[1]
        self._message_info.watch(
            lambda value: self.open(value)
        )
        return self._message_info
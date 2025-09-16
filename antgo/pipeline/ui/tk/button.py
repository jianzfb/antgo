# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : button.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from antgo.pipeline.ui.data import DataS, AttrMap


class ButtonOp(object):
    def __init__(self, func, text="OK", font=('微软雅黑', 10, 'italic'), gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):  
        self._func = func
        self._btn = None
        self._attr = AttrMap(
            text    =DataS(default=text) if not isinstance(text, DataS) else text,
            gridx   =DataS(default=gridx) if not isinstance(gridx, DataS) else gridx,
            gridy   =DataS(default=gridy) if not isinstance(gridy, DataS) else gridy,
            spanx   =DataS(default=spanx) if not isinstance(spanx, DataS) else spanx,
            spany   =DataS(default=spany) if not isinstance(spany, DataS) else spany,
            padx    =DataS(default=padx) if not isinstance(padx, DataS) else padx,
            pady    =DataS(default=pady) if not isinstance(pady, DataS) else pady,
            stick   =DataS(default=stick) if not isinstance(stick, DataS) else stick
        )
        self._font = font

    def click(self, *args, **kwargs):
        self._func(*args, **kwargs)

    @property
    def element(self):
        return self._btn

    @property
    def attr(self):
        return self._attr

    def __getattr__(self, name):
        if name not in ['text']:
            return super(ButtonOp, self).__getattribute__(name)
        return None

    def __setattr__(self, name, value):
        if name not in ['text']:
            super(ButtonOp, self).__setattr__(name, value)
            return
        if name == 'text':
            self._btn.config(text=value)

    def __call__(self, *args, **kwds):
        # 格式
        # args = [parent_node, data, ...]
        parent_node = args[0].element
        self._btn = tk.Button(parent_node, text=self.attr.text.get(), command=lambda: self.click(*args[1:]), font=self._font)
        self.attr.text.watch(lambda value: self._btn.config(text=value))
    
        layout_params = {
            'row': self.attr.gridy.get(), 
            'column': self.attr.gridx.get(),
            'padx': self.attr.padx.get(),
            'pady': self.attr.pady.get(),
            'rowspan': self.attr.spany.get(),
            'columnspan': self.attr.spanx.get(),
            'sticky': self.attr.stick.get()
        }

        self._btn.grid(**layout_params)
        return self

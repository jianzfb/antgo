# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : button.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from tkinter import messagebox

class ButtonOp(object):
    def __init__(self, func, text="OK", gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):  
        self._func = func
        self._text = text
        self._gridx = gridx
        self._gridy = gridy
        self._spanx = spanx
        self._spany = spany
        self._padx = padx
        self._pady = pady
        self._stick = stick

        self._btn = None

    def click(self, *args, **kwargs):
        self._func(*args, **kwargs)

    def element(self):
        return self._btn

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
        parent_node = args[0].element()
        self._btn = tk.Button(parent_node, text=self._text, command=lambda: self.click(*args[1:]))
        layout_params = {
            'row': self._gridy, 
            'column': self._gridx,
            'padx': self._padx,
            'pady': self._pady,
            'rowspan': self._spany,
            'columnspan': self._spanx,
            'sticky': self._stick
        }

        self._btn.grid(**layout_params)
        return self

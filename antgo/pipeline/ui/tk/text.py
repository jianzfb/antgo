# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : label.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from tkinter import messagebox

class TextOp(object):
    def __init__(self, text="", gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):  
        self._text = text
        self._gridx = gridx
        self._gridy = gridy
        self._spanx = spanx
        self._spany = spany
        self._padx = padx
        self._pady = pady
        self._stick = stick

        self._entry = None

    def element(self):
        return self._entry

    def __getattr__(self, name):
        if name not in ['text']:
            return super(TextOp, self).__getattribute__(name)
        if name == 'text':
            return self._entry.get()
        
        return None

    def __setattr__(self, name, value):
        if name not in ['text']:
            super(TextOp, self).__setattr__(name, value)
            return
        if name == 'text':
            self._entry.insert(0, value)

    def __call__(self, *args, **kwds):
        parent_node = args[0].element()
        self._entry = tk.Entry(parent_node)
        layout_params = {
            'row': self._gridy, 
            'column': self._gridx,
            'padx': self._padx,
            'pady': self._pady,
            'rowspan': self._spany,
            'columnspan': self._spanx,
            'sticky': self._stick
        }

        self._entry.grid(**layout_params)
        return self
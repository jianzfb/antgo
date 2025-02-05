# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : label.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import PhotoImage
import numpy as np


class LabelOp(object):
    def __init__(self, text=None, gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):  
        self._text = text
        self._gridx = gridx
        self._gridy = gridy
        self._spanx = spanx
        self._spany = spany
        self._padx = padx
        self._pady = pady
        self._stick = stick

        self._label = None

    def element(self):
        return self._label

    def __getattr__(self, name):
        if name not in ['text']:
            return super(LabelOp, self).__getattribute__(name)
        return None

    def __setattr__(self, name, value):
        if name not in ['text', 'image']:
            super(LabelOp, self).__setattr__(name, value)
            return
        if name == 'text':
            self._label.config(text=value)
        if name == 'image':
            # 支持numpy, pil, path
            if isinstance(value, np.ndarray):
                value = Image.fromarray(value)
                value = ImageTk.PhotoImage(value)
            elif isinstance(value, Image.Image):
                value = ImageTk.PhotoImage(value)
            else:
                value = PhotoImage(file=value)
            self._label.config(image=value)

    def __call__(self, *args, **kwds):
        parent_node = args[0].element()
        params = {}
        if self._text is not None:
            params['text'] = self._text
        self._label = tk.Label(parent_node, **params)
        layout_params = {
            'row': self._gridy, 
            'column': self._gridx,
            'padx': self._padx,
            'pady': self._pady,
            'rowspan': self._spany,
            'columnspan': self._spanx,
            'sticky': self._stick
        }

        self._label.grid(**layout_params)
        return self
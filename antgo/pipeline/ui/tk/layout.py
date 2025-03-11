# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : layout.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk

class LayoutOp(object):
    def __init__(self, gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick='', width=1.0, height=1.0, bg='white', thickness=1):
        self._gridx = gridx
        self._gridy = gridy
        self._spanx = spanx
        self._spany = spany
        self._padx = padx
        self._pady = pady
        self._stick = stick
        self._width = width
        self._height = height
        self._bg = bg
        self._thickness = thickness

    def element(self):
        return self._layout
    
    def __call__(self, *args, **kwds):
        parent_node = args[0].element()
        width = self._width * parent_node.winfo_width()
        height = self._height * parent_node.winfo_height()
        params = {
            'width': width, 
            'height': height,
            'highlightbackground': self._bg,
            'highlightthickness': self._thickness
        }
        self._layout = tk.Frame(parent_node, **params)

        layout_params = {
            'row': self._gridy, 
            'column': self._gridx,
            'padx': self._padx,
            'pady': self._pady,
            'rowspan': self._spany,
            'columnspan': self._spanx,
            'sticky': self._stick
        }
        self._layout.grid(**layout_params)
        return self
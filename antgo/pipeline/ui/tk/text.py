# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : label.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from antgo.pipeline.ui.data import DataS, AttrMap


class TextOp(object):
    def __init__(self, text="", gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):  
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
        self._attr.text.config_proxy(self)
        self._entry = None

    @property
    def element(self):
        return self._entry

    @property
    def attr(self):
        return self._attr

    def get(self, _):
        if self._entry is None:
            return ''
        return self._entry.get()

    def set(self, _, value):
        if self._entry is None:
            return
        self._entry.insert(0, value)

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
        parent_node = args[0].element
        params = {
            'text': self._attr.text.get()
        }
        self._entry = tk.Entry(parent_node, **params)
        
        layout_params = {
            'row': self._attr.gridy.get(), 
            'column': self._attr.gridx.get(),
            'padx': self._attr.padx.get(),
            'pady': self._attr.pady.get(),
            'rowspan': self._attr.spany.get(),
            'columnspan': self._attr.spanx.get(),
            'sticky': self._attr.stick.get()
        }

        self._entry.grid(**layout_params)
        return self
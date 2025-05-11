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
from antgo.pipeline.ui.data import DataS, AttrMap


class LabelOp(object):
    def __init__(self, text=None, image=None, gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick=''):
        self._attr = AttrMap(
            text    =DataS(default=text) if not isinstance(text, DataS) else text,
            gridx   =DataS(default=gridx) if not isinstance(gridx, DataS) else gridx,
            gridy   =DataS(default=gridy) if not isinstance(gridy, DataS) else gridy,
            spanx   =DataS(default=spanx) if not isinstance(spanx, DataS) else spanx,
            spany   =DataS(default=spany) if not isinstance(spany, DataS) else spany,
            padx    =DataS(default=padx) if not isinstance(padx, DataS) else padx,
            pady    =DataS(default=pady) if not isinstance(pady, DataS) else pady,
            stick   =DataS(default=stick) if not isinstance(stick, DataS) else stick,
            image   =DataS(default=image) if not isinstance(image, DataS) else image
        )
        self._label = None
        self._image = None

    @property
    def element(self):
        return self._label

    @property
    def attr(self):
        return self._attr

    def setImage(self, value):
        if value is None:
            return
        if isinstance(value, np.ndarray):
            value = Image.fromarray(value)
            value = ImageTk.PhotoImage(value)
        elif isinstance(value, Image.Image):
            value = ImageTk.PhotoImage(value)
        else:
            value = PhotoImage(file=value)
        
        self._image = value
        self._label.config(image=self._image)

    def __call__(self, *args, **kwds):
        parent_node = args[0].element
        params = {}
        if self._attr.text.get() is not None:
            params['text'] = self._attr.text.get()

        if self._attr.image.get() is not None:
            self._image = ImageTk.PhotoImage(Image.open(self._attr.image.get()))
            params['image'] = self._image

        self._label = tk.Label(parent_node, **params)
        self._attr.text.watch(lambda value: self._label.config(text=value))
        self._attr.image.watch(lambda value: self.setImage(value))

        layout_params = {
            'row': self._attr.gridy.get(), 
            'column': self._attr.gridx.get(),
            'padx': self._attr.padx.get(),
            'pady': self._attr.pady.get(),
            'rowspan': self._attr.spany.get(),
            'columnspan': self._attr.spanx.get(),
            'sticky': self._attr.stick.get()
        }

        self._label.grid(**layout_params)
        return self
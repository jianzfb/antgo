# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : layout.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk
from antgo.pipeline.ui.data import DataS, AttrMap


class LayoutOp(object):
    def __init__(self, gridx=0, gridy=0, spanx=1, spany=1, padx=0, pady=0, stick='', width=1.0, height=1.0, bg='white', thickness=1):
        self._attr = AttrMap(
            gridx=DataS(default=gridx) if not isinstance(gridx, DataS) else gridx,
            gridy=DataS(default=gridy) if not isinstance(gridy, DataS) else gridy,
            spanx=DataS(default=spanx) if not isinstance(spanx, DataS) else spanx,
            spany=DataS(default=spany) if not isinstance(spany, DataS) else spany,
            padx=DataS(default=padx) if not isinstance(padx, DataS) else padx,
            pady=DataS(default=pady) if not isinstance(pady, DataS) else pady,
            stick=DataS(default=stick) if not isinstance(stick, DataS) else stick,
            width=DataS(default=width) if not isinstance(width, DataS) else width,
            height=DataS(default=height) if not isinstance(height, DataS) else height,
            bg=DataS(default=bg) if not isinstance(bg, DataS) else bg,
            thickness=DataS(default=thickness) if not isinstance(thickness, DataS) else thickness
        )
        self._layout = None

    @property
    def element(self):
        return self._layout
    
    @property
    def attr(self):
        return self._attr

    def __call__(self, *args, **kwds):
        parent_node = args[0].element
        width = self._attr.width.get() * parent_node.winfo_width()
        height = self._attr.height.get() * parent_node.winfo_height()

        # for i in range(self._attr.gridx.get(), self._attr.gridx.get()+self._attr.padx.get(), 1):
        #    parent_node.grid_columnconfigure(i, weight=1)

        params = {
            # 'width': width, 
            # 'height': height,
            'highlightbackground': self._attr.bg.get(),
            'highlightthickness': self._attr.thickness.get()
        }
        self._layout = tk.Frame(parent_node, **params)

        layout_params = {
            'row': self._attr.gridy.get(), 
            'column': self._attr.gridx.get(),
            'padx': self._attr.padx.get(),
            'pady': self._attr.pady.get(),
            'rowspan': self._attr.spany.get(),
            'columnspan': self._attr.spanx.get(),
            'sticky': tk.NSEW
        }
        self._layout.grid(**layout_params)

        return self
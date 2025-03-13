# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : canvas.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from antgo.pipeline.ui.smart.data import DataS, AttrMap


class CanvasOp(object):
    def __init__(self, elem_type='line', **kwargs):
        self._entry = None
        self.elem_type = elem_type
        self._attr = None
        if elem_type == 'line':
            fill_attr = kwargs.get('fill', 'blue')
            width_attr = kwargs.get('width', 1)
            arrow_attr = kwargs.get("arrow", 'none')
            capstyle_attr = kwargs.get("capstyle", 'butt')
            joinstyle_attr = kwargs.get('joinstyle', 'miter')
            self._attr = AttrMap(
                fill    =DataS(default=fill_attr) if not isinstance(fill_attr, DataS) else fill_attr,
                width   =DataS(default=width_attr) if not isinstance(width_attr, DataS) else width_attr,
                arrow   =DataS(default=arrow_attr) if not isinstance(arrow_attr, DataS) else arrow_attr,
                capstyle=DataS(default=capstyle_attr) if not isinstance(capstyle_attr, DataS) else capstyle_attr,
                joinstyle   =DataS(default=joinstyle_attr) if not isinstance(joinstyle_attr, DataS) else joinstyle_attr,
            )

        self.callback_map = {
            'line': self.drawLine,
            'rect': self.drawRect,
            'text': self.drawText,
            'circle': self.drawCircle
        }

    @property
    def element(self):
        return self._entry

    @property
    def attr(self):
        return self._attr

    def drawLine(self, data):
        # [[[x0,y0],[x1,y1],...]]
        # step 1: clear
        self._entry.delete('all')

        # step 2: draw
        for line_i in range(len(data)):
            line_data = data[line_i]
            for point_i in range(len(line_data) - 1):
                point_data = line_data[point_i]
                next_point_data = line_data[point_i + 1]
                self._entry.create_line(int(point_data[0]), int(point_data[1]), int(next_point_data[0]), int(next_point_data[1]), **self._attr.all())

    def drawRect(self, data):
        # [[],[],...]
        pass
    def drawText(self, data):
        # ["", "", ...]
        pass
    def drawCircle(self, data):
        # [[],[],...]
        pass

    def __call__(self, *args, **kwds):
        # 格式
        # args = [parent_node, data, ...]
        parent_node = args[0].element
        self._entry = tk.Canvas(parent_node)
        self._entry.pack(fill=tk.BOTH, expand=True)
        for arg in args[1:]:
            # 数据监听，动态重绘
            arg.watch(self.callback_map[self.elem_type])

        return self


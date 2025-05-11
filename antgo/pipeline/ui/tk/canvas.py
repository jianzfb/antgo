# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : canvas.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
import tkinter as tk  
from antgo.pipeline.ui.data import DataS, AttrMap


class CanvasOp(object):
    def __init__(self, elem_type='line', **kwargs):
        self.elem_type = elem_type
        if self.elem_type not in ['line', 'rectangle', 'text', 'oval', 'any']:
            raise ValueError("elem_type must be one of ['line', 'rectangle', 'text', 'oval']")
        self._parent = None
        self._attr = None
        self._entry = None

        fill_attr = kwargs.get('fill', 'blue')
        width_attr = kwargs.get('width', 1)
        arrow_attr = kwargs.get("arrow", 'none')
        capstyle_attr = kwargs.get("capstyle", 'butt')
        joinstyle_attr = kwargs.get('joinstyle', 'miter')
        outline_attr = kwargs.get('outline', 'black')
        anchor_attr = kwargs.get('anchor', 'center')    # center, sw
        if anchor_attr == 'center':
            anchor_attr = tk.CENTER 

        if self.elem_type == 'line':
            self._attr = AttrMap(
                fill    =DataS(default=fill_attr) if not isinstance(fill_attr, DataS) else fill_attr,
                width   =DataS(default=width_attr) if not isinstance(width_attr, DataS) else width_attr,
                arrow   =DataS(default=arrow_attr) if not isinstance(arrow_attr, DataS) else arrow_attr,
                capstyle=DataS(default=capstyle_attr) if not isinstance(capstyle_attr, DataS) else capstyle_attr,
                joinstyle   =DataS(default=joinstyle_attr) if not isinstance(joinstyle_attr, DataS) else joinstyle_attr,
            )
        elif self.elem_type == 'rectangle':
            self._attr = AttrMap(
                fill    =DataS(default=fill_attr) if not isinstance(fill_attr, DataS) else fill_attr,
                width   =DataS(default=width_attr) if not isinstance(width_attr, DataS) else width_attr,
                outline =DataS(default=outline_attr) if not isinstance(outline_attr, DataS) else outline_attr,
            )
        elif self.elem_type == 'text':
            self._attr = AttrMap(
                fill    =DataS(default=fill_attr) if not isinstance(fill_attr, DataS) else fill_attr,
                anchor  =DataS(default=anchor_attr) if not isinstance(anchor_attr, DataS) else anchor_attr,
            )
        elif self.elem_type == 'oval':
            self._attr = AttrMap(
                fill    =DataS(default=fill_attr) if not isinstance(fill_attr, DataS) else fill_attr,
                width   =DataS(default=width_attr) if not isinstance(width_attr, DataS) else width_attr,
                outline =DataS(default=outline_attr) if not isinstance(outline_attr, DataS) else outline_attr,
            )

        self.data = kwargs.get('data', None)
        if self.data is not None:
            self.data = DataS(default=self.data) if not isinstance(self.data, DataS) else self.data

        self.callback_map = {
            'line': self.drawLine,
            'rectangle': self.drawRect,
            'text': self.drawText,
            'oval': self.drawOval,
            'any': self.drawAny,
        }
        self.callback_func = self.callback_map[self.elem_type]

    @property
    def element(self):
        return self._entry

    @property
    def attr(self):
        return self._attr

    def drawLine(self, data):
        # [[[x0,y0],[x1,y1],...]]
        # step 1: clear
        if len(data) > 0:
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
        if len(data) > 0:
            self._entry.delete('all')

        for rect_i in range(len(data)):
            rect_data = data[rect_i]
            x0, y0, x1, y1 = rect_data
            self._entry.create_rectangle(int(x0), int(y0), int(x1), int(y1), **self._attr.all())

    def drawText(self, data):
        # [[x0,y0,""], [x0,y0,""], ...]
        if len(data) > 0:
            self._entry.delete('all')

        for text_i in range(len(data)):
            print(data)
            x0, y0, text_data = data[text_i]
            self._entry.create_text(int(x0), int(y0), text=text_data, **self._attr.all())

    def drawOval(self, data):
        # [[],[],...]
        if len(data) > 0:
            self._entry.delete('all')

        for oval_i in range(len(data)):
            oval_data = data[oval_i]
            x0, y0, x1, y1 = oval_data
            self._entry.create_oval(int(x0), int(y0), int(x1), int(y1), **self._attr.all())

    def drawAny(self, data):
        # {
        #     "line": [],
        #     "rectangle": [],
        # }
        pass

    def __call__(self, *args, **kwds):
        # 格式
        # args = [parent_node, data, ...]
        self._entry = tk.Canvas(args[0].element)
        self._entry.pack(fill=tk.BOTH, expand=True)

        for arg in args[1:]:
            # 数据监听，动态重绘
            arg.watch(self.callback_func)

        if self.data is not None:
            # 数据监听，动态重绘
            self.data.watch(self.callback_func)

        return self


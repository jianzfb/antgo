# -*- coding: UTF-8 -*-
# @Time    : 2024/11/27 22:42
# @File    : field.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import cv2
from antgo.pipeline.functional.mixins.db import *
from sqlalchemy import and_, or_


class FieldOp(object):
    def __init__(self, table, data):
        self.table = table
        self.data = data

    def __call__(self, *args):
        if '/' not in self.data:
            # 表内属性
            return getattr(args[0], self.data)
        else:
            # 跨表属性
            related_obj,related_field = data_name.split('/')
            related_obj = getattr(args[0], related_obj)
            if related_obj is not None:
                return getattr(related_obj, related_field)

            return None

